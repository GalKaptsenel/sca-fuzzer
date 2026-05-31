"""
File: x86 implementation of the test case generator

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import abc
import math
import random
import copy
import struct
from itertools import chain
from typing import Any, List, Tuple, Optional, Type, Callable, Set, Dict
from dataclasses import dataclass, field

from ..config import CONF
from ..isa_loader import InstructionSet
from ..interfaces import TestCase, Operand, Instruction, BasicBlock, Function, InstructionSpec, \
    GeneratorException, RegisterOperand, ImmediateOperand, MAIN_AREA_SIZE, FAULTY_AREA_SIZE, \
    MemoryOperand, AgenOperand, OT, OperandSpec, CondOperand, TargetDesc
from ..generator import ConfigurableGenerator, RandomGenerator, Pass, Printer
from ..config import CONF
from .aarch64_target_desc import Aarch64TargetDesc
from .aarch64_elf_parser import Aarch64ElfParser
from .aarch64_contract_executor import InstrTraceEntry, ContractExecutionResult


class FaultFilter:

    def __init__(self) -> None:
        self.div_by_zero: bool = 'div-by-zero' in CONF.generator_faults_allowlist
        self.div_overflow: bool = 'div-overflow' in CONF.generator_faults_allowlist
        self.non_canonical_access: bool = 'non-canonical-access' in CONF.generator_faults_allowlist
        self.u2k_access: bool = 'user-to-kernel-access' in CONF.generator_faults_allowlist


class Aarch64Generator(ConfigurableGenerator, abc.ABC):
    faults: FaultFilter
    target_desc: Aarch64TargetDesc

    def __init__(self, instruction_set: InstructionSet, seed: int):
        super(Aarch64Generator, self).__init__(instruction_set, seed)
        self.target_desc = Aarch64TargetDesc()
        self.elf_parser = Aarch64ElfParser(self.target_desc)
        self.faults = FaultFilter()

        # configure instrumentation passes
        self.passes = [
#            Aarch64PatchUndefinedLoadsPass(self.target_desc),
            Aarch64PatchUndefinedLoadsStoresPass(self.target_desc),
#            Aarch64SandboxPass(),
#            Aarch64DsbSyPass(),
        ]

        self.printer = Aarch64Printer(self.target_desc)

    def get_return_instruction(self) -> Instruction:
        return Instruction("ret", False, "", True, template="RET")

    def get_unconditional_jump_instruction(self) -> Instruction:
        return Instruction("b", False, "UNCOND_BR", True, template="B {label}")

    def get_elf_data(self, test_case: TestCase, obj_file: str) -> None:
        self.elf_parser.parse(test_case, obj_file)


class Aarch64DsbSyPass(Pass):

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:
                insertion_points = []
                for instr in bb:
                    # make a copy to avoid infinite insertions
                    insertion_points.append(instr)

                for instr in insertion_points:
                    bb.insert_after(instr, Instruction("DSB SY", True, template="DSB SY"))


# Single-dest load with base writeback (post/pre-index).
# dest == base → UNPREDICTABLE.
_LDR_WRITEBACK = frozenset({
    "ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw",
})

# Two-dest load, all address forms.
# dest0 == dest1 → UNPREDICTABLE.
# Post/pre-index additionally: dest0|dest1 == base → UNPREDICTABLE.
_LDP_ANY = frozenset({"ldp", "ldpsw"})

# Exclusive two-dest load (always has a base, no explicit writeback field,
# but the base register must not alias either destination).
_LDXP = frozenset({"ldxp", "ldaxp"})

# Store-exclusive: status register must not alias data or base.
_STXR = frozenset({"stxr", "stlxr", "stxrb", "stlxrb", "stxrh", "stlxrh"})

# Store-exclusive pair: same as STXR plus data0==data1 check on status.
_STXP = frozenset({"stxp", "stlxp"})

# Store pair with writeback: src0|src1 == base → UNPREDICTABLE.
_STP_WRITEBACK = frozenset({"stp"})


class Aarch64PatchUndefinedLoadsStoresPass(Pass):
    """
    Patch all UNPREDICTABLE register-collision constraints for AArch64
    load and store instructions.
    """

    def __init__(self, target_desc) -> None:
        self.target_desc: Aarch64TargetDesc = target_desc
        super().__init__()

    # ------------------------------------------------------------------
    # Pass entry point
    # ------------------------------------------------------------------

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:
                for inst in bb:
                    self._patch_instruction(inst)

    def _patch_instruction(self, inst: Instruction) -> None:
        name = inst.name.lower()

        if any(name.startswith(m) for m in _LDR_WRITEBACK):
            self._patch_ldr_writeback(inst)

        elif any(name.startswith(m) for m in _LDP_ANY):
            self._patch_ldp(inst)

        elif any(name.startswith(m) for m in _LDXP):
            self._patch_ldxp(inst)

        elif any(name.startswith(m) for m in _STXR):
            self._patch_stxr(inst)

        elif any(name.startswith(m) for m in _STXP):
            self._patch_stxp(inst)

        elif any(name.startswith(m) for m in _STP_WRITEBACK):
            self._patch_stp_writeback(inst)

    def _patch_ldr_writeback(self, inst: Instruction) -> None:
        if not inst.get_imm_operands():
            return  # unsigned-offset or register-offset form — no writeback

        ops = inst.operands
        if len(ops) < 2:
            return

        assert isinstance(ops[0], RegisterOperand)
        assert isinstance(ops[1], MemoryOperand)

        norm_dest = self._norm(ops[0].value)
        if norm_dest in self._mem_regs_normalized(ops[1]):
            self._replace_reg(ops[0], forbidden={norm_dest} | self._mem_regs_normalized(ops[1]))

    def _patch_ldp(self, inst: Instruction) -> None:
        ops = inst.operands
        if len(ops) < 3:
            return

        assert isinstance(ops[0], RegisterOperand)
        assert isinstance(ops[1], RegisterOperand)
        assert isinstance(ops[2], MemoryOperand)

        has_writeback = bool(inst.get_imm_operands())
        norm0 = self._norm(ops[0].value)
        norm1 = self._norm(ops[1].value)
        base_norms = self._mem_regs_normalized(ops[2])

        # Constraint: dest0 != dest1
        if norm0 == norm1:
            self._replace_reg(ops[1], forbidden={norm1})
            norm1 = self._norm(ops[1].value)  # refresh after patch

        if has_writeback:
            # Constraint: dest0 not in base
            if norm0 in base_norms:
                self._replace_reg(ops[0], forbidden=base_norms | {norm1})
                norm0 = self._norm(ops[0].value)

            # Constraint: dest1 not in base
            if norm1 in base_norms:
                self._replace_reg(ops[1], forbidden=base_norms | {norm0})

    def _patch_ldxp(self, inst: Instruction) -> None:
        ops = inst.operands
        if len(ops) < 3:
            return

        assert isinstance(ops[0], RegisterOperand)
        assert isinstance(ops[1], RegisterOperand)
        assert isinstance(ops[2], MemoryOperand)

        base_norms = self._mem_regs_normalized(ops[2])
        norm0 = self._norm(ops[0].value)
        norm1 = self._norm(ops[1].value)

        if norm0 == norm1:
            self._replace_reg(ops[1], forbidden={norm1})
            norm1 = self._norm(ops[1].value)

        if norm0 in base_norms:
            self._replace_reg(ops[0], forbidden=base_norms | {norm1})
            norm0 = self._norm(ops[0].value)

        if norm1 in base_norms:
            self._replace_reg(ops[1], forbidden=base_norms | {norm0})

    def _patch_stp_writeback(self, inst: Instruction) -> None:
        if not inst.get_imm_operands():
            return  # unsigned-offset form — no writeback constraint

        ops = inst.operands
        if len(ops) < 3:
            return

        assert isinstance(ops[0], RegisterOperand)
        assert isinstance(ops[1], RegisterOperand)
        assert isinstance(ops[2], MemoryOperand)

        base_norms = self._mem_regs_normalized(ops[2])

        if self._norm(ops[0].value) in base_norms:
            self._replace_reg(ops[0], forbidden=base_norms)

        if self._norm(ops[1].value) in base_norms:
            self._replace_reg(ops[1], forbidden=base_norms)

    def _patch_stxr(self, inst: Instruction) -> None:
        ops = inst.operands
        if len(ops) < 3:
            return

        assert isinstance(ops[0], RegisterOperand)  # Ws (status)
        assert isinstance(ops[1], RegisterOperand)  # Rt (data)
        assert isinstance(ops[2], MemoryOperand)

        norm_status = self._norm(ops[0].value)
        norm_data = self._norm(ops[1].value)
        base_norms = self._mem_regs_normalized(ops[2])

        forbidden = {norm_data} | base_norms
        if norm_status in forbidden:
            self._replace_reg(ops[0], forbidden=forbidden)

    def _patch_stxp(self, inst: Instruction) -> None:
        ops = inst.operands
        if len(ops) < 4:
            return

        assert isinstance(ops[0], RegisterOperand)  # Ws (status)
        assert isinstance(ops[1], RegisterOperand)  # Rt1
        assert isinstance(ops[2], RegisterOperand)  # Rt2
        assert isinstance(ops[3], MemoryOperand)

        norm_status = self._norm(ops[0].value)
        forbidden = {
            self._norm(ops[1].value),
            self._norm(ops[2].value),
        } | self._mem_regs_normalized(ops[3])

        if norm_status in forbidden:
            self._replace_reg(ops[0], forbidden=forbidden)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _norm(self, reg: str) -> str:
        """Normalise a register name through the target descriptor."""
        return self.target_desc.reg_normalized[reg]

    def _mem_regs_normalized(self, mem_op: MemoryOperand) -> Set[str]:
        """
        Return the set of normalised register names referenced by a
        MemoryOperand.  mem_op.value is a string like "x1" or "x1, x2"
        depending on the addressing mode.
        """
        result = set()
        result.add(self._norm(mem_op.value))
#        for reg in mem_op.value.split(","):
#            reg = reg.strip()
#            if reg:
#                result.add(self._norm(reg))
        return result

    def _replace_reg(self, operand: RegisterOperand, forbidden: Set[str]) -> None:
        """
        Replace operand.value with a randomly chosen register of the same
        width that is not in *forbidden* (compared after normalisation).
        """
        candidates = [
            r for r in self.target_desc.registers[operand.width]
            if self._norm(r) not in forbidden
        ]
        if not candidates:
            # Should not happen in a correctly configured target descriptor,
            # but guard against it rather than crashing the fuzzer.
            raise RuntimeError("unable to solve constraints! unexpected!")

        operand.value = random.choice(candidates)

class Aarch64PatchUndefinedLoadsPass(Pass):
    def __init__(self, target_desc) -> None:
        self.target_desc: Aarch64TargetDesc = target_desc
        super().__init__()

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:

                to_patch: List[Instruction] = []
                for inst in bb:
                    # check if it's a load with post-index
                    if "ldr" in inst.name and inst.get_imm_operands():
                        ops = inst.operands
                        assert isinstance(ops[0], RegisterOperand)
                        assert isinstance(ops[1], MemoryOperand)
                        normalized_dest = self.target_desc.reg_normalized[ops[0].value]
                        if normalized_dest in ops[1].value:
                            to_patch.append(inst.operands[0])

                    if "ldp" in inst.name:
                        ops = inst.operands
                        assert isinstance(ops[0], RegisterOperand)
                        assert isinstance(ops[1], RegisterOperand)
                        normalized_dest0 = self.target_desc.reg_normalized[ops[0].value]
                        normalized_dest1 = self.target_desc.reg_normalized[ops[1].value]
                        if normalized_dest0 == normalized_dest1:
                            to_patch.append(inst.operands[0])

                # fix operands
                for org_dest in to_patch:
                    options = self.target_desc.registers[org_dest.width]
                    options = [i for i in options if i != org_dest.value]
                    new_value = random.choice(options)
                    org_dest.value = new_value


class Aarch64NonCanonicalAddressPass(Pass):

    def run_on_test_case(self, test_case: TestCase) -> None:
        pass

class Aarch64TagMemoryAccesses(Pass):
    def __init__(self, memory_accesses_to_guess_tag: Optional[List[int]] = None):
        super().__init__()
        if memory_accesses_to_guess_tag is None:
            memory_accesses_to_guess_tag = []

        self.memory_accesses_to_guess_tag = memory_accesses_to_guess_tag

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:

                memory_instructions = []

                for inst in bb:
                    if inst.has_memory_access:
                        memory_instructions.append(inst)

                for inst in memory_instructions:
                    to_subtract = []
                    mem_operands = inst.get_mem_operands()
                    base_operand: Optional[MemoryOperand] = None
                    for operand in mem_operands:
                        if operand.value in chain.from_iterable(Aarch64TargetDesc.registers.values()) and base_operand is None:
                            base_operand = operand
                            if inst.memory_access_id not in self.memory_accesses_to_guess_tag:
                                mte_tag = base_operand.mte_memory_tag
                            else:
                                lst = list(range(0, 15))
                                lst.remove(base_operand.mte_memory_tag)
                                mte_tag = random.choice(lst)

                            x7_register = RegisterOperand("x7", 64, True, True)
                            x7_register.name = "x7_reg"
                            imm_width = 4
                            imm_op = ImmediateOperand(f'0b{mte_tag:0{imm_width}b}', imm_width)
                            imm_op.name = "imm_op"
                            tag_register_instruction = Instruction("MOV", True).add_op(x7_register).add_op(imm_op)
                            tag_register_instruction.template = f"MOV {{{x7_register.name}}}, {{{imm_op.name}}}"
                            bb.insert_before(position=inst ,inst=tag_register_instruction)


                            ubfx_instruction = Instruction("UBFX", True).add_op(base_operand).add_op(base_operand)
                            ubfx_instruction.template = f"UBFX {{{base_operand.name}}}, {{{base_operand.name}}}, #0, #56"
                            bb.insert_before(position=inst ,inst=ubfx_instruction)

                            set_tag_register_instruction = Instruction("ORR", True).add_op(base_operand).add_op(base_operand).add_op(x7_register)
                            set_tag_register_instruction.template = f"ORR {{{base_operand.name}}}, {{{base_operand.name}}}, {{{x7_register.name}}}, LSL 56"
                            bb.insert_before(position=inst ,inst=set_tag_register_instruction)
                        else:
                            to_subtract.append(operand)

                    if base_operand is not None:
                        for operand in to_subtract:
                            base_operand_cpy = copy.deepcopy(base_operand)
                            other_operand_cpy = copy.deepcopy(operand)
                            base_operand_cpy.name += '0'
                            other_operand_cpy.name += '1'
                            other_operand_cpy.src = True
                            other_operand_cpy.dest = False
                            base_operand_cpy.src = True
                            base_operand_cpy.dest = True
                            sub_inst = Instruction("SUB", True).add_op(base_operand_cpy).add_op(base_operand_cpy).add_op(other_operand_cpy)

                            sub_inst.template = f'sub {{{base_operand_cpy.name}}}, {{{base_operand_cpy.name}}}, {{{other_operand_cpy.name}}}'  # TODO: this should be done in the constructor
                            bb.insert_before(inst, sub_inst)


class Aarch64ASMLayout:
    prologue_template = [
        ".test_case_enter:",
    ]

    epilogue_template = [
        ".section .data.main",
        ".test_case_exit:",
        ""
    ]
 

    def __init__(self, test_case: TestCase):
        self._instruction_counter = 0
        self.content: List[str] = []
        self.instruction_address: Dict[Instruction, int] = {}
        self._create_asm(test_case)


    def _create_asm(self, test_case: TestCase):

        for line in self.prologue_template:
            self.content.append(line)

        for func in test_case.functions:
            self._create_function(func)

        for line in self.epilogue_template:
            self.content.append(line)


    def _create_function(self, func: Function):
        self.content.append(f'.section .data.{func.owner.name}')
        self.content.append(func.name + ":")

        for bb in func:
            self._create_basic_block(bb)

        self._create_basic_block(func.exit)


    def _create_basic_block(self, bb: BasicBlock):
        self.content.append(bb.name.lower() + ":")

        for inst in bb:
            string, instruction_count = self._instruction_to_str(inst)
            self.instruction_address[inst] = self._instruction_counter * 4
            self.content.append(string)
            self._instruction_counter += 1

        for inst in bb.terminators:
            string, instruction_count = self._instruction_to_str(inst)
            self.instruction_address[inst] = self._instruction_counter * 4
            self.content.append(string)
            self._instruction_counter += 1


    def _instruction_to_str(self, inst: Instruction) -> Tuple[str, int]:
        if inst.name == "macro":
            return self._macro_to_str(inst)

        instruction = inst.to_asm_string()

        if inst.is_instrumentation:
            comment = "// instrumentation"
        elif inst.is_noremove:
            comment = "// noremove"
        else:
            comment = ""

        return f"{instruction} {comment}", 1

    def _macro_to_str(self, inst: Instruction) -> Tuple[str, int]:
        macro_placeholder = "NOP"
        if inst.operands[1].value.lower() == ".noarg":
            return f".macro{inst.operands[0].value}: {macro_placeholder}", 1 # For now, assume macro adds exactly 1 instruction
        else:
            return f".macro{inst.operands[0].value}{inst.operands[1].value}: {macro_placeholder}", 1 # For now, assume macro adds exactly 1 instruction



class Aarch64Printer(Printer):

    def __init__(self, _: Aarch64TargetDesc) -> None:
        super().__init__()

    def print_layout(self, layout: Aarch64ASMLayout, outfile: str = None) -> str:
        data = "\n".join(layout.content)

        if outfile is not None:
            with open(outfile, "w") as f:
                f.write(data)

        return data

    def print(self, test_case: TestCase, outfile: str = None) -> str:
        return self.print_layout(Aarch64ASMLayout(test_case), outfile)




from enum import Enum, auto

class PACVariant(Enum):
    """Labels for the TC variants produced by PAC stage-2 instrumentation."""
    STRIP_ONLY    = auto()  # XPAC strip before every access — no auth, arch-safe baseline
    AUTH_CORRECT  = auto()  # AUTIA with correct signed pointer and context (auth always succeeds)
    AUTH_WRONG    = auto()  # AUTIA with wrong PAC bits/ctx for spec slots — NI test

class MTEVariant(Enum):
    """Labels for the TC variants produced by MTE stage-2 instrumentation."""
    BASELINE      = auto()  # NOP placeholders — no tag change, reference behavior
    RANDOMIZE_TAG = auto()  # IRG on spec accesses — random tag (correct for NI contract)
    WRONG_TAG     = auto()  # MOVK wrong upper16 for spec accesses — deterministic wrong tag

class PACKey(Enum):
    IA = 'ia'
    IB = 'ib'
    DA = 'da'
    DB = 'db'
    G  = 'g'

SLOT_SIG_POS  = 0   # slot position 0: NOP (stage-1) or MOVK upper-16 signature (TC1/TC2/TC3)
AUTH_SLOT_POS = 1   # slot position 1: XPAC (stage-1/TC1) or AUTH (TC2/TC3)
SLOT_SIZE     = 2

_PAC_INFO: Dict[str, Tuple[PACKey, str, str]] = {
    # pac_mnemonic: (key, auth_mnemonic, xpac_mnemonic)
    'pacia':  (PACKey.IA, 'autia',  'xpaci'),  'pacib':  (PACKey.IB, 'autib',  'xpaci'),
    'pacda':  (PACKey.DA, 'autda',  'xpacd'),  'pacdb':  (PACKey.DB, 'autdb',  'xpacd'),
    'paciza': (PACKey.IA, 'autiza', 'xpaci'),  'pacizb': (PACKey.IB, 'autizb', 'xpaci'),
    'pacdza': (PACKey.DA, 'autdza', 'xpacd'),  'pacdzb': (PACKey.DB, 'autdzb', 'xpacd'),
}
# Reverse maps derived from _PAC_INFO (keyed by auth mnemonic)
_AUTH_TO_KEY:  Dict[str, PACKey] = {auth: key  for _, (key, auth, _)    in _PAC_INFO.items()}
_AUTH_TO_XPAC: Dict[str, str]    = {auth: xpac for _, (_,   auth, xpac) in _PAC_INFO.items()}
_AUTH_TO_PAC:  Dict[str, str]    = {auth: pac  for pac, (_,  auth, _)   in _PAC_INFO.items()}


@dataclass
class FixPoint:
    slot_id: int
    slot_insts: List[Instruction] = field(default_factory=list)
    committed_inst: Optional[Any] = None  # AUT* Instruction committed at stage-1 build time
    # Per-input values populated by executor from CE trace (reset between inputs):
    spec_nesting: Optional[int] = None
    correct_sig: Optional[int] = None  # upper-16 PAC bits signed by kernel; None if CE never reached
    alt_sig:     Optional[int] = None  # upper-16 PAC bits from an alternative (ptr/ctx) combo; for TC3 spec

    def reset(self) -> None:
        self.spec_nesting = None
        self.correct_sig  = None
        self.alt_sig      = None


class _SandboxInstrumentationBase:
    """Shared helpers for sandbox-taint-based instrumentation passes."""

    _norm: Dict[str, str]
    _sandbox_mask: str
    _sandbox_base_reg: str

    def _norm_reg(self, reg: str) -> str:
        return self._norm.get(reg, reg)

    def _dest_regs(self, inst: Instruction) -> frozenset:
        result = frozenset(
            self._norm_reg(op.value)
            for op in inst.operands + inst.implicit_operands
            if op.dest and op.type == OT.REG and op.value in self._norm
        )
        # Writeback forms (pre/post-index) update the base register, but base.json may not
        # mark it as dest.  For LDR/LDRB/etc. the offset operand is named "simm" (either
        # IMM=post-index or MEM=pre-index); unsigned-offset uses "pimm" (MEM) — no writeback.
        # For LDP/STP post-index the offset is "imm" as IMM.
        if inst.has_memory_access:
            has_simm = any(op.name.lower() == 'simm' for op in inst.operands + inst.implicit_operands)
            has_imm = any(op.name.lower() == 'imm' for op in inst.operands + inst.implicit_operands)
            if has_simm or (len(inst.operands) == 4 and has_imm):
                base = self._get_mem_base_reg(inst)
                if base is not None and base in self._norm:
                    result = result | frozenset([self._norm_reg(base)])
        return result

    def _get_mem_base_reg(self, inst: Instruction) -> Optional[str]:
        for op in inst.operands + inst.implicit_operands:
            if op.type == OT.MEM:
                return op.value
        return None

    def _make_offset_sub_insts(self, mem_inst: Instruction, base_reg: str) -> List[Instruction]:
        """Return SUB instructions that pre-subtract the memory instruction's offset from
        base_reg so that [base_reg + offset] lands at exactly base_reg's sandboxed address."""
        result: List[Instruction] = []
        for op in mem_inst.get_mem_operands()[1:]:
            if op.name.lower() == "pimm":
                current = int(op.value)
                while current > 0:
                    chunk = min(4095, current)
                    result.append(Instruction("sub", True, "", False,
                                              template=f"SUB {base_reg}, {base_reg}, #{chunk}"))
                    current -= chunk
            else:
                result.append(Instruction("sub", True, "", False,
                                          template=f"SUB {base_reg}, {base_reg}, {op.value}"))
        return result

    def _make_sandbox_insts(self, reg: str) -> List[Instruction]:
        """Return [AND reg, reg, #mask; ADD reg, reg, x29] that sandbox reg into the input region."""
        and_inst = Instruction("and", True, "", False, template=f"AND {reg}, {reg}, {self._sandbox_mask}")
        add_inst = Instruction("add", True, "", False, template=f"ADD {reg}, {reg}, {self._sandbox_base_reg}")
        return [and_inst, add_inst]

    @staticmethod
    def _topo_sort(func: Function) -> Tuple[Dict[BasicBlock, List[BasicBlock]], List[BasicBlock]]:
        """Return (predecessors, topo_order) for func's CFG."""
        predecessors: Dict[BasicBlock, List[BasicBlock]] = {}
        for bb in func:
            predecessors.setdefault(bb, [])
            for succ in bb.successors:
                predecessors.setdefault(succ, []).append(bb)
        topo: List[BasicBlock] = []
        seen: Set[BasicBlock] = set()
        def _dfs(bb: BasicBlock) -> None:
            if bb in seen:
                return
            seen.add(bb)
            for succ in bb.successors:
                _dfs(succ)
            topo.append(bb)
        _dfs(func.get_first_bb())
        topo.reverse()
        return predecessors, topo


class AuthInstructionSpec(InstructionSpec):
    """AUT* instruction spec that retries generation until ptr_reg ≠ ctx_reg."""

    def generate(self, generator) -> Instruction:
        norm = generator.target_desc.reg_normalized
        for _ in range(20):
            inst = super().generate(generator)
            if len(inst.operands) < 2:
                return inst  # zero-context variant — always valid
            if norm.get(inst.operands[0].value) != norm.get(inst.operands[1].value):
                return inst
        raise RuntimeError(f"Cannot generate {self.name} with ptr_reg ≠ ctx_reg")


class PACInstrumentation(_SandboxInstrumentationBase):

    def __init__(self, generator: Aarch64Generator, xpac_weight: int, auth_weight: int):
        xpac_weight = max(xpac_weight, 0)
        auth_weight = max(auth_weight, 0)
        total = (xpac_weight + auth_weight) * 1.0 or 1.0
        self._xpac_prob = xpac_weight / total
        self._auth_prob = auth_weight / total
        self.generator = generator
        self._norm = generator.target_desc.reg_normalized

        pac_instructions = [i for i in generator.instruction_set.instruction_unfiltered
                            if "BASE-PAC" in i.tags and
                            (CONF.supported_instructions is None or i.name in CONF.supported_instructions)]
        # Keep only instructions with 1 or 2 explicit operands where the first is a dest GPR.
        # This excludes: 0-operand system variants (pacia1716, paciasp…), 3-op pacga,
        # and src-only 1-op variants (autiasppcr, autibsppcr).
        def _is_usable_pac(i) -> bool:
            return 1 <= len(i.operands) <= 2 and i.operands[0].dest
        signing_instructions = list(filter(lambda i: i.name.lower().startswith('pac') and _is_usable_pac(i), pac_instructions))
        verification_instructions = list(filter(lambda i: i.name.lower().startswith('aut') and _is_usable_pac(i), pac_instructions))
        strip_sign_instructions  = list(filter(lambda i: i.name.lower().startswith('xpac') and _is_usable_pac(i), pac_instructions))


        self._pac_specs = {s.name.lower(): s for s in signing_instructions}
        self._auth_specs = {
            s.name.lower(): AuthInstructionSpec(
                s.name, s.category, s.control_flow, s.datatype,
                s.template, s.operands, s.implicit_operands, s.tags)
            for s in verification_instructions
        }
        self._xpac_specs = {s.name.lower(): s for s in strip_sign_instructions}
        # Sandbox parameters: mask lower bits of address and add sandbox base (x29).
        # Bundled with every signing operation so that signed values are always sandboxed.
        _mask_bits = int(math.log(MAIN_AREA_SIZE + FAULTY_AREA_SIZE, 2))
        self._sandbox_mask = f"#0x{(1 << _mask_bits) - 1:x}"
        self._sandbox_base_reg = "x29"

        # PAC/AUT/XPAC instructions are allowed in the base test case; the stage-1 pass
        # locates every AUT* and replaces it with an XPAC placeholder slot.

    # ------------------------------------------------------------------
    # Instruction builders
    # ------------------------------------------------------------------

    def _get_signing_instruction(self, reg_operand: Optional[Operand] = None, modifier: Optional[Operand] = None) -> Instruction:
        for _ in range(20):
            spec = random.choice(list(self._pac_specs.values()))
            instruction = self.generator.generate_instruction(spec)
            if reg_operand is not None and len(instruction.operands) >= 1:
                assert instruction.operands[0].type == OT.REG and reg_operand.type == OT.REG
                instruction.operands[0] = copy.deepcopy(reg_operand)
            if modifier is not None and len(instruction.operands) >= 2:
                assert instruction.operands[1].type == OT.REG and modifier.type == OT.REG
                instruction.operands[1] = copy.deepcopy(modifier)
            if len(instruction.operands) < 2:
                return instruction  # zero-context variant — always valid
            r0 = self._norm_reg(instruction.operands[0].value)
            r1 = self._norm_reg(instruction.operands[1].value)
            if r0 != r1:
                return instruction
        raise RuntimeError("Unable to generate PAC signing instruction")

    def _get_auth_instruction(self) -> Instruction:
        """Generate a random AUT* instruction with distinct operand registers."""
        for _ in range(20):
            spec = random.choice(list(self._auth_specs.values()))
            instruction = self.generator.generate_instruction(spec)
            if len(instruction.operands) < 2:
                return instruction  # zero-context variant — always valid
            r0 = self._norm_reg(instruction.operands[0].value)
            r1 = self._norm_reg(instruction.operands[1].value)
            if r0 != r1:
                return instruction
        raise RuntimeError("Unable to generate PAC auth instruction")

    def _get_mem_auth_instruction(self, mem_reg: str) -> Optional[Instruction]:
        """Generate a random AUT* with ptr_reg forced to mem_reg and ctx_reg != mem_reg.

        Returns None if no valid instruction can be found (caller falls back to standalone sandbox).
        """
        norm_mem = self._norm_reg(mem_reg)
        for _ in range(20):
            candidate = self._get_auth_instruction()
            if len(candidate.operands) < 2:
                candidate.operands[0].value = mem_reg
                return candidate
            if self._norm_reg(candidate.operands[1].value) != norm_mem:
                candidate.operands[0].value = mem_reg
                return candidate
        return None

    def _make_auth_inst(self, mnemonic: str, reg: str, ctx_reg: Optional[str]) -> Instruction:
        inst = self.generator.generate_instruction(self._auth_specs[mnemonic])
        inst.operands[0].value = reg
        if ctx_reg is not None and len(inst.operands) > 1:
            inst.operands[1].value = ctx_reg
        return inst

    def _make_xpac_inst(self, mnemonic: str, reg: str, slot_id: int, pos: int) -> Instruction:
        inst = self.generator.generate_instruction(self._xpac_specs[mnemonic])
        inst.operands[0].value = reg
        inst._pac_slot_id = slot_id
        inst._pac_slot_pos = pos
        return inst

    def _make_nop(self, slot_id: int, pos: int) -> Instruction:
        nop = Instruction("nop", True, "", False, template="NOP")
        nop._pac_slot_id = slot_id
        nop._pac_slot_pos = pos
        return nop

    def _make_movk(self, slot_id: int, pos: int, reg: str, imm: int, lsl: int) -> Instruction:
        inst = Instruction("movk", True, "", False, template=f"MOVK {reg}, #0x{imm & 0xFFFF:04x}, LSL #{lsl}")
        inst._pac_slot_id = slot_id
        inst._pac_slot_pos = pos
        return inst

    # ------------------------------------------------------------------
    # Stage 1: replace every AUT* with an XPAC placeholder slot;
    #          optionally insert a slot before memory accesses.
    # ------------------------------------------------------------------

    def _build_func_slots(
        self,
        func: Function,
        slot_counter: int,
        fix_points: List[FixPoint],
        auth_replacements: List,      # (old_auth_inst, bb, new_slot_insts)
        xpac_insertions: List,        # (mem_inst, bb, slot_insts, offset_subs, sandbox_insts)
        standalone_insertions: List,  # (mem_inst, bb, sandbox_insts + offset_subs)
    ) -> int:
        for bb in func:
            for inst in bb:
                mn = inst.name.lower()

                # AUT* in generated code: replace with [NOP, XPAC] slot
                if mn in self._auth_specs:
                    xpac_mn = _AUTH_TO_XPAC[mn]
                    ptr_reg = inst.operands[0].value
                    # Disallow ptr_reg == ctx_reg: resample ctx from the same allowed value pool.
                    if len(inst.operands) > 1 and \
                            self._norm_reg(ptr_reg) == self._norm_reg(inst.operands[1].value):
                        norm_ptr = self._norm_reg(ptr_reg)
                        for _ in range(20):
                            fresh = self.generator.generate_instruction(self._auth_specs[mn])
                            if len(fresh.operands) > 1 and \
                                    self._norm_reg(fresh.operands[1].value) != norm_ptr:
                                inst.operands[1].value = fresh.operands[1].value
                                break
                    sid = slot_counter; slot_counter += 1
                    slot_insts = [
                        self._make_nop(sid, SLOT_SIG_POS),
                        self._make_xpac_inst(xpac_mn, ptr_reg, sid, AUTH_SLOT_POS),
                    ]
                    fix_points.append(FixPoint(slot_id=sid, slot_insts=slot_insts,
                                               committed_inst=copy.deepcopy(inst)))
                    auth_replacements.append((inst, bb, slot_insts))
                    continue  # don't also process as memory access

                # Memory access: sandbox + optional PAC slot
                if inst.has_memory_access:
                    mem_reg = self._get_mem_base_reg(inst)
                    if mem_reg is None:
                        continue
                    sandbox_insts = self._make_sandbox_insts(mem_reg)
                    offset_subs   = self._make_offset_sub_insts(inst, mem_reg)
                    if self._auth_specs and random.random() < self._auth_prob:
                        auth_inst = self._get_mem_auth_instruction(mem_reg)
                        if auth_inst is not None:
                            xpac_mn = _AUTH_TO_XPAC[auth_inst.name.lower()]
                            sid = slot_counter; slot_counter += 1
                            slot_insts = [
                                self._make_nop(sid, SLOT_SIG_POS),
                                self._make_xpac_inst(xpac_mn, mem_reg, sid, AUTH_SLOT_POS),
                            ]
                            fix_points.append(FixPoint(slot_id=sid, slot_insts=slot_insts,
                                                       committed_inst=auth_inst))
                            xpac_insertions.append((inst, bb, slot_insts, offset_subs, sandbox_insts))
                            continue
                    standalone_insertions.append((inst, bb, sandbox_insts + offset_subs))

        return slot_counter

    def instrument_stage1(self, test_case: TestCase) -> Tuple[TestCase, List[FixPoint]]:
        """Replace AUT* instructions with XPAC placeholder slots; add slots before some memory accesses.

        Returns (instrumented_tc, fix_points).
        """
        tc = copy.deepcopy(test_case)
        fix_points: List[FixPoint] = []
        slot_counter = 0

        for func in tc.functions:
            auth_replacements: List = []
            xpac_insertions: List = []
            standalone_insertions: List = []
            slot_counter = self._build_func_slots(
                func, slot_counter, fix_points,
                auth_replacements, xpac_insertions, standalone_insertions)

            # AUT* → [NOP, XPAC] (delete old instruction, insert 2 new ones)
            for old_inst, bb, new_insts in auth_replacements:
                for ni in new_insts:
                    bb.insert_before(old_inst, ni)
                bb.delete(old_inst)

            # Memory access with slot: [sandbox, NOP, XPAC, offset_subs, mem_access]
            for mem_inst, bb, slot_insts, offset_subs, sandbox_insts in xpac_insertions:
                for s in [*sandbox_insts, *slot_insts, *offset_subs]:
                    bb.insert_before(mem_inst, s)

            # Memory access without slot: [sandbox, offset_subs, mem_access]
            for mem_inst, bb, insts in standalone_insertions:
                for s in insts:
                    bb.insert_before(mem_inst, s)

            _slot_ids_now = {
                i._pac_slot_id
                for _f in tc.functions for _b in _f for i in _b
                if hasattr(i, '_pac_slot_id')
            }
            for fp in fix_points:
                assert fp.slot_id in _slot_ids_now, (
                    f"stage1 insertion bug: slot_id={fp.slot_id} missing from tc after insertions"
                )

        return tc, fix_points

    # ------------------------------------------------------------------
    # Stage 2: produce TC1 / TC2 / TC3 from stage-1 result
    # ------------------------------------------------------------------

    def _committed_info(self, fp: FixPoint) -> Tuple[str, str, str, Optional[str]]:
        """Return (auth_mn, xpac_mn, ptr_reg, ctx_reg) from fp.committed_inst."""
        auth_mn = fp.committed_inst.name.lower()
        xpac_mn = _AUTH_TO_XPAC[auth_mn]
        ptr_reg = fp.committed_inst.operands[0].value
        ctx_reg = fp.committed_inst.operands[1].value if len(fp.committed_inst.operands) > 1 else None
        return auth_mn, xpac_mn, ptr_reg, ctx_reg

    def _make_tc1_slot(self, fp: FixPoint) -> List[Instruction]:
        """TC1 (STRIP_ONLY): [MOVK correct_sig LSL#48, XPAC] if sig known, else [NOP, XPAC]."""
        _, xpac_mn, ptr_reg, _ = self._committed_info(fp)
        sig_inst = (
            self._make_movk(fp.slot_id, SLOT_SIG_POS, ptr_reg, fp.correct_sig, 48)
            if fp.correct_sig is not None
            else self._make_nop(fp.slot_id, SLOT_SIG_POS)
        )
        return [sig_inst, self._make_xpac_inst(xpac_mn, ptr_reg, fp.slot_id, AUTH_SLOT_POS)]

    def _make_tc2_slot(self, fp: FixPoint) -> List[Instruction]:
        """TC2 (AUTH_CORRECT): [MOVK ptr_reg, #correct_sig, LSL#48, AUTH ptr_reg, ctx_reg]."""
        auth_mn, _, ptr_reg, ctx_reg = self._committed_info(fp)
        assert fp.correct_sig is not None  # caller must guard: only call when CE reached the slot
        movk = self._make_movk(fp.slot_id, SLOT_SIG_POS, ptr_reg, fp.correct_sig, 48)
        auth = self._make_auth_inst(auth_mn, ptr_reg, ctx_reg)
        auth._pac_slot_id  = fp.slot_id
        auth._pac_slot_pos = AUTH_SLOT_POS
        return [movk, auth]

    def _make_tc3_spec_slot(self, fp: FixPoint) -> List[Instruction]:
        """TC3 spec: [MOVK ptr_reg, #alt_sig, LSL#48, AUTH ptr_reg, ctx_reg].

        alt_sig is an alternative signing combination (different ptr and/or ctx). Always set by executor.
        """
        auth_mn, _, ptr_reg, ctx_reg = self._committed_info(fp)
        assert fp.alt_sig is not None, f"slot_id={fp.slot_id}: alt_sig not set by executor"
        movk = self._make_movk(fp.slot_id, SLOT_SIG_POS, ptr_reg, fp.alt_sig, 48)
        auth = self._make_auth_inst(auth_mn, ptr_reg, ctx_reg)
        auth._pac_slot_id  = fp.slot_id
        auth._pac_slot_pos = AUTH_SLOT_POS
        return [movk, auth]

    def instrument_stage2(
        self, prep_tc: TestCase, fix_points: List[FixPoint],
    ) -> Dict[PACVariant, TestCase]:
        """Produce TC1/TC2/TC3 variants from the stage-1 TC.

        TC1 (STRIP_ONLY)   — [MOVK correct_sig, XPAC]     — same strip as stage-1, sig pre-loaded
        TC2 (AUTH_CORRECT) — [MOVK correct_sig, AUTH]     — AUTH always succeeds
        TC3 (AUTH_WRONG)   — arch slots: same as TC2
                           — spec slots: [MOVK alt_sig, AUTH]
        If CE never reached a slot (correct_sig is None): all variants use TC1.
        """
        strip_tc   = copy.deepcopy(prep_tc)
        correct_tc = copy.deepcopy(prep_tc)
        wrong_tc   = copy.deepcopy(prep_tc)
        maps = {
            PACVariant.STRIP_ONLY:   self._find_slot_insts(strip_tc),
            PACVariant.AUTH_CORRECT: self._find_slot_insts(correct_tc),
            PACVariant.AUTH_WRONG:   self._find_slot_insts(wrong_tc),
        }

        for fp in fix_points:
            self._fill_slot(maps[PACVariant.STRIP_ONLY], fp, self._make_tc1_slot(fp))
            if fp.correct_sig is None:
                # CE never reached this slot: CORRECT → TC1 (safe), WRONG → alt_sig
                self._fill_slot(maps[PACVariant.AUTH_CORRECT], fp, self._make_tc1_slot(fp))
                self._fill_slot(maps[PACVariant.AUTH_WRONG],   fp, self._make_tc3_spec_slot(fp))
                continue
            # spec_nesting is None when CE never reached the slot → treat as speculative
            is_spec = fp.spec_nesting != 0 if fp.spec_nesting is not None else True
            self._fill_slot(maps[PACVariant.AUTH_CORRECT], fp, self._make_tc2_slot(fp))
            if not is_spec:
                self._fill_slot(maps[PACVariant.AUTH_WRONG], fp, self._make_tc2_slot(fp))
            else:
                self._fill_slot(maps[PACVariant.AUTH_WRONG], fp, self._make_tc3_spec_slot(fp))

        return {
            PACVariant.STRIP_ONLY:   strip_tc,
            PACVariant.AUTH_CORRECT: correct_tc,
            PACVariant.AUTH_WRONG:   wrong_tc,
        }

    def _find_slot_insts(self, tc: TestCase) -> Dict[int, Dict[int, Tuple[Instruction, BasicBlock]]]:
        """Walk tc and return {slot_id: {pos: (inst, bb)}} for all tagged slot instructions."""
        slot_map: Dict[int, Dict[int, Tuple[Instruction, BasicBlock]]] = {}
        for func in tc.functions:
            for bb in func:
                for inst in bb:
                    if hasattr(inst, '_pac_slot_id'):
                        sid: int = inst._pac_slot_id
                        pos: int = inst._pac_slot_pos
                        slot_map.setdefault(sid, {})[pos] = (inst, bb)
        return slot_map

    def _fill_slot(self, slot_map: Dict, fp: FixPoint, new_insts: List[Instruction]) -> None:
        """Replace the SLOT_SIZE instructions in slot_map[fp.slot_id] with new_insts (padded with NOPs)."""
        positions = slot_map.get(fp.slot_id)
        assert positions is not None, (
            f"slot_id={fp.slot_id} not found in slot_map "
            f"(slot_map keys={sorted(slot_map.keys())})"
        )
        for pos in range(SLOT_SIZE):
            old_inst, bb = positions[pos]
            new_inst = new_insts[pos] if pos < len(new_insts) else self._make_nop(fp.slot_id, pos)
            bb.insert_before(old_inst, new_inst)
            bb.delete(old_inst)


# ===========================================================================
# MTE non-interference instrumentation
# ===========================================================================

MTE_SLOT_SIZE = 1  # one NOP placeholder per memory access


@dataclass
class MTEFixPoint:
    """Per-memory-access metadata for MTE stage-2 variant generation."""
    slot_id: int
    bb: BasicBlock
    mem_inst: Instruction
    reg: str                       # original base register name
    slot_insts: List[Instruction]  # single-element: [nop placeholder]
    spec_nesting: Optional[int] = None

    def reset(self) -> None:
        self.spec_nesting = None


class MTEInstrumentation(_SandboxInstrumentationBase):
    """
    Two-stage MTE non-interference instrumentation.

    Stage 1: insert a NOP placeholder before every memory access.
             For registers not yet correctly sandbox-tagged, AND+ADD is prepended.
             Taint = frozenset of normalized register names holding a correctly-
             sandbox-tagged address (via AND+ADD with x29, not subsequently overwritten).
             Taint is cleared on any register write; intersection at CFG join nodes.
             ADDG/SUBG with imm4==0 propagate the source tag to the destination.

    Stage 2: replace each NOP placeholder to produce TC1/TC2/TC3:
             TC1 → NOP (correct flow, arch_tag everywhere — baseline)
             TC2 → arch: NOP;  spec: IRG Xd,Xd  (random tag)
             TC3 → arch: NOP;  spec: MOVK Xd,#wrong_upper16,LSL#48  (deterministic wrong tag)
    """

    def __init__(self, generator: Aarch64Generator):
        self.generator = generator
        self._norm = generator.target_desc.reg_normalized
        _mask_bits = int(math.log(MAIN_AREA_SIZE + FAULTY_AREA_SIZE, 2))
        self._sandbox_mask = f"#0x{(1 << _mask_bits) - 1:x}"
        self._sandbox_base_reg = "x29"
        self.last_taint_log: List[str] = []

    # ------------------------------------------------------------------
    # Instruction builders
    # ------------------------------------------------------------------

    def _make_mte_nop(self, slot_id: int) -> Instruction:
        nop = Instruction("nop", True, "", False, template="NOP")
        nop._mte_slot_id = slot_id
        return nop

    def _make_mte_irg(self, reg: str, slot_id: int) -> Instruction:
        inst = Instruction("irg", True, "", False, template=f"IRG {reg}, {reg}")
        inst._mte_slot_id = slot_id
        return inst

    def _make_mte_movk_wrong_tag(self, reg: str, wrong_upper16: int, slot_id: int) -> Instruction:
        inst = Instruction("movk", True, "", False,
                           template=f"MOVK {reg}, #0x{wrong_upper16 & 0xFFFF:04x}, LSL #48")
        inst._mte_slot_id = slot_id
        return inst

    # ------------------------------------------------------------------
    # Taint helpers
    # ------------------------------------------------------------------

    def _mte_tag_propagates(self, inst: Instruction) -> Optional[Tuple[str, str]]:
        """If inst preserves the tag unchanged (ADDG/SUBG with imm4==0),
        return (dest_reg, src_reg). Otherwise None."""
        if inst.name.lower() not in ('addg', 'subg'):
            return None
        if len(inst.operands) < 4:
            return None
        try:
            imm4 = int(inst.operands[3].value)
        except (ValueError, IndexError):
            return None
        if imm4 != 0:
            return None
        return inst.operands[0].value, inst.operands[1].value

    # ------------------------------------------------------------------
    # Stage-1 dataflow pass
    # ------------------------------------------------------------------

    def _build_mte_slots(
        self,
        func: Function,
        slot_counter: int,
        fix_points: List,
        insertions: List,
        taint_log: List,
    ) -> int:
        """Topological taint pass: build NOP-placeholder fix_points for every memory access.

        taint (curr) = frozenset of normalized register names that hold a
        correctly-sandbox-tagged address (via AND+ADD with x29).
        Cleared on any register write; intersection at CFG join nodes.
        ADDG/SUBG with imm4==0 propagate the source tag to the destination.

        For each memory access:
          tainted base  → offset_subs + NOP placeholder only
          untainted base → AND+ADD + offset_subs + NOP placeholder; reg added to taint
        """
        predecessors, topo = self._topo_sort(func)
        taint_out: Dict[BasicBlock, frozenset] = {}

        for bb in topo:
            processed = [p for p in predecessors.get(bb, []) if p in taint_out]
            if not processed:
                curr: frozenset = frozenset()
            elif len(processed) == 1:
                curr = taint_out[processed[0]]
            else:
                curr = taint_out[processed[0]]
                for p in processed[1:]:
                    curr = curr & taint_out[p]

            for inst in bb:
                if inst.has_memory_access:
                    mem_reg = self._get_mem_base_reg(inst)
                    if mem_reg is not None:
                        norm_mem = self._norm_reg(mem_reg)
                        offset_subs = self._make_offset_sub_insts(inst, mem_reg)
                        sid = slot_counter
                        slot_counter += 1
                        nop = self._make_mte_nop(sid)
                        fp = MTEFixPoint(slot_id=sid, bb=bb, mem_inst=inst,
                                         reg=mem_reg, slot_insts=[nop])
                        fix_points.append(fp)

                        if norm_mem in curr:
                            insertions.append((inst, bb, [], offset_subs, [nop]))
                            taint_log.append(
                                f"  MEM-ACCESS   inst={inst.name:12s}  base={mem_reg}"
                                f"  decision=NOP-ONLY"
                                f"  taint={sorted(curr)}")
                        else:
                            sandbox_insts = self._make_sandbox_insts(mem_reg)
                            insertions.append((inst, bb, sandbox_insts, offset_subs, [nop]))
                            curr = curr | frozenset([norm_mem])
                            taint_log.append(
                                f"  MEM-ACCESS   inst={inst.name:12s}  base={mem_reg}"
                                f"  decision=SANDBOX+NOP"
                                f"  taint={sorted(curr)}")
                    else:
                        taint_log.append(
                            f"  MEM-ACCESS   inst={inst.name:12s}  base=None(implicit?)"
                            f"  decision=SKIP")

                prop = self._mte_tag_propagates(inst)
                for dreg in self._dest_regs(inst):
                    if prop is not None and dreg == self._norm_reg(prop[0]):
                        norm_src = self._norm_reg(prop[1])
                        if norm_src in curr:
                            curr = curr | frozenset([dreg])
                            taint_log.append(
                                f"  TAG-PRESERVE inst={inst.name:12s}  {norm_src}->{dreg}")
                        else:
                            if dreg in curr:
                                taint_log.append(
                                    f"  ARITH-CLEAR  inst={inst.name:12s}  clears={dreg}")
                            curr = curr - frozenset([dreg])
                    else:
                        if dreg in curr:
                            taint_log.append(
                                f"  ARITH-CLEAR  inst={inst.name:12s}  clears={dreg}")
                        curr = curr - frozenset([dreg])

            taint_out[bb] = curr

        return slot_counter

    # ------------------------------------------------------------------
    # Stage 1 public API
    # ------------------------------------------------------------------

    def instrument_stage1(self, test_case: TestCase) -> Tuple[TestCase, List[MTEFixPoint]]:
        """Instrument test_case with NOP placeholders before every memory access.

        Returns (instrumented_tc, fix_points).
        instrumented_tc contains:
          - optional AND+ADD before each memory access (for untainted registers)
          - optional offset SUBs (for immediate-offset addressing modes)
          - NOP placeholder immediately before each memory access
        """
        tc = copy.deepcopy(test_case)
        fix_points: List[MTEFixPoint] = []
        slot_counter = 0
        self.last_taint_log = []

        for func in tc.functions:
            insertions: List = []
            func_log: List[str] = []
            slot_counter = self._build_mte_slots(
                func, slot_counter, fix_points, insertions, func_log)
            self.last_taint_log.extend(func_log)

            for mem_inst, bb, sandbox_insts, offset_subs, nop_insts in insertions:
                for s in sandbox_insts:
                    bb.insert_before(mem_inst, s)
                for s in offset_subs:
                    bb.insert_before(mem_inst, s)
                for s in nop_insts:
                    bb.insert_before(mem_inst, s)

        return tc, fix_points

    # ------------------------------------------------------------------
    # Stage-2 slot helpers
    # ------------------------------------------------------------------

    def _find_slot_insts(self, tc: TestCase) -> Dict[int, Tuple[Any, BasicBlock]]:
        """Return {slot_id: (inst, bb)} for all MTE-tagged instructions in tc."""
        slot_map: Dict[int, Tuple[Any, BasicBlock]] = {}
        for func in tc.functions:
            for bb in func:
                for inst in bb:
                    if hasattr(inst, '_mte_slot_id'):
                        slot_map[inst._mte_slot_id] = (inst, bb)
        return slot_map

    def _fill_slot(self, slot_map: Dict, fp: MTEFixPoint, new_inst: Instruction) -> None:
        """Replace the NOP placeholder for fp.slot_id with new_inst."""
        entry = slot_map.get(fp.slot_id)
        assert entry is not None, (
            f"MTE slot_id={fp.slot_id} not in slot_map "
            f"(keys={sorted(slot_map.keys())})")
        old_inst, bb = entry
        bb.insert_before(old_inst, new_inst)
        bb.delete(old_inst)

    # ------------------------------------------------------------------
    # Stage 2 public API
    # ------------------------------------------------------------------

    def instrument_stage2(
        self,
        prep_tc: TestCase,
        fix_points: List[MTEFixPoint],
        sandbox_base: int,
    ) -> Dict[MTEVariant, TestCase]:
        """Produce TC1/TC2/TC3 from the stage-1 instrumented TC.

        sandbox_base is used to compute the deterministic wrong tag for TC3.
        spec_nesting must be populated on each FixPoint before calling.

        TC1 — correct flow: all placeholders → NOP
        TC2 — arch: NOP;  spec: IRG Xd,Xd  (randomizes bits[59:56])
        TC3 — arch: NOP;  spec: MOVK Xd,#wrong_upper16,LSL#48
                   wrong_upper16 preserves bits[63:60] and [55:48] of sandbox_base,
                   and sets bits[59:56] (tag) to arch_tag XOR 1.
        """
        sandbox_upper16 = (sandbox_base >> 48) & 0xFFFF
        arch_tag = (sandbox_base >> 56) & 0xF
        wrong_tag = arch_tag ^ 1
        wrong_upper16 = (sandbox_upper16 & ~(0xF << 8)) | (wrong_tag << 8)

        baseline_tc   = copy.deepcopy(prep_tc)
        randomize_tc  = copy.deepcopy(prep_tc)
        wrong_tag_tc  = copy.deepcopy(prep_tc)
        maps = {
            MTEVariant.BASELINE:      self._find_slot_insts(baseline_tc),
            MTEVariant.RANDOMIZE_TAG: self._find_slot_insts(randomize_tc),
            MTEVariant.WRONG_TAG:     self._find_slot_insts(wrong_tag_tc),
        }

        for fp in fix_points:
            # spec_nesting=0    → arch path → RANDOMIZE_TAG=NOP, WRONG_TAG=NOP (correct tag preserved)
            # spec_nesting>0    → spec path → RANDOMIZE_TAG=IRG, WRONG_TAG=MOVK wrong tag
            # spec_nesting=None → CE never executed this memory access (non-arch path);
            #                     hardware may speculate on it → treat as spec
            is_spec = fp.spec_nesting != 0

            self._fill_slot(maps[MTEVariant.BASELINE], fp, self._make_mte_nop(fp.slot_id))

            if is_spec:
                self._fill_slot(maps[MTEVariant.RANDOMIZE_TAG], fp, self._make_mte_irg(fp.reg, fp.slot_id))
            else:
                self._fill_slot(maps[MTEVariant.RANDOMIZE_TAG], fp, self._make_mte_nop(fp.slot_id))

            if is_spec:
                self._fill_slot(maps[MTEVariant.WRONG_TAG], fp,
                                self._make_mte_movk_wrong_tag(fp.reg, wrong_upper16, fp.slot_id))
            else:
                self._fill_slot(maps[MTEVariant.WRONG_TAG], fp, self._make_mte_nop(fp.slot_id))

        return {
            MTEVariant.BASELINE:      baseline_tc,
            MTEVariant.RANDOMIZE_TAG: randomize_tc,
            MTEVariant.WRONG_TAG:     wrong_tag_tc,
        }


class BitmapAccessor:
	def __init__(self, parent, attr_name, size_in_bits):
		self.parent = parent
		self.attr_name = attr_name
		self.size_in_bits = size_in_bits
	
	def _get_value(self):
		return getattr(self.parent, self.attr_name)

	def _set_value(self, value):
		setattr(self.parent, self.attr_name, value & ((1 << self.size_in_bits) - 1))  # mask to size

	def __getitem__(self, index):
		value = self._get_value()

		if isinstance(index, slice):
			start, stop, step = index.indices(self.size_in_bits)
			if step != 1:
				raise ValueError("BitmapAccessor does not support stepped slices")
			width = stop - start
			if width <= 0:
				return 0
			mask = (1 << width) - 1
			return (value >> start) & mask

		if index < 0 or index >= self.size_in_bits:
				raise IndexError(f"Bit index out of range: {index=} not in [0, {self.size_in_bits}]")
		return bool((value >> index) & 1)

	def __setitem__(self, index, val):
		value = self._get_value()

		if isinstance(index, slice):
			start, stop, step = index.indices(self.size_in_bits)
			if step != 1:
				raise ValueError("BitmapAccessor does not support stepped slices")
			width = stop - start
			if width <= 0:
				return
			mask = ((1 << width) - 1) << start
			value = (value & ~mask) | ((int(val) << start) & mask)
			self._set_value(value)
			return

		if index < 0 or index >= self.size_in_bits:
			raise IndexError(f"Bit index out of range: {index=} not in [0, {self.size_in_bits}]")

		if val:
			value |= (1 << index)
		else:
			value &= ~(1 << index)
		self._set_value(value)

	def __int__(self):
		return self._get_value()

	def __index__(self):
		return self._get_value()

	def __repr__(self):
		return f"{self._get_value():0{self.size_in_bits}b}"

	def __str__(self):
		return f"0b{self._get_value():0{self.size_in_bits}b}"

	def __call__(self, new_val: int):
		self._set_value(new_val)

	def __invert__(self): return ~self._get_value()
	def __and__(self, other): return int(self) & int(other)
	def __or__(self, other):  return int(self) | int(other)
	def __xor__(self, other): return int(self) ^ int(other)
	def __rand__(self, other): return int(other) & int(self)
	def __ror__(self, other):  return int(other) | int(self)
	def __rxor__(self, other): return int(other) ^ int(self)
	
	def __iand__(self, other): self._set_value(int(self) & int(other)); return self
	def __ior__(self, other): self._set_value(int(self) | int(other)); return self
	def __ixor__(self, other): self._set_value(int(self) ^ int(other)); return self
	
	def __lshift__(self, other): return int(self) << int(other)
	def __rshift__(self, other): return int(self) >> int(other)
	def __ilshift__(self, other): self._set_value(int(self) << int(other)); return self
	def __irshift__(self, other): self._set_value(int(self) >> int(other)); return self

	def __add__(self, other): return int(self) + int(other)
	def __sub__(self, other): return int(self) - int(other)
	def __mul__(self, other): return int(self) * int(other)
	def __floordiv__(self, other): return int(self) // int(other)
	def __mod__(self, other): return int(self) % int(other)
	def __pow__(self, other, modulo=None): return pow(int(self), int(other), modulo)
	def __neg__(self): return -int(self)
	def __pos__(self): return +int(self)

	def __iadd__(self, other): self._set_value(int(self) + int(other)); return self
	def __isub__(self, other): self._set_value(int(self) - int(other)); return self
	def __imul__(self, other): self._set_value(int(self) * int(other)); return self
	def __ifloordiv__(self, other): self._set_value(int(self) // int(other)); return self
	def __imod__(self, other): self._set_value(int(self) % int(other)); return self


	def __eq__(self, other): return int(self) == int(other)
	def __ne__(self, other): return int(self) != int(other)
	def __lt__(self, other): return int(self) < int(other)
	def __le__(self, other): return int(self) <= int(other)
	def __gt__(self, other): return int(self) > int(other)
	def __ge__(self, other): return int(self) >= int(other)



class Aarch64SandboxPass(Pass):
	def __init__(self):
		super().__init__()
		input_memory_size = MAIN_AREA_SIZE + FAULTY_AREA_SIZE
		mask_size = int(math.log(input_memory_size, 2))
		self.sandbox_address_mask = "0b" + "1" * mask_size

	def run_on_test_case(self, test_case: TestCase) -> None:
		for func in test_case.functions:
			for bb in func:

				memory_instructions = []

				for inst in bb:
					if inst.has_mem_operand(True):
						memory_instructions.append(inst)

				for inst in memory_instructions:
					self.sandbox_memory_access(inst, bb)

	def sandbox_memory_access(self, instr: Instruction, parent: BasicBlock):
		""" Force the memory accesses into the page starting from R14 """

		def generate_template(mnemonic: str, op0: Operand, op1: Operand, op2: Operand) -> Tuple[
			str, Operand, Operand, Operand]:
			op0_cpy = copy.deepcopy(op0)
			op1_cpy = copy.deepcopy(op1)
			op2_cpy = copy.deepcopy(op2)
			op0_cpy.name += "0"
			op1_cpy.name += "1"
			op2_cpy.name += "2"
			template = f"{mnemonic} {{{op0_cpy.name}}}, {{{op1_cpy.name}}}, {{{op2_cpy.name}}}"
			return template, op0_cpy, op1_cpy, op2_cpy

		mem_operands = instr.get_mem_operands()
		implicit_mem_operands = instr.get_implicit_mem_operands()
		if mem_operands and not implicit_mem_operands:
			#            assert len(mem_operands) == 1, f"Unexpected instruction format {instr.name}"
			base_operand: Operand = mem_operands[0]
			base_operand_copy = RegisterOperand(base_operand.value, base_operand.width, True, True)
			base_operand_copy.name = base_operand.name
			
			# TODO: Very bad implemented! Must fix
			
			imm_width = min(base_operand_copy.width, 32)
			imm_op = ImmediateOperand(self.sandbox_address_mask, imm_width)
			imm_op.name = "imm_op"
			template, op0, op1, op2 = generate_template("AND", base_operand_copy,
			                                            base_operand_copy, imm_op)
			apply_mask = Instruction("AND", True).add_op(op0).add_op(op1).add_op(op2)
			apply_mask.template = template
			parent.insert_before(instr, apply_mask)
			
			x29_register = RegisterOperand("x29", 64, True, False) # TODO: IMPORTANT! REMEMBER: I changed from x30 to x29
			x29_register.name = "x29_reg"
			template, op0, op1, op2 = generate_template("ADD", base_operand_copy, base_operand_copy,
			                                            x29_register)
			add_base = Instruction("ADD", True).add_op(op0).add_op(op1).add_op(op2)
			add_base.template = template
			parent.insert_before(instr, add_base)
			
			for op in mem_operands[1:]:
				if op.name.lower() == "pimm":
					current = int(op.value)
					while current > 0:
						offset_op = ImmediateOperand(str(min(4095, current)), 12)
						template, op0, op1, op2 = generate_template("SUB", base_operand_copy, base_operand_copy, offset_op)
						op2.dest = False
						op2.src = True
						sub_inst = Instruction("SUB", True).add_op(op0).add_op(op1).add_op(op2)
						sub_inst.template = template  # TODO: this should be done in the constructor
						parent.insert_before(instr, sub_inst)
						current -= 4095
				else:
				    template, op0, op1, op2 = generate_template("SUB", base_operand_copy, base_operand_copy, op)
				    op2.dest = False
				    op2.src = True
				    sub_inst = Instruction("SUB", True).add_op(op0).add_op(op1).add_op(op2)
				    sub_inst.template = template  # TODO: this should be done in the constructor
				    parent.insert_before(instr, sub_inst)

			return

		if implicit_mem_operands:
			raise GeneratorException("Implicit memory accesses are not supported")

		raise GeneratorException("Attempt to sandbox an instruction without memory operands")


class Aarch64RandomGenerator(Aarch64Generator, RandomGenerator):

    def __init__(self, instruction_set: InstructionSet, seed: int):
        super().__init__(instruction_set, seed)

    def _filter_invalid_operands(self, spec: OperandSpec, inst: Instruction) -> List[str]:
        result: List[str] = []
        register_prefixes = ("x", "w", "q", "v", "d", "s", "h", "b", "sp")

        for op in spec.values:
            if 'pc' == op:
                result.append(op)
            elif not op.startswith(register_prefixes):
                result.append(op)
            elif op in chain.from_iterable(self.target_desc.registers.values()):
                # We omit situations where the same physical register is in memory operand and outside the memory operand.
                # in causes warning of the assembler and unrecognized instructions
                cond = lambda o: o.type == OT.MEM and o.value in chain.from_iterable(self.target_desc.registers.values())
                if spec.type == OT.MEM:
                    cond = lambda _: True

                memory_registers = [o for o in inst.operands if cond(o)]
                if any(o.value not in Aarch64TargetDesc.reg_normalized for o in memory_registers):
                    continue
                if all(Aarch64TargetDesc.reg_normalized[op] != Aarch64TargetDesc.reg_normalized[o.value] for o in memory_registers):
                    result.append(op)
        return result

    def generate_reg_operand(self, spec: OperandSpec, inst: Instruction) -> Operand:
        choices = self._filter_invalid_operands(spec, inst)
        reg = random.choice(choices)
        return RegisterOperand(reg, spec.width, spec.src, spec.dest)

    def generate_cond_operand(self, spec: OperandSpec, _: Instruction) -> Operand:
        cond = random.choice(spec.values)
        return CondOperand(cond)

    def generate_mem_operand(self, spec: OperandSpec, inst: Instruction) -> Operand:
        choices = self._filter_invalid_operands(spec, inst)
        address_reg = random.choice(choices)
        return MemoryOperand(address_reg, spec.width, spec.src, spec.dest)
