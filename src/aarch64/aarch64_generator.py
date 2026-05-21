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




FIX_COUNT_CTX = 4   # MOVZ + MOVK×3 to restore the context register
FIX_COUNT_PTR = 4   # MOVZ + MOVK×3 to restore the signed pointer
CTX_SLOT_START = 0                              # slot positions 0-3: context restore
PTR_SLOT_START = FIX_COUNT_CTX                  # slot positions 4-7: pointer restore
AUTH_SLOT_POS  = FIX_COUNT_CTX + FIX_COUNT_PTR # slot position  8  : AUTH instruction
SLOT_SIZE = AUTH_SLOT_POS + 1                   # = 9

_PAC_TO_AUTH: Dict[str, str] = {  # sign mnemonic → matching auth mnemonic
    # 2-operand (with context register)
    "pacia": "autia", "pacib": "autib",
    "pacda": "autda", "pacdb": "autdb",
    # 1-operand zero-context (modifier = XZR)
    "paciza": "autiza", "pacizb": "autizb",
    "pacdza": "autdza", "pacdzb": "autdzb",
}
_PAC_CODE_MNEMONICS: frozenset = frozenset({"pacia", "pacib", "paciza", "pacizb"})


@dataclass(frozen=True)
class SignedRegInfo:
    reg: str                    # actual register name, e.g. "x0"
    ctx_reg: Optional[str]      # context register name; None for zero-context variants (paciza…)
    pac_mnemonic: str           # e.g. "pacia" or "paciza"
    auth_mnemonic: str          # e.g. "autia" or "autiza"
    xpac_mnemonic: str          # "xpaci" or "xpacd"
    # Reference to the Instruction object inserted by stage1 (not part of identity).
    # Used to locate the PACIA PC in the layout so ctx_reg can be captured at signing time.
    pac_inst: Optional[Any] = field(default=None, compare=False, hash=False)


@dataclass
class FixPoint:
    slot_id: int
    bb: BasicBlock
    mem_inst: Instruction
    info: SignedRegInfo
    # All PACIA Instruction objects whose output is the "last signing of info.reg" on at
    # least one CFG path that reaches this fix-point.  At a join node there may be one
    # per incoming path.  The executor maps every entry here into pac_offset_to_fps so
    # that whichever PACIA the arch execution actually ran is captured.
    pac_insts: frozenset = field(default_factory=frozenset)
    slot_insts: List[Instruction] = field(default_factory=list)
    signed_value: Optional[int] = None  # register value captured right after PACIA executes (signed ptr, PAC bits in [63:48])
    ctx_value: Optional[int] = None     # context register value captured at PACIA execution time
    spec_nesting: Optional[int] = None  # speculation nesting at PACIA capture time

    def reset(self) -> None:
        self.signed_value = None
        self.ctx_value = None
        self.spec_nesting = None


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


class PACInstrumentation(_SandboxInstrumentationBase):

    def __init__(self, generator: Aarch64Generator, xpac_weight: int, auth_weight: int, sign_weight: int):
        xpac_weight = max(xpac_weight, 0)
        auth_weight = max(auth_weight, 0)
        sign_weight = max(sign_weight, 0)
        total = (xpac_weight + auth_weight + sign_weight) * 1.0
        self._xpac_prob = xpac_weight / total
        self._auth_prob = auth_weight / total
        self._sign_prob = sign_weight / total
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
        self._auth_specs = {s.name.lower(): s for s in verification_instructions}
        self._xpac_specs = {s.name.lower(): s for s in strip_sign_instructions}
        # Sandbox parameters: mask lower bits of address and add sandbox base (x29).
        # Bundled with every signing operation so that signed values are always sandboxed.
        _mask_bits = int(math.log(MAIN_AREA_SIZE + FAULTY_AREA_SIZE, 2))
        self._sandbox_mask = f"#0x{(1 << _mask_bits) - 1:x}"
        self._sandbox_base_reg = "x29"

        # Claim ownership of BASE-PAC: no PAC/AUT/XPAC instruction may appear in
        # the random base test case.  They enter only through this class's stage1
        # instrumentation, which uses taint analysis to guarantee AUT targets are signed.
        generator.register_controlled_instructions({"BASE-PAC"})

        # Populated by instrument_stage1 — human-readable taint decision log.
        self.last_taint_log: List[str] = []

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

    def _make_movz(self, slot_id: int, pos: int, reg: str, imm: int) -> Instruction:
        inst = Instruction("movz", True, "", False, template=f"MOVZ {reg}, #0x{imm & 0xFFFF:04x}")
        inst._pac_slot_id = slot_id
        inst._pac_slot_pos = pos
        return inst

    def _make_movk(self, slot_id: int, pos: int, reg: str, imm: int, lsl: int) -> Instruction:
        inst = Instruction("movk", True, "", False, template=f"MOVK {reg}, #0x{imm & 0xFFFF:04x}, LSL #{lsl}")
        inst._pac_slot_id = slot_id
        inst._pac_slot_pos = pos
        return inst

    def _make_load_imm_insts(self, slot_id: int, reg: str, value: int, start_pos: int) -> List[Instruction]:
        """Return 4 instructions (MOVZ + 3×MOVK) loading `value` into `reg` at slot positions start_pos…start_pos+3."""
        chunks = [(value >> (16 * i)) & 0xFFFF for i in range(4)]
        insts: List[Instruction] = [self._make_movz(slot_id, start_pos, reg, chunks[0])]
        for i in range(1, 4):
            insts.append(self._make_movk(slot_id, start_pos + i, reg, chunks[i], i * 16))
        return insts

    def _nops_range(self, slot_id: int, start: int, count: int) -> List[Instruction]:
        return [self._make_nop(slot_id, start + i) for i in range(count)]

    # ------------------------------------------------------------------
    # Taint analysis
    # ------------------------------------------------------------------

    def _apply_pac(self, curr: frozenset, pac_inst: Instruction) -> frozenset:
        name = pac_inst.name.lower()
        if name not in _PAC_TO_AUTH:
            return curr
        reg = pac_inst.operands[0].value
        norm_reg = self._norm_reg(reg)
        ctx = pac_inst.operands[1].value if len(pac_inst.operands) > 1 else None
        info = SignedRegInfo(
            reg=reg, ctx_reg=ctx,
            pac_mnemonic=name,
            auth_mnemonic=_PAC_TO_AUTH[name],
            xpac_mnemonic="xpaci" if name in _PAC_CODE_MNEMONICS else "xpacd",
            pac_inst=pac_inst,
        )
        # Use normalized comparison so that register aliases resolve to the same physical reg.
        return frozenset({x for x in curr if self._norm_reg(x.reg) != norm_reg} | {info})

    def _compute_taints(
        self,
        func: Function,
        sign_ins: Dict[Instruction, Instruction],
        auth_ins: Dict[Instruction, Tuple[Instruction, BasicBlock]],
        choose_auth: bool = False,
    ) -> None:
        """Forward DFS taint analysis.

        sign_ins  — {anchor_inst: pac_inst}  (read-only)
        auth_ins  — {anchor_inst: (bb, info: SignedRegInfo)}
          choose_auth=False: read existing entries to strip taint
          choose_auth=True : randomly CREATE entries (running taint
                             prevents inserting a second AUTIA for the same reg)
        """
        self._taint_bb(func.get_first_bb(), frozenset(), sign_ins, auth_ins,
                       choose_auth, {})

    def _taint_bb(
        self,
        bb: BasicBlock,
        curr: frozenset,
        sign_ins: Dict,
        auth_ins: Dict,
        choose_auth: bool,
        visited: Dict,
    ) -> None:
        if bb in visited:
            return
        visited[bb] = True
        for inst in bb:
            if inst in sign_ins:
                curr = self._apply_pac(curr, sign_ins[inst])
            if choose_auth:
                if random.random() < self._auth_prob and curr:
                    info = random.choice(list(curr))
                    auth_ins[inst] = (bb, info)
                    curr = frozenset(x for x in curr if self._norm_reg(x.reg) != self._norm_reg(info.reg))
            else:
                if inst in auth_ins:
                    _, auth_info = auth_ins[inst]
                    curr = frozenset(x for x in curr if self._norm_reg(x.reg) != self._norm_reg(auth_info.reg))
            written = self._dest_regs(inst)
            if written:
                curr = frozenset(x for x in curr if self._norm_reg(x.reg) not in written)
        for succ in bb.successors:
            self._taint_bb(succ, curr, sign_ins, auth_ins, choose_auth, visited)

    # ------------------------------------------------------------------
    # Stage 1 helpers
    # ------------------------------------------------------------------

    def _build_xpac_slots(
        self,
        func: Function,
        sign_ins: Dict,
        auth_ins: Dict,
        slot_counter: int,
        fix_points: List,
        xpac_insertions: List,
        standalone_insertions: List,
        all_pac_at_auth: Dict,
        taint_log: List,
    ) -> int:
        """Data-flow pass: builds xpac_slots and standalone_insertions.

        Uses a running taint carried across BB boundaries (no per-BB reset).
        Creating an xpac_slot clears that register's taint (PAC consumed).
        Join nodes (multiple predecessors) use taint intersection.
        all_pac_at_auth is an output dict filled with {anchor_inst: all_pac_snapshot}
        taken just before the auth strip, used by instrument_stage1 to build auth-slot
        pac_insts with the correct union across all CFG paths.
        """
        predecessors, topo = self._topo_sort(func)

        taint_out: Dict[BasicBlock, frozenset] = {}
        # Parallel to taint_out: maps each BB to {norm_reg -> frozenset[Instruction]}.
        # Each entry holds all PACIA objects that are the most-recent signing of that
        # register on at least one path reaching the end of that BB.
        # At join nodes: union across predecessors for regs surviving the intersection.
        all_pac_out: Dict[BasicBlock, Dict[str, frozenset]] = {}

        for bb in topo:
            preds = predecessors.get(bb, [])
            processed = [p for p in preds if p in taint_out]

            if not processed:
                curr = frozenset()
                all_pac: Dict[str, frozenset] = {}
            elif len(processed) == 1:
                curr = taint_out[processed[0]]
                all_pac = dict(all_pac_out[processed[0]])
            else:
                # Intersection: only keep what is tainted on ALL incoming paths.
                curr = taint_out[processed[0]]
                for p in processed[1:]:
                    curr = curr & taint_out[p]
                # For regs surviving the intersection (same SignedRegInfo on all paths),
                # union their PACIA sets so the arch path's signing is always captured.
                surviving = {self._norm_reg(x.reg) for x in curr}
                all_pac = {}
                for nreg in surviving:
                    merged: frozenset = frozenset()
                    for p in processed:
                        # nreg is in taint_out[p] (intersection), so all_pac_out[p]
                        # is guaranteed to have nreg (taint_out and all_pac_out are
                        # always kept in sync).
                        assert nreg in all_pac_out[p], \
                            f"all_pac_out/taint_out sync violation: {nreg} missing from predecessor"
                        merged = merged | all_pac_out[p][nreg]
                    all_pac[nreg] = merged

            for inst in bb:
                # Fresh sign adds to taint (sign_ins values are (pac_inst, bb) tuples).
                if inst in sign_ins:
                    pac_inst_obj = sign_ins[inst][0]
                    norm_signed = self._norm_reg(pac_inst_obj.operands[0].value)
                    curr = self._apply_pac(curr, pac_inst_obj)
                    # Replace: this is now the sole last-signer for this reg on this path.
                    all_pac[norm_signed] = frozenset({pac_inst_obj})
                    taint_log.append(
                        f"  SIGN-ANCHOR  inst={inst.name:12s}  signed={norm_signed}"
                        f"  taint-after={sorted(self._norm_reg(x.reg) for x in curr)}"
                    )
                # auth_slot will strip this register's PAC → clear from running taint.
                if inst in auth_ins:
                    _, auth_info = auth_ins[inst]
                    norm_auth = self._norm_reg(auth_info.reg)
                    # Snapshot all_pac BEFORE the strip so instrument_stage1 can build
                    # pac_insts for this auth-slot with the correct union across all paths.
                    all_pac_at_auth[inst] = dict(all_pac)
                    # Use normalized comparison to stay in sync with all_pac keys.
                    curr = frozenset(x for x in curr if self._norm_reg(x.reg) != norm_auth)
                    all_pac.pop(norm_auth, None)
                    taint_log.append(
                        f"  AUTH-ANCHOR  inst={inst.name:12s}  stripped={norm_auth}"
                        f"  taint-after={sorted(self._norm_reg(x.reg) for x in curr)}"
                    )

                # Sandbox the memory access and prepare the fix point if needed
                if inst.has_memory_access:
                    mem_reg = self._get_mem_base_reg(inst)
                    if mem_reg is not None:
                        norm_mem = self._norm_reg(mem_reg)
                        matching = [x for x in curr if self._norm_reg(x.reg) == norm_mem]
                        offset_subs = self._make_offset_sub_insts(inst, mem_reg)
                        if matching:
                            assert len(matching) == 1
                            info = matching[0]
                            sid = slot_counter
                            slot_counter += 1
                            xpac = self._make_xpac_inst(info.xpac_mnemonic, info.reg, sid, AUTH_SLOT_POS)
                            nops = [self._make_nop(sid, pos) for pos in range(AUTH_SLOT_POS)]
                            slot_insts = nops + [xpac]
                            assert norm_mem in all_pac, \
                                f"all_pac/curr sync violation: tainted reg {norm_mem} missing from all_pac"
                            fix_points.append(FixPoint(
                                slot_id=sid, bb=bb, mem_inst=inst, info=info,
                                pac_insts=all_pac[norm_mem],
                                slot_insts=slot_insts))
                            xpac_insertions.append((inst, bb, slot_insts, offset_subs))
                            taint_log.append(
                                f"  MEM-ACCESS   inst={inst.name:12s}  base={mem_reg}"
                                f"  decision=XPAC(slot={sid})"
                                f"  taint={sorted(self._norm_reg(x.reg) for x in curr)}"
                                f"  n_offset_subs={len(offset_subs)}"
                            )
                            # PAC consumed: clear so subsequent accesses use standalone.
                            curr = frozenset(x for x in curr if self._norm_reg(x.reg) != norm_mem)
                            all_pac.pop(norm_mem, None)
                        else:
                            standalone_insertions.append(
                                (inst, bb, self._make_sandbox_insts(mem_reg) + offset_subs))
                            taint_log.append(
                                f"  MEM-ACCESS   inst={inst.name:12s}  base={mem_reg}"
                                f"  decision=STANDALONE"
                                f"  taint={sorted(self._norm_reg(x.reg) for x in curr)}"
                                f"  n_offset_subs={len(offset_subs)}"
                            )
                    else:
                        taint_log.append(
                            f"  MEM-ACCESS   inst={inst.name:12s}  base=None(implicit?)"
                            f"  decision=SKIP"
                        )

                # Arithmetic (and writeback) clears taint for written registers.
                for dreg in self._dest_regs(inst):
                    if dreg in {self._norm_reg(x.reg) for x in curr}:
                        taint_log.append(
                            f"  ARITH-CLEAR  inst={inst.name:12s}  clears={dreg}"
                        )
                    curr = frozenset(x for x in curr if self._norm_reg(x.reg) != dreg)
                    all_pac.pop(dreg, None)

            taint_out[bb] = curr
            all_pac_out[bb] = dict(all_pac)

        return slot_counter

    # ------------------------------------------------------------------
    # Stage 1: instrument with PACIA, safe AUTIA, and XPAC slots
    # ------------------------------------------------------------------

    def instrument_stage1(self, test_case: TestCase) -> Tuple[TestCase, List[FixPoint]]:
        """
        Returns (instrumented_tc, fix_points).

        instrumented_tc has:
          - random PACIA instructions inserted
          - random AUTIA instructions inserted only where the target register is
            provably signed by taint analysis (safe on FEAT_FPAC hardware)
          - SLOT_SIZE-instruction slots [XPAC, NOP×(SLOT_SIZE-1)] before each
            memory access that uses a signed register; each slot instruction is
            tagged with _pac_slot_id and _pac_slot_pos for later retrieval

        fix_points carries metadata about each slot for instrument_stage2.
        """
        tc = copy.deepcopy(test_case)
        fix_points: List[FixPoint] = []
        slot_counter = 0
        self.last_taint_log = []

        for func in tc.functions:
            # Step 1: choose random positions for signing insertions.
            # Each signing block is [AND reg,reg,#mask; ADD reg,reg,x29; PACIA reg,ctx]:
            # the sandbox is integrated into the sign so every signed value is already
            # sandboxed and no separate sandbox pass is needed.
            sign_ins: Dict[Instruction, Tuple[Instruction, BasicBlock]] = {}
            sandbox_ins: Dict[Instruction, List[Instruction]] = {}
            for bb in func:
                for inst in bb:
                    if random.random() < self._sign_prob:
                        pac_inst = self._get_signing_instruction()
                        reg = pac_inst.operands[0].value
                        sign_ins[inst] = (pac_inst, bb)
                        sandbox_ins[inst] = self._make_sandbox_insts(reg)

            sign_specs = {i: p for i, (p, _) in sign_ins.items()}

            # Step 2: choose AUTIA insertions via forward taint pass
            auth_ins: Dict[Instruction, Tuple[Instruction, BasicBlock]] = {}
            self._compute_taints(func, sign_specs, auth_ins, choose_auth=True)
            # _taint_bb DFS follows bb.successors and can reach func.exit (which has a
            # .measurement_end macro but is NOT in func._all_bb).  Slots inserted into
            # func.exit are invisible to _find_slot_insts / post-step-6 scan.  Drop any
            # auth anchor whose bb is not in func._all_bb before proceeding.
            _valid_bbs = set(func._all_bb)
            auth_ins = {k: v for k, v in auth_ins.items() if v[0] in _valid_bbs}

            # Step 4/5a: running-taint data-flow pass — builds xpac_slots and
            # standalone_insertions.
            xpac_insertions: List[Tuple[Instruction, BasicBlock, List[Instruction], List[Instruction]]] = []
            standalone_insertions: List[Tuple[Instruction, BasicBlock, List[Instruction]]] = []
            all_pac_at_auth: Dict[Instruction, Dict[str, frozenset]] = {}
            func_taint_log: List[str] = []
            slot_counter = self._build_xpac_slots(
                func, sign_ins, auth_ins, slot_counter,
                fix_points, xpac_insertions, standalone_insertions,
                all_pac_at_auth, func_taint_log)
            self.last_taint_log.extend(func_taint_log)

            # Step 5b: build AUTIA slots — identical layout to XPAC-before-memory slots.
            # Stage1 uses XPACI at pos 8 (safe strip, never faults) so the execution
            # completes cleanly.  Stage2 replaces pos 8 with the real AUTH instruction.
            auth_slot_insertions: List[Tuple[Instruction, BasicBlock, List[Instruction]]] = []
            for anchor, (bb, info) in auth_ins.items():
                # _build_xpac_slots records all_pac_at_auth for every anchor in auth_ins,
                # so this is always populated regardless of path.
                assert anchor in all_pac_at_auth, (
                    f"auth-slot anchor {anchor.name} not in all_pac_at_auth — "
                    f"unreachable BB or topo traversal bug"
                )
                norm_reg = self._norm_reg(info.reg)
                pac_insts_set = all_pac_at_auth[anchor].get(norm_reg)
                if pac_insts_set is None:
                    # Topo/DFS join-node disagreement: DFS reached this anchor via a path
                    # where the reg was signed, but on at least one other incoming path it
                    # was not.  The topological intersection already dropped the reg, so AUTH
                    # here would operate on an unsigned pointer on those paths — skip it.
                    continue
                assert pac_insts_set, (
                    f"all_pac_at_auth has empty frozenset for {norm_reg} at anchor "
                    f"{anchor.name} — invariant violation in _build_xpac_slots"
                )

                sid = slot_counter
                slot_counter += 1
                xpac = self._make_xpac_inst(info.xpac_mnemonic, info.reg, sid, AUTH_SLOT_POS)
                nops = [self._make_nop(sid, pos) for pos in range(AUTH_SLOT_POS)]
                slot_insts = nops + [xpac]

                fix_points.append(FixPoint(
                    slot_id=sid, bb=bb, mem_inst=anchor, info=info,
                    pac_insts=pac_insts_set,
                    slot_insts=slot_insts))
                auth_slot_insertions.append((anchor, bb, slot_insts))

            # Step 6: perform all insertions
            # Pre-check: every fix_point slot added this function must appear in some insertion list
            _xpac_slot_ids = {s._pac_slot_id for _, _, sl, _ in xpac_insertions for s in sl}
            _auth_slot_ids = {s._pac_slot_id for _, _, sl in auth_slot_insertions for s in sl}
            _all_pending = _xpac_slot_ids | _auth_slot_ids
            for fp in fix_points:
                assert fp.slot_id in _all_pending, (
                    f"pre-step6 bug: slot_id={fp.slot_id} is in fix_points but NOT in any "
                    f"insertion list (xpac={sorted(_xpac_slot_ids)}, auth={sorted(_auth_slot_ids)})"
                )
            for next_inst, (pac_inst, bb) in sign_ins.items():
                for s in sandbox_ins.get(next_inst, []):
                    bb.insert_before(next_inst, s)
                bb.insert_before(next_inst, pac_inst)
            for anchor, bb, slot_insts in auth_slot_insertions:
                for s in slot_insts:
                    bb.insert_before(anchor, s)
                # Verify: all slot_insts must now be reachable in bb
                _live = {id(i) for i in bb}
                for s in slot_insts:
                    assert id(s) in _live, (
                        f"stage1 insertion bug: auth slot_id={s._pac_slot_id} "
                        f"pos={s._pac_slot_pos} inst NOT found in bb after insert_before(anchor={anchor.name})"
                    )
            for mem_inst, bb, slot_insts, offset_subs in xpac_insertions:
                for s in slot_insts:
                    bb.insert_before(mem_inst, s)
                for s in offset_subs:
                    bb.insert_before(mem_inst, s)
                # Verify: all slot_insts must now be reachable in bb
                _live = {id(i) for i in bb}
                for s in slot_insts:
                    assert id(s) in _live, (
                        f"stage1 insertion bug: xpac slot_id={s._pac_slot_id} "
                        f"pos={s._pac_slot_pos} inst NOT found in bb after insert_before(mem_inst={mem_inst.name})"
                    )
            for mem_inst, bb, offset_subs in standalone_insertions:
                for s in offset_subs:
                    bb.insert_before(mem_inst, s)

            # Post-step-6 sanity: every fix_point slot added THIS function must be in tc now.
            _slot_ids_now = set()
            for _func2 in tc.functions:
                for _bb2 in _func2:
                    for _inst2 in _bb2:
                        if hasattr(_inst2, '_pac_slot_id'):
                            _slot_ids_now.add(_inst2._pac_slot_id)
            for fp in fix_points:
                assert fp.slot_id in _slot_ids_now, (
                    f"post-step6 bug: slot_id={fp.slot_id} missing from tc after insertions "
                    f"(xpac_insertions={[e[0].name+'@slot'+str(e[2][0]._pac_slot_id) for e in xpac_insertions]}, "
                    f"auth_slot_insertions={[e[0].name+'@slot'+str(e[2][0]._pac_slot_id) for e in auth_slot_insertions]}, "
                    f"slot_ids_in_tc={sorted(_slot_ids_now)})"
                )

        return tc, fix_points

    # ------------------------------------------------------------------
    # Stage 2: produce TC1 / TC2 / TC3 from stage-1 result
    # ------------------------------------------------------------------

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

    def _make_slot_insts(
        self, fp: FixPoint, include_ctx: bool, include_ptr: bool, include_auth: bool
    ) -> List[Instruction]:
        """Create a fresh SLOT_SIZE instruction list (never reuse objects between fills)."""
        sid = fp.slot_id
        if include_ctx and fp.info.ctx_reg is not None:
            if fp.ctx_value is None:
                raise RuntimeError(
                    f"PAC stage2: ctx_value is None for slot {fp.slot_id} "
                    f"(ctx_reg={fp.info.ctx_reg}) but include_ctx=True — "
                    f"CE failed to capture the context register at signing"
                )
            ctx = self._make_load_imm_insts(sid, fp.info.ctx_reg, fp.ctx_value, CTX_SLOT_START)
        else:
            ctx = self._nops_range(sid, CTX_SLOT_START, FIX_COUNT_CTX)
        if include_ptr:
            if fp.signed_value is None:
                raise RuntimeError(
                    f"PAC stage2: signed_value is None for slot {fp.slot_id} "
                    f"(reg={fp.info.reg}) but include_ptr=True — "
                    f"CE failed to capture the signed pointer"
                )
            ptr = self._make_load_imm_insts(sid, fp.info.reg, fp.signed_value, PTR_SLOT_START)
        else:
            ptr = self._nops_range(sid, PTR_SLOT_START, FIX_COUNT_PTR)
        auth_inst = self._make_auth_inst(fp.info.auth_mnemonic, fp.info.reg, fp.info.ctx_reg)
        auth_inst._pac_slot_id = sid
        auth_inst._pac_slot_pos = AUTH_SLOT_POS
        last = [auth_inst] if include_auth else self._nops_range(sid, AUTH_SLOT_POS, 1)
        return ctx + ptr + last

    def _make_tc1_slot_insts(self, fp: FixPoint) -> List[Instruction]:
        """TC1 slot: ctx_restore (0-3), MOVZ+MOVK×3 (4-7), XPAC (8).

        Positions 4-7: full 64-bit load of signed_value (CE-captured PACIA output),
        identical to TC2's ptr restore.  XPAC at position 8 strips the CE's PAC to
        produce the canonical address.  This avoids depending on runtime bit55 to
        reconstruct bits[63:48], which caused a kernel crash when hardware's PACIA
        placed the PAC signature into bit55 (wide PAC implementations).
        After XPAC: reg = P — same as TC2's AUTIA result.
        """
        sid = fp.slot_id
        if fp.info.ctx_reg is not None:
            if fp.ctx_value is None:
                raise RuntimeError(
                    f"PAC stage2 TC1: ctx_value is None for slot {fp.slot_id} "
                    f"(ctx_reg={fp.info.ctx_reg}) — CE failed to capture context register"
                )
            ctx = self._make_load_imm_insts(sid, fp.info.ctx_reg, fp.ctx_value, CTX_SLOT_START)
        else:
            ctx = self._nops_range(sid, CTX_SLOT_START, FIX_COUNT_CTX)
        if fp.signed_value is None:
            raise RuntimeError(
                f"PAC stage2 TC1: signed_value is None for slot {fp.slot_id} "
                f"(reg={fp.info.reg}) — CE failed to capture the signed pointer"
            )
        ptr = self._make_load_imm_insts(sid, fp.info.reg, fp.signed_value, PTR_SLOT_START)
        xpac = self._make_xpac_inst(fp.info.xpac_mnemonic, fp.info.reg, sid, AUTH_SLOT_POS)
        return ctx + ptr + [xpac]

    def instrument_stage2(
        self, prep_tc: TestCase, fix_points: List[FixPoint]
    ) -> Tuple[TestCase, TestCase, TestCase]:
        """
        Produce TC1/TC2/TC3 variants for the *current* input from the preparation TC.

        Call once per input after populating FixPoint.{signed_value, ctx_value,
        spec_nesting} from the contract trace for that input.

        Slot layout (SLOT_SIZE = 9):
          positions 0-3 : context restore  (MOVZ/MOVK for ctx_reg, or NOPs)
          positions 4-7 : pointer restore  (MOVZ/MOVK for signed ptr, or NOPs)
          position  8   : AUTH instruction

        TC1 — [ctx_restore, MOVZ+MOVK×3, XPAC]  full 64-bit CE-signed value loaded then
               stripped by XPAC to produce the canonical pointer
        TC2 — [ctx_restore, ptr_restore, AUTH]  always correct
        TC3 — arch slots identical to TC2; speculative slots each independently
              draw a random combination from:
                (False, False) no restore
                (True,  False) ctx only
                (False, True)  ptr only
                (True,  True)  both restore

        Raises RuntimeError if signed_value is None, or if ctx_value is None
        for a non-zero-context variant (ctx_reg is not None).
        ctx_value being None is valid for zero-context variants (e.g. pacdza).

        Returns (tc1, tc2, tc3).
        """
        tc1 = copy.deepcopy(prep_tc)
        tc2 = copy.deepcopy(prep_tc)
        tc3 = copy.deepcopy(prep_tc)
        maps = {
            'tc1': self._find_slot_insts(tc1),
            'tc2': self._find_slot_insts(tc2),
            'tc3': self._find_slot_insts(tc3),
        }

        _combos = [(False, False), (True, False), (False, True), (True, True)]

        for fp in fix_points:
            sid = fp.slot_id
            # spec_nesting=0      → definitively arch  → TC3 identical to TC2
            # spec_nesting>0      → explicitly spec     → TC3 uses random combo
            # spec_nesting=None   → CE never executed signing (non-arch path) — hardware
            #                       may speculatively execute it → treat as spec for TC3
            is_spec = fp.spec_nesting != 0

            if fp.signed_value is not None:
                # Values captured by CE: fill TC1 (ctx+ptr+XPAC) and TC2 (ctx+ptr+AUTH) normally.
                self._fill_slot(maps['tc1'], fp, self._make_tc1_slot_insts(fp))
                self._fill_slot(maps['tc2'], fp, self._make_slot_insts(fp, True, True, True))
            elif not is_spec:
                # Arch slot without a captured value is a CE capture bug — raise loudly.
                raise RuntimeError(
                    f"PAC stage2: signed_value is None for arch slot {fp.slot_id} "
                    f"(reg={fp.info.reg}) — CE failed to capture the signed pointer"
                )
            # else: spec/non-arch slot, CE never executed the signing instruction.
            # Stage-1 XPAC placeholder is kept in TC1 and TC2 for this slot.

            if not is_spec:
                # Arch slots: TC3 identical to TC2 — must succeed architecturally.
                self._fill_slot(maps['tc3'], fp, self._make_slot_insts(fp, True, True, True))
            else:
                # Spec/non-arch slots: pick a random ctx/ptr restore combination, excluding
                # any combo that requires a value CE did not capture.
                available = [
                    (c, p) for (c, p) in _combos
                    if not (p and fp.signed_value is None)
                    and not (c and fp.info.ctx_reg is not None and fp.ctx_value is None)
                ]
                include_ctx, include_ptr = random.choice(available or [(False, False)])
                self._fill_slot(maps['tc3'], fp, self._make_slot_insts(fp, include_ctx, include_ptr, True))

        return tc1, tc2, tc3


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
    ) -> Tuple[TestCase, TestCase, TestCase]:
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

        tc1 = copy.deepcopy(prep_tc)
        tc2 = copy.deepcopy(prep_tc)
        tc3 = copy.deepcopy(prep_tc)
        maps = {
            'tc1': self._find_slot_insts(tc1),
            'tc2': self._find_slot_insts(tc2),
            'tc3': self._find_slot_insts(tc3),
        }

        for fp in fix_points:
            # spec_nesting=0    → arch path → TC2=NOP, TC3=NOP (correct tag preserved)
            # spec_nesting>0    → spec path → TC2=IRG, TC3=MOVK wrong tag
            # spec_nesting=None → CE never executed this memory access (non-arch path);
            #                     hardware may speculate on it → treat as spec
            is_spec = fp.spec_nesting != 0

            self._fill_slot(maps['tc1'], fp, self._make_mte_nop(fp.slot_id))

            if is_spec:
                self._fill_slot(maps['tc2'], fp, self._make_mte_irg(fp.reg, fp.slot_id))
            else:
                self._fill_slot(maps['tc2'], fp, self._make_mte_nop(fp.slot_id))

            if is_spec:
                self._fill_slot(maps['tc3'], fp,
                                self._make_mte_movk_wrong_tag(fp.reg, wrong_upper16, fp.slot_id))
            else:
                self._fill_slot(maps['tc3'], fp, self._make_mte_nop(fp.slot_id))

        return tc1, tc2, tc3


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
