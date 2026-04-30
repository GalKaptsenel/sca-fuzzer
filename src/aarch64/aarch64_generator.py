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
from typing import List, Tuple, Optional, Type, Callable, Set, Dict
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
        pac_instructions = [i for i in self.instruction_set.instruction_unfiltered if "BASE-PAC" in i.tags and i.name in CONF.supported_instructions]
        self._signing_instructions = list(filter(lambda i: i.name.lower().startswith('aut'), pac_instructions))
        self._verification_instructions = list(filter(lambda i: i.name.lower().startswith('pac'), pac_instructions))
        self._strip_sign_instructions  = list(filter(lambda i: i.name.lower().startswith('xpac'), pac_instructions))

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




class PACInstrumentation:
    @dataclass(frozen=True)
    class SignPair:
        sign_inst: Instruction
        signed_value: int

    class custom_comperator_operand:
        def __init__(self, op):
            self.op = op
        def __eq__(self, other):
            return self.op.value == other.op.value
        def __hash__(self):
            return hash(self.op.value[1:]) # TODO better to conf.normalize


    def __init__(self, aarch64_generator: Aarch64Generator, xpac_weight: int, auth_weight: int, sign_weight: int):
        xpac_weight = max(xpac_weight, 0)
        auth_weight = max(auth_weight, 0)
        sign_weight = max(sign_weight, 0)
        total_weight = (xpac_weight + auth_weight + sign_weight) * 1.0
        self._xpac_prob = xpac_weight / total_weight
        self._auth_prob = auth_weight / total_weight
        self._sign_prob = sign_weight / total_weight
        self.generator = aarch64_generator

    def _sign_inst(self):
        return self.generator.get_signing_instruction()

    def _auth_inst(self):
        return self.generator.get_verification_instruction()

    def _xpac_inst(self, reg_operand: Optional[Operand] = None):
        return self.generator.get_strip_sign_instruction(reg_operand)

    def _inner_collect_instructions(self, func: Function, pred: Callable[Instruction, bool]) -> Set[Tuple[BasicBlock, Instruction]]:
        collected = set()
        for bb in func:
            for inst in bb:
                if pred(inst):
                    collected.add((bb, inst))

        return collected

    def _build_signs(self, func: Function, layout: Aarch64ASMLayout) -> Dict[int, Tuple[Instruction, BasicBlock, Instruction]]:
        instructions = self._inner_collect_instructions(func, lambda inst: random.random() < self._sign_prob)
        offsets = map(lambda tupl: layout.instruction_address[tupl[1]], instructions)
        signed_instructions: Dict[int, Tuple[Instruction, BasicBlock, Instruction]] = {}
        for offset, (bb, next_instruction) in zip(offsets, instructions):
            signed_instructions[offset] = (self._sign_inst() , bb, next_instruction) # sign instruction
        return signed_instructions

    def _build_authentications(self, func: Function, layout: Aarch64ASMLayout) -> Dict[int, Tuple[Instruction, BasicBlock, Instruction]]:
        instructions = self._inner_collect_instructions(func, lambda inst: random.random() < self._auth_prob)
        offsets = map(lambda tupl: layout.instruction_address[tupl[1]], instructions)
        authentication_instructions: Dict[int, Tuple[Instruction, BasicBlock, Instruction]] = {}
        for offset, (bb, next_instruction) in zip(offsets, instructions):
            authentication_instructions[offset] = (self._auth_inst() , bb, next_instruction) # auth instruction
        return authentication_instructions

    def _build_xpac_stage1(self, func: Function, layout: Aarch64ASMLayout, sign_taints: Dict[Instruction, Set[custom_comperator_operand]]) -> Dict[int, Tuple[Instruction, BasicBlock, Instruction]]:
        instructions = self._inner_collect_instructions(func,
                        lambda inst: inst.has_memory_access and list(filter(lambda op: op.type == OT.MEM and self.custom_comperator_operand(op) in sign_taints[inst], chain(inst.operands, inst.implicit_operands))))
        offsets = map(lambda tupl: layout.instruction_address[tupl[1]], instructions)
        xpac_instructions: Dict[int, Tuple[Instruction, BasicBlock, Instruction]] = {}
        for offset, (bb, next_instruction) in zip(offsets, instructions):
            reg_op_list = list(filter(lambda op: op.type == OT.MEM and self.custom_comperator_operand(op) in sign_taints[next_instruction], chain(next_instruction.operands, next_instruction.implicit_operands)))
            for reg_op in reg_op_list:
                xpac_instructions[offset] = (self._xpac_inst(reg_op), bb, next_instruction) # xpac instruction
        return xpac_instructions

    def _taint_by_instruction(self, inst: Instruction, address: int, signed_op: Set[custom_comperator_operand], address_to_sign_inst: Dict[int, Instruction], address_to_auth_inst: Dict[int, Instruction]) -> Set[custom_comperator_operand]:

        signed_op = set(signed_op)
        if address in address_to_sign_inst:
            for op in address_to_sign_inst[address][0].operands + address_to_sign_inst[address][0].implicit_operands:
                if op.dest:
                    signed_op.add(self.custom_comperator_operand(op))

        if address in address_to_auth_inst:
            for op in address_to_auth_inst[address][0].operands + address_to_auth_inst[address][0].implicit_operands:
                if op.dest:
                    signed_op.add(self.custom_comperator_operand(op))

        for op in inst.operands + inst.implicit_operands:
            if op.dest:
                signed_op.add(self.custom_comperator_operand(op))

        return signed_op

    def _collect_sign_taints_inner(self, bb: BasicBlock, input_signed_set: Set[custom_comperator_operand], inst_to_address: Dict[Instruction, int], address_to_sign_inst: Dict[int, Instruction], address_to_auth_inst: Dict[int, Instruction], visited: Optional[Set] = None) -> Dict[Instruction, Set[custom_comperator_operand]]:
        if visited is None:
            visited = set()
        if bb in visited:
            return {}
        visited.add(bb)

        sign_map: Dict[Instruction, Set[custom_comperator_operand]] = {}
        curr_signed_set  = set(input_signed_set)

        for inst in bb:
            curr_signed_set = self._taint_by_instruction(inst, inst_to_address[inst], curr_signed_set, address_to_sign_inst, address_to_auth_inst)
            sign_map[inst] = curr_signed_set

        for next_bb in bb.successors:
            sign_map.update(self._collect_sign_taints_inner(next_bb, curr_signed_set, inst_to_address, address_to_sign_inst, address_to_auth_inst, visited=visited))

        return sign_map

    def _collect_sign_taints(self, func: Function, inst_to_address: Dict[Instruction, int], address_to_sign_inst: Dict[int, Instruction], address_to_auth_inst: Dict[int, Instruction]) -> Dict[Instruction, Set[custom_comperator_operand]]:
        return self._collect_sign_taints_inner(func.get_first_bb(), set(), inst_to_address, address_to_sign_inst, address_to_auth_inst)

    def instrument_stage1(self, test_case: TestCase) -> TestCase:
        test_case = copy.deepcopy(test_case)
        layout: Aarch64ASMLayout = Aarch64ASMLayout(test_case)
        import pdb;pdb.set_trace()
        for func in test_case.functions:
            sign_inst = self._build_signs(func, layout)
            auth_inst = self._build_authentications(func, layout)
            sign_taints = self._collect_sign_taints(func, layout.instruction_address, sign_inst, auth_inst)
            xpac_inst = self._build_xpac_stage1(func, layout, sign_taints)
            for (new_inst, bb, next_inst) in chain(sign_inst.values(), auth_inst.values(), xpac_inst.values()):
                bb.insert_before(next_inst, new_inst)
        return test_case


    def instrument_stage2(self, test_case: TestCase, contract_trace: ContractExecutionResult, inst_to_ite: Dict[Instruction, InstrTraceEntry]) -> Tuple[TestCase, TestCase, TestCase]:
        layout: Aarch64ASMLayout = Aarch64ASMLayout(test_case)
        for func in test_case.functions:
            sign_inst = self._build_signs(func, layout)
            auth_inst = self._build_authentications(func, layout)
            sign_taints = self._collect_sign_taints(func, sign_inst, auth_inst)
            auth_inst_arch = self._build_arch_norm(func, layout, inst_to_ite, sign_taints)
            auth_inst_spec = self._build_spec_versions(func, layout, inst_to_ite, sign_taints)
            assert 0 == len(auth_inst_arch & auth_inst_spec)
            # add manual fix for correct verification for any arch verification instruction for operand that is dirty (aka written after signing)
            # similar, create 3 options for speculative paths: correct fix, nop and xpac

            for bb, bb_holes in sign_inst.items():
                for inst in bb_holes:
                    ite = mapper[inst]
                    for _ in range(4):
                        bb.insert_before(position=inst, inst=Instruction("nop", False, "", True, template="NOP"))


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
