"""
File: x86 implementation of the test case generator

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import abc
import math
import random
import copy
from itertools import chain
from typing import List, Tuple, Optional

from ..isa_loader import InstructionSet
from ..interfaces import TestCase, Operand, Instruction, BasicBlock, Function, InstructionSpec, \
    GeneratorException, RegisterOperand, ImmediateOperand, MAIN_AREA_SIZE, FAULTY_AREA_SIZE, \
    MemoryOperand, AgenOperand, OT, OperandSpec, CondOperand
from ..generator import ConfigurableGenerator, RandomGenerator, Pass, Printer
from ..config import CONF
from .aarch64_target_desc import Aarch64TargetDesc
from .aarch64_elf_parser import Aarch64ElfParser


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
            Aarch64PatchUndefinedLoadsPass(self.target_desc),
            Aarch64SandboxPass(),
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
                            to_patch.append(inst)

                # fix operands
                for inst in to_patch:
                    org_dest = inst.operands[0]
                    options = self.target_desc.registers[org_dest.width]
                    options = [i for i in options if i != org_dest.value]
                    new_value = random.choice(options)
                    inst.operands[0].value = new_value


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


class Aarch64MarkMemoryAccesses(Pass):

    @staticmethod
    def mark_memory_access(bb: BasicBlock, inst: Instruction):

        access_id = inst.memory_access_id

        if not (0 <= access_id <= 127):
            raise ValueError("SVE bit must be between 0 and 127, inclusive")


        sve_register_bitmap = 'z0'
        sve_register_temporary_1 = 'z1'
        sve_register_temporary_2 = 'z2'
        predicate_register = 'p1'

        byte_index = access_id // 8
        bit_shift = access_id % 8

        ptrue_template = f"ptrue {predicate_register}.B, ALL"
        index_template = f"index {sve_register_temporary_2}.B, #0, #1"
        compeq_template = f"cmpeq {predicate_register}.B, {predicate_register}/z, {sve_register_temporary_2}.B, #{byte_index}"
        mov_template = f"mov {sve_register_temporary_1}.B, #0b{1 << bit_shift:08b}"
        orr_template = f"orr {sve_register_bitmap}.B, {predicate_register}/M, {sve_register_bitmap}.B, {sve_register_temporary_1}.B"

        ptrue_instruction = Instruction("PTRUE", True, template=ptrue_template)
        index_instruction = Instruction("INDEX", True, template=index_template)
        cmpeq_instruction = Instruction("CMPEQ", True, template=compeq_template)
        mov_instruction = Instruction("MOV", True, template=mov_template)
        orr_instruction = Instruction("ORR", True, template=orr_template)

        bb.insert_after(position=inst, inst=orr_instruction)
        bb.insert_after(position=inst, inst=mov_instruction)
        bb.insert_after(position=inst, inst=cmpeq_instruction)
        bb.insert_after(position=inst, inst=index_instruction)
        bb.insert_after(position=inst, inst=ptrue_instruction)

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:

                memory_instructions = []

                for inst in bb:
                    if inst.has_memory_access:
                        memory_instructions.append(inst)

                for inst in memory_instructions:
                    Aarch64MarkMemoryAccesses.mark_memory_access(bb, inst)


class Aarch64SandboxPass(Pass):
    def __init__(self):
        super().__init__()

        input_memory_size = MAIN_AREA_SIZE + FAULTY_AREA_SIZE
        mask_size = int(math.log(input_memory_size, 2))
        self.sandbox_address_mask = "0b" + "1" * mask_size

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:

                # collect all instructions that require sandboxing
                memory_instructions = []

                for inst in bb:
                    if inst.has_mem_operand(True):
                        memory_instructions.append(inst)

                # sandbox them
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

            x30_register = RegisterOperand("x30", 64, True, False)
            x30_register.name = "x30_reg"
            template, op0, op1, op2 = generate_template("ADD", base_operand_copy, base_operand_copy,
                                                        x30_register)
            add_base = Instruction("ADD", True).add_op(op0).add_op(op1).add_op(op2)
            add_base.template = template
            parent.insert_before(instr, add_base)

            # for op in mem_operands[1:]:
            #
            #     template, op0, op1, op2 = generate_template("SUB", base_operand_copy,
            #                                                 base_operand_copy, op)
            #     op2.dest = False
            #     op2.src = True
            #     sub_inst = Instruction("SUB", True).add_op(op0).add_op(op1).add_op(op2)
            #     sub_inst.template = template  # TODO: this should be done in the constructor
            #     parent.insert_before(instr, sub_inst)

            return

        if implicit_mem_operands:
            raise GeneratorException("Implicit memory accesses are not supported")

        raise GeneratorException("Attempt to sandbox an instruction without memory operands")


class Aarch64Printer(Printer):
    prologue_template = [
        ".test_case_enter:\n",
    ]

    epilogue_template = [
        ".section .data.main\n",
        ".test_case_exit:\n",
    ]

    def __init__(self, _: Aarch64TargetDesc) -> None:
        super().__init__()

    def print(self, test_case: TestCase, outfile: str) -> None:
        with open(outfile, "w") as f:
            # print prologue
            for line in self.prologue_template:
                f.write(line)

            # print the test case
            for func in test_case.functions:
                self.print_function(func, f)

            # print epilogue
            for line in self.epilogue_template:
                f.write(line)

    def print_function(self, func: Function, file):
        file.write(f".section .data.{func.owner.name}\n")
        file.write(f"{func.name}:\n")
        for bb in func:
            self.print_basic_block(bb, file)

        self.print_basic_block(func.exit, file)

    def print_basic_block(self, bb: BasicBlock, file):
        file.write(f"{bb.name.lower()}:\n")
        for inst in bb:
            file.write(self.instruction_to_str(inst) + "\n")
        for inst in bb.terminators:
            file.write(self.instruction_to_str(inst) + "\n")

    def instruction_to_str(self, inst: Instruction):
        if inst.name == "macro":
            return self.macro_to_str(inst)

        values = {}
        for op in inst.operands:
            values[op.name] = op.value

        instruction = inst.template.format(**values)

        if inst.is_instrumentation:
            comment = "// instrumentation"
        elif inst.is_noremove:
            comment = "// noremove"
        else:
            comment = ""
        return f"{instruction} {comment}"

    def operand_to_str(self, op: Operand) -> str:
        if isinstance(op, MemoryOperand) or isinstance(op, AgenOperand):
            return f"[{op.value}]"

        if isinstance(op, ImmediateOperand) or isinstance(op, AgenOperand):
            return f"#{op.value}"

        return op.value

    def macro_to_str(self, inst: Instruction):
        macro_placeholder = "NOP"
        if inst.operands[1].value.lower() == ".noarg":
            return f".macro{inst.operands[0].value}: {macro_placeholder}"
        else:
            return f".macro{inst.operands[0].value}{inst.operands[1].value}: {macro_placeholder}"


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
