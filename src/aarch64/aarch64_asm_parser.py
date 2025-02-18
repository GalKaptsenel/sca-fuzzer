"""
File: Parsing of assembly files into our internal representation (TestCase).
      This file contains x86-specific code.

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import re
import os
from typing import List, Dict

from .aarch64_generator import Aarch64Generator
from ..asm_parser import AsmParserGeneric, parser_assert, parser_error
from ..interfaces import OT, Instruction, InstructionSpec, LabelOperand, Operand, RegisterOperand, \
    MemoryOperand, ImmediateOperand, AgenOperand, CondOperand

PATTERN_CONST_INT = re.compile("^-?[0-9]+$")
PATTERN_CONST_HEX = re.compile("^-?0x[0-9abcdef]+$")
PATTERN_CONST_BIN = re.compile("^-?0b[01]+$")
PATTERN_CONST_SUM = re.compile("^-?[0-9]+ *[+-] *[0-9]+$")


class Aarch64AsmParser(AsmParserGeneric):
    generator: Aarch64Generator

    asm_prefixes = []
    asm_synonyms = {
    }
    memory_sizes = {

    }

    def parse_line(self, line: str, line_num: int,
                   instruction_map: Dict[str, List[InstructionSpec]]) -> Instruction:
        line = line.lower()

        # instrumentation?
        is_instrumentation = "instrumentation" in line
        is_noremove = "noremove" in line

        re_tokenize = re.compile(
            r"^([^ .]+\.?)([^ ]+)? ([^ ,]+)(,[^ ,]+)?(,[^ ,]+)?(,[^ ,]+)?( //.*)?")
        matches = re_tokenize.findall(line)
        if not matches:
            re_tokenize_nops = re.compile(r"^([^ .]+\.?)([^ ]+)?")
            matches = re_tokenize_nops.findall(line)

        parser_assert([] != matches, line_num, f"Failure to parse the line: {line}")

        name = matches[0][0]
        operand_tokens = ["COND"] if matches[0][1] else []
        operand_tokens += [op.value.removeprefix(",") for op in matches[0][2:6] if op]
        comment = matches[0][-1][3:]

        # find a spec that describes this instruction
        spec_candidates = instruction_map.get(name, [])
        parser_assert(len(spec_candidates) > 0, line_num, f"Unknown instruction {line}")

        # find a matching spec:
        # - check the number of operands
        matching_specs = [s for s in spec_candidates if len(s.operands) == len(operand_tokens)]

        # - check the other operands
        size_map = {"x": 64, "w": 32, "v": 128, "q": 128, "d": 64, "s": 32, "h": 16, "b": 8}

        for op_id, op_raw in enumerate(operand_tokens):
            if "COND" == op_raw or op_raw in self.target_desc.branch_conditions:
                matching_specs = [s for s in matching_specs if s.operands[op_id].type == OT.COND]
            elif "." == op_raw[0]:  # match label
                matching_specs = [s for s in matching_specs if s.operands[op_id].type == OT.LABEL]
            elif "[" in op_raw:  # match address
                matching_specs = [s for s in matching_specs if s.operands[op_id].type == OT.MEM]
            elif "#" == op_raw[0]:  # match immediate
                matching_specs = [s for s in matching_specs if s.operands[op_id].type == OT.IMM]
            elif "sp" == op_raw:
                matching_specs = [s for s in matching_specs if s.operands[op_id].width == 64]
            elif op_raw[0] in size_map.keys():  # match register
                matching_specs = [s for s in matching_specs if s.operands[op_id].type == OT.REG and
                                  s.operands[op_id].width == size_map[op_raw[0]]]
            elif op_raw in ["SY", "LD", "ST"]:  # match keyword immediate
                matching_specs = [s for s in matching_specs if s.operands[op_id].type == OT.IMM]
            else:
                parser_error(line_num, f"Unknown type of the operand |{op_raw}|")

        parser_assert(
            len(matching_specs) != 0, line_num, f"Could not find a matching spec for {line}")

        # we might find several matches if the instruction has a magic operand value
        if len(matching_specs) > 1:
            magic_value_specs = list(filter(lambda x: (x.has_magic_value), matching_specs))
            if magic_value_specs:
                matching_specs = magic_value_specs

        # at this point we should have only one spec, but even if we don't, all of them should
        # be equivalent. Just pick the first
        spec: InstructionSpec = matching_specs[0]
        inst = Instruction.from_spec(spec, is_instrumentation=is_instrumentation)
        inst.is_noremove = is_noremove

        op: Operand
        for op_id, op_raw in enumerate(operand_tokens):
            op_spec = spec.operands[op_id]
            if op_spec.type == OT.REG:
                op = RegisterOperand(op_raw, op_spec.width, op_spec.src, op_spec.dest)
            elif op_spec.type == OT.MEM:
                address_match = re.search(r'\[(.*)\]', op_raw)
                parser_assert(address_match is not None, line_num, "Invalid memory address")
                address = address_match.group(1)  # type: ignore
                op = MemoryOperand(address, op_spec.width, op_spec.src, op_spec.dest)
            elif op_spec.type == OT.IMM:
                op = ImmediateOperand(op_raw, op_spec.width)
            elif op_spec.type == OT.LABEL:
                assert spec.control_flow
                op = LabelOperand(op_raw)
            elif op_spec.type == OT.COND:
                op = CondOperand(op_raw)
            else:
                parser_error(line_num, f"Unknown spec operand type {op_spec.type}")

            inst.operands.append(op)

        for op_spec in spec.implicit_operands:
            op = self.generator.generate_operand(op_spec, inst)
            inst.implicit_operands.append(op)

        return inst

    def _patch_asm(self, asm_file: str, patched_asm_file: str):
        """
        Ensure function labels are exposed in AArch64:
          - Adds a global function label if missing.
          - Adds NOP at function end for easier size calculations.
          - Inserts `.function_0` if missing.
          - Ensures `.test_case_exit` is within `.data.main` with a single NOP.
        """

        def is_instruction(line: str) -> bool:
            """Check if the line is an actual instruction or data directive."""
            return line and not line.startswith("#") and (
                not line.startswith(".") or
                line.startswith((
                                ".bcd", ".byte", ".long", ".quad", ".macro", ".value", ".2byte",
                                ".4byte", ".8byte"))
            )

        main_function_label = ""
        enter_found = False
        has_measurement_start = False
        has_measurement_end = False
        prev_line = ""

        with open(asm_file, "r") as f:
            with open(patched_asm_file, "w") as patched:
                for line in f:
                    line = line.strip().lower()

                    if line.startswith(".macro.measurement_start"):
                        has_measurement_start = True
                    elif line.startswith(".macro.measurement_end"):
                        has_measurement_end = True

                    if not enter_found:
                        if line == ".test_case_enter:":
                            enter_found = True
                        patched.write(line + "\n")
                        continue

                    if ".test_case_exit:" in line:
                        if not main_function_label:
                            patched.write(".function_0:\n")
                            main_function_label = ".function_0"
                        if ".data.main" not in prev_line or "measurement_end" in prev_line:
                            patched.write(".section .data.main\n")
                        patched.write(".test_case_exit:\n    nop\n")
                        continue

                    if line.startswith(".function_") and not main_function_label:
                        main_function_label = line[:-1]
                    elif not main_function_label and is_instruction(line):
                        patched.write(".function_0:\n")
                        main_function_label = ".function_0"

                    patched.write(line + "\n")
                    prev_line = line

        macro_placeholder = " nop"  # AArch64 NOP (0xd503201f)

        # Add jump placeholders after macros
        with open(patched_asm_file, "r") as f:
            with open(patched_asm_file + ".tmp", "w") as patched:
                for line in f:
                    line = line.lower()
                    if line.startswith(".macro"):
                        if "nop" not in line:
                            patched.write(line[:-1] + macro_placeholder + "\n")
                        else:
                            assert macro_placeholder in line, "Unexpected NOP placeholder: " + line
                            patched.write(line)
                    else:
                        patched.write(line)
        os.rename(patched_asm_file + ".tmp", patched_asm_file)

        # Add .macro.measurement_start after .function_0
        if not has_measurement_start:
            with open(patched_asm_file, "r") as f:
                with open(patched_asm_file + ".tmp", "w") as patched:
                    for line in f:
                        line = line.lower()
                        patched.write(line)
                        if line.startswith(main_function_label):
                            patched.write(".macro.measurement_start:\n    nop\n")
            os.rename(patched_asm_file + ".tmp", patched_asm_file)

        # Add .macro.measurement_end before .test_case_exit
        if not has_measurement_end:
            with open(patched_asm_file, "r") as f:
                with open(patched_asm_file + ".tmp", "w") as patched:
                    prev_line = ""
                    for line in f:
                        line = line.lower()
                        if line.startswith(".test_case_exit:"):
                            if prev_line.startswith(".section"):
                                patched.write(".function_end:\n")
                            patched.write(".macro.measurement_end:\n    nop\n")
                        patched.write(line)
                        prev_line = line
            os.rename(patched_asm_file + ".tmp", patched_asm_file)
