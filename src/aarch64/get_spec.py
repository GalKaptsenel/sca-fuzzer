"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import glob
import json
import os
import re
import copy
import subprocess
from typing import List, Optional, Tuple
import numpy as np
from xml.etree import ElementTree as ET
from .aarch64_target_desc import Aarch64TargetDesc

class OperandSpec:
    values: List[str]
    name: str
    type_: str
    width: int
    signed: bool = True
    comment: str
    src: bool = False
    dest: bool = False
    magic: bool = False

    def __init__(self, name, type, values, src, dest, width, signed):
        self.name = name
        self.type_ = type
        self.values = values
        self.src = src
        self.dest = dest
        self.width = width
        self.signed = signed


    def to_json(self) -> str:
        values_lower = []
        for v in self.values:
            values_lower.append(v.lower())
        self.values = values_lower
        return json.dumps(self, default=vars)


class InstructionSpec:
    name: str
    category: str = ""
    control_flow: bool = False
    operands: List[OperandSpec]
    implicit_operands: List[OperandSpec]
    template: str = ""
    datatype: str = ""

    def __init__(self) -> None:
        self.operands = []
        self.implicit_operands = []

    def __str__(self) -> str:
        return f"{self.name} {self.control_flow} {self.category} " \
               f"{len(self.operands)} {len(self.implicit_operands)}"

    def to_json(self) -> str:
        s = "{"
        s += f'"name": "{self.name.lower()}", "category": "{self.category}", '
        s += f'"control_flow": {str(self.control_flow).lower()},\n'
        s += '  "operands": [\n    '
        s += ',\n    '.join([o.to_json() for o in self.operands])
        s += '\n  ],\n'
        if self.implicit_operands:
            s += '  "implicit_operands": [\n    '
            s += ',\n    '.join([o.to_json() for o in self.implicit_operands])
            s += '\n  ]'
        else:
            s += '  "implicit_operands": []'
        s += ", \n"
        s += f'"template": "{self.template}"'
        s += "\n}"
        return s


class ParseFailed(Exception):
    pass


class Aarch64Transformer:
    tree: ET.ElementTree
    instructions: List[InstructionSpec]
    current_spec: InstructionSpec
    reg_sizes = {
        **{f"x{i}": 64 for i in range(31)},
        **{f"w{i}": 32 for i in range(31)},
        **{f"v{i}": 128 for i in range(32)},
        **{f"q{i}": 128 for i in range(32)},
        **{f"d{i}": 64 for i in range(32)},
        **{f"s{i}": 32 for i in range(32)},
        **{f"h{i}": 16 for i in range(32)},
        **{f"b{i}": 8 for i in range(32)},
        "sp": 64, "wsp": 32, "xzr": 64, "wzr": 32,
        "fpsr": 32, "fpcr": 32,
        **{f"sctlr_el{i}": 64 for i in range(1, 4)},
        "ttbr0_el1": 64, "ttbr1_el1": 64,
        "tcr_el1": 64,
        "mair_el1": 64, "amair_el1": 64,
        **{f"vbar_el{i}": 64 for i in range(1, 4)},
        "spsr_el1": 64, "elr_el1": 64,
        **{f"tpidr_el{i}": 64 for i in range(3)},
        "tpidrro_el0": 64,
        ** {f"dbgbvr{i}": 64 for i in range(16)},
        **{f"dbgbcr{i}": 32 for i in range(16)},
        **{f"dbgwvr{i}": 64 for i in range(16)},
        **{f"dbgwcr{i}": 32 for i in range(16)},
        "daif": 32, "esr_el1": 32, "esr_el2": 32,
        "far_el1": 64, "far_el2": 64,
        "icc_ctlr_el1": 32, "icc_iar1_el1": 32,
        "cntvct_el0": 64, "cntkctl_el1": 64, "pmccntr_el0": 64,
        "sysregs": 64
    }

    not_control_flow = ["int", "int1", "int3", "into"]
    """ a list of instructions that have RIP as an operand but should
    not be considered as control-flow instructions by the generator"""

    def __init__(self, files) -> None:
        self.instructions = []
        self.tree = Aarch64Transformer.load_files(files)

    @staticmethod
    def load_files(self, files: List[str]):
        # get the data from all files
        root = ET.Element("root")
        tree = ET.ElementTree(root)

        for filename in files:
            data = ET.parse(filename).getroot()
            root.append(data)

        return tree

    @staticmethod
    def parse_instruction_docvars(docvars_element: ET.Element) -> Tuple[str, str]:
        address_form = ""
        category = ""
        for op_node in docvars_element.iter("docvar"):
            if op_node.attrib["key"] == "instr-class":
                category = op_node.attrib["value"]
            elif op_node.attrib["key"] == "address-form":
                address_form = op_node.attrib["value"]

        return category, address_form


    @staticmethod
    def parse_explanations(mnemonic: str, explanations: ET.Element, variant_name: str, variable: str, hint: str) -> OperandSpec:
        def set_if_memory_access(mnemonic: str, hint: str) -> Tuple[Optional[str], Optional[str]]:
            def is_store(mnemonic: str, hint: str) -> bool:
                store_prefixes = ["str", "stg", "stp", "st2", "stl", "stn", "sts", "stt", "stu", "stx", "stz"]
                return mnemonic[:3].lower() in ["str", "stg"]  #store_prefixes

            def is_load(mnemonic: str, hint: str) -> bool:
                load_prefixes = ["ldr", "ldg", "ldn", "ldp", "ldt", "ldu", "ldx"]
                return mnemonic[:3].lower() in ["ldr", "ldg"] #load_prefixes

            if is_store(mnemonic, hint):
                src = "MEM" != hint
                dest = "MEM" == hint
            elif is_load(mnemonic, hint):
                src = "MEM" == hint
                dest = "MEM" != hint
            else:
                src = None
                dest = None
            return src, dest


        def is_correct_explanation_clause(explnation: ET.Element, variant_name: str, variable: str) -> bool:

            if not any(variant_name.strip() == item.strip() for item in enclist_value.split(",")):
                return False

            if not any(variable in symbol.text for symbol in explanation.findall(".//symbol")):
                return False

            return True


        def handle_table(table: ET.Element) -> Tuple[List[str], str, str, str]:
            tbody = table.findall('.//tbody')
            table_intro= explanation.findall('.//intro')
            assert table_intro and len(table_intro) == 1, "Each <explanation> assumed to contain a single table intro"
            intro = table_intro[0]

            hint = "IMM"
            src = True
            dest = False

            if "standard conditions" in intro.text:
                hint = "COND"
                src = False
                dest = False

            assert len(tbody) == 1, "Each <explanation> assumed to contain a single table body"

            entries = tbody[0].findall('.//entry[@class="symbol"]')

            values = []

            for entry in entries:
                if entry.text == "RESERVED":
                    continue
                values.append(entry.text)

            return values, hint, src, dest

        def handle_account(account: ET.Element, hint: str, src: Optional[str], dest: Optional[str]) -> Tuple[List[str], str, str, str, str, str]:
            def _get_full_text(element):
                text = element.text or ""
                for child in element:
                    text += (child.text or "")
                    text += (child.tail or "")
                return text

            account_text = _get_full_text(account.find(".//para"))
            src = any(word in account_text for word in ["loaded", "source"]) if src is None else src
            dest = any(word in account_text for word in ["stored", "destination"]) if dest is None else dest
            values = []
            signed = False
            width = 0
            number_of_bits = re.search(r"(\d+)-bit", account_text)
            match_range = re.search(r"\[([+-]?\d+)-([+-]?\d+)\]", account_text)
            if number_of_bits:
                width = int(number_of_bits.group(1))
            elif "the number" in account_text:
                if not match_range:
                    raise ParseFailed("Unavailable options for numbers")
                hint = "IMM"
                src = True
                dest = False
                a, b = map(int, match_range.groups())
#               values = list([str(i) for i in range(a, b+1)])
                values = [f'[{a}-{b}]']
                signed = a < 0
                width = int(np.log2(b-a+1))
                return values, hint, src, dest, width, signed

            if "label" in account_text:
                hint = "LABEL"
                src = True
                dest = False
                signed = True
            elif "general-purpose" in account_text and width > 0:
                assert "register" in account_text
                assert number_of_bits
                assert width in Aarch64TargetDesc.registers
                values = Aarch64TargetDesc.registers[width]
#               For now we don't support access using sp, because it requires allignment to 16 bytes
#                   if " wsp" in account_text:
#                       values.append("wsp")
#                   if " sp" in account_text:
#                       values.append("sp")

            elif "SIMD" in account_text and width > 0:
                assert number_of_bits
                assert width in Aarch64TargetDesc.simd_registers
                values = Aarch64TargetDesc.simd_registers[width]

            elif "scalable vector" in account_text:
                values = Aarch64TargetDesc.sve_scalable_vector_registers
                width = 0 # it is implementation specific

            elif "scalable predicate" in account_text:
                values = Aarch64TargetDesc.sve_predicate_registers
                width = 0 # it is implementation specific

            elif "IMM" == hint:
                range_match = re.search(r"([+-]?\d+) to ([+-]?\d+)", account_text)
                if range_match:

                    a, b = map(int, range_match.groups())
                    signed = a < 0
                    width = int(np.log2(b-a+1))

                    multiple_match = re.search(r"multiple of ([+]?\d+)", account_text)
                    m = 1
                    if multiple_match:
                        m = int(multiple_match.group(1))

                    values = [str(i) for i in range(a, b+1) if i % m == 0]
#                       else:
#                           values = [f'[{a}-{b}]']
                else:
                    raise ParseFailed(
                            f"Unexpected Error Parsing Explanation Fields of {variant_name}:{variable}")
                signed = "unsigned" not in account_text
                src = True
                dest = False
            else:
                raise ParseFailed(
                            f"Unexpected Error Parsing Explanation Fields of {variant_name}:{variable}")

            return values, hint, src, dest, width, signed

        width = 0
        src, dest = set_if_memory_access(mnemonic, hint)
        values = []
        signed = False
        for explanation in explanations.findall(".//explanation"):
            enclist_value = explanation.get("enclist", "")

            if not is_correct_explanation_clause(explanation, variant_name, variable):
                continue

            account_elements = explanation.findall('.//account')
            table_elements = explanation.findall('.//table')

            assert len(account_elements) + len(
                table_elements) == 1, "Each <explanation> assumed to contain either an <account> or a <table>"

            if len(table_elements) == 1:
                table = table_elements[0]
                values, hint, src, dest = handle_table(table)

            elif len(account_elements) == 1:
                account = account_elements[0]
                values, hint, src, dest, width, signed = handle_account(account, hint, src, dest)

            else:
                raise ParseFailed(f"Unexpected Error Parsing Explanation Fields of {variant_name}:{variable}")

        return OperandSpec(variable, hint, values, src, dest, width, signed)

    @staticmethod
    def parse_instruction_variant(instruction_variant_element: ET.Element, instruction_spec: InstructionSpec, explanations: ET.Element) -> List[InstructionSpec]:
        asmtemplate = instruction_variant_element.findall(".//asmtemplate/*")
        if asmtemplate is None:
            return

        asmtemplate_data = ""
        for word in asmtemplate:
            asmtemplate_data += word.text

        return Aarch64Transformer.parse_instruction_variant_inner(
            instruction_variant_element.get("name"),asmtemplate_data,instruction_spec,0,
            explanations, 0)

    @staticmethod
    def parse_instruction_variant_inner(variant_name: str, template: str,
                                        current_instruction: InstructionSpec, recursive: int,
                                        explanations: ET.Element, index: int) -> List[InstructionSpec]:

        def find_closing_bracket(s: str, index: int) -> Optional[int]:
            bracket_map = {'(': ')', '{': '}', '[': ']', "<": ">"}

            if index < 0 or index >= len(s) or s[index] not in bracket_map:
                return None

            stack = []

            for i in range(index, len(s)):
                char = s[i]

                if char in bracket_map:
                    stack.append(char)

                elif char in bracket_map.values():
                    if not stack:
                        return None
                    last_open = stack.pop()
                    if bracket_map[last_open] == char:
                        if not stack:
                            return i

            return None

        if index == len(template):
            return [current_instruction]

        special_characters = "<>!|(){}#[]"

        default_hint = "REG"
        hint = default_hint

        while index < len(template):
            ch = template[index]
            if ch not in special_characters:
                current_instruction.template += ch
                index += 1
            else:
                if ch in "<":

                    closing_bracket_index = find_closing_bracket(template, index)
                    variable = template[index+1:closing_bracket_index]
                    current_instruction.template += f"{{{variable}}}"
                    operand = Aarch64Transformer.parse_explanations(current_instruction.name, explanations, variant_name,
                                                                   variable, hint)

                    # TODO: make it cleaner and clearer! should not be here!
                    if recursive != 0:
                        operand.type_ = "MEM"

                    current_instruction.operands.append(operand)
                    index = closing_bracket_index + 1
                    hint = default_hint if recursive == 0 else hint
                elif ch in "{(":
                    closing_bracket_index = find_closing_bracket(template, index)
                    if closing_bracket_index is None:
                        raise ParseFailed(f"Did not close opening bracket: {template}")
                    assert closing_bracket_index < len(template)
                    if ch == "{":
                        template_inst1 = (template[:index] + template[index+1:closing_bracket_index] +
                                          template[closing_bracket_index + 1:])
                        template_inst2 = template[:index] + template[closing_bracket_index + 1:]
                        inst1 = Aarch64Transformer.parse_instruction_variant_inner(variant_name,
                                                                                   template_inst1,
                                                                                   copy.deepcopy(current_instruction),
                                                                                   recursive,
                                                                                   explanations, index)
                        inst2 = Aarch64Transformer.parse_instruction_variant_inner(variant_name,
                                                                                   template_inst2,
                                                                                   copy.deepcopy(current_instruction),
                                                                                   recursive,
                                                                                   explanations, index)
                        return inst1 + inst2
                    if ch == "(":
                        template_inner = template[index + 1:closing_bracket_index]
                        values = []
                        for option in template_inner.split("|"):
                            new_template = template[:index] + option.strip() + template[closing_bracket_index + 1:]
                            values += (Aarch64Transformer.parse_instruction_variant_inner(
                                variant_name, new_template, copy.deepcopy(current_instruction), recursive, explanations, index))
                        return values

                elif ch in "})|":
                    raise ParseFailed(f"Unexpected symbol: '{ch}' in {template}")
                elif ch == "!":
                    current_instruction.template += "!"
                    index += 1
                elif ch == "[":
                    current_instruction.template += "["
                    hint = "MEM"
                    recursive += 1
                    index += 1
                elif ch == "]":
                    current_instruction.template += "]"

                    if recursive <= 0:
                        raise ParseFailed(f"Unexpected symbol: '{ch}' in {template}")

                    recursive -= 1

                    if recursive == 0:
                        hint = default_hint
                    index += 1
                elif ch == "#":
                    current_instruction.template += "#"
                    hint = "IMM"
                    index += 1
                else:
                    raise ParseFailed(f"Unexpected character: {template}")


        return [current_instruction]

    @staticmethod
    def postprocessing_instruction(instruction: InstructionSpec) -> bool:
        """ Not all valid encodings form a logical instruction """
        has_extend = False
        has_general_purpose_regiers_32_bit_for_memory = (False, None)
        has_general_purpose_regiers_64_bit_for_memory = (False, None)
        has_lsl = False
        has_amount = False

        for operand in instruction.operands:

            if operand.name == "extend":
                has_extend = True

                for value in  operand.values:
                    if value.lower() == "lsl":
                        has_lsl = True


            elif operand.name == "amount":
                has_amount = True

            elif operand.type_ == "MEM":
                for value in  operand.values:
                    if value in Aarch64TargetDesc.registers[32]:
                        has_general_purpose_regiers_32_bit_for_memory = (True, operand)

                    if value in Aarch64TargetDesc.registers[64]:
                        has_general_purpose_regiers_64_bit_for_memory = (True, operand)

            elif operand.type_ == "LABEL":

                if not instruction.control_flow:

                    return False # TODO: Currentlu, any instruction that has a label but not a control flow, is not supported as the label is assumed (wrongly) to be of a branch (for example PRFM). Additionally, for some reason cbcc instructions considered non-control flow




        # is the 64-bit operands inside the memory clause come before the 32-bit operand, we need an extend clause
        if has_general_purpose_regiers_64_bit_for_memory[0] and \
                has_general_purpose_regiers_32_bit_for_memory[0] \
                and not has_extend:
                    op32 = has_general_purpose_regiers_32_bit_for_memory[1]
                    op64 = has_general_purpose_regiers_64_bit_for_memory[1]
                    assert op32 is not None
                    assert op64 is not None
                    name_to_search_op32 = f"{{{op32.name}}}"
                    name_to_search_op64 = f"{{{op64.name}}}"
                    if instruction.template.find(op64.name) < instruction.template.find(op32.name):
                        return False

        if has_extend:
            return False
            if has_general_purpose_regiers_32_bit_for_memory[0]:
                has_lsl = False
                valid_32_bit_extends = ["uxtw", "sxtw"]
                valid_extends = valid_32_bit_extends
            elif has_general_purpose_regiers_64_bit_for_memory[0]:
                valid_64_bit_extends = ["lsl", "uxtx", "sxtx"]
                valid_extends = valid_64_bit_extends
            else:
                return False

            for operand in instruction.operands:
                if operand.name == "extend":
                    operand.values = [v for v in operand.values if isinstance(v, str) and v.lower() in valid_extends]

        if has_lsl and not has_amount:
            return False



        return True



    def parse_instruction(self, instruction_node, explanations: ET.Element) -> List[InstructionSpec]:

        docvars = instruction_node.find("docvars")
        if not docvars:
            return []

        category, address_form = Aarch64Transformer.parse_instruction_docvars(docvars)

        # TODO: temporary
        if category != "general":
            return []

        flags_op = Aarch64Transformer.get_flags_from_asl(instruction_node)


        current_spec = InstructionSpec()

        variants = []
        # get all asm variants of the instructions
        for variant in instruction_node.findall("classes/iclass/encoding"):
            current_spec = InstructionSpec()

            # instruction info
            docvars = variant.find("docvars")
            assert docvars
            for op_node in docvars.iter("docvar"):
                if op_node.attrib["key"] == "instr-class":
                    current_spec.category = op_node.attrib["value"]
                elif op_node.attrib["key"] == "branch-offset":
                    current_spec.control_flow = True
                elif op_node.attrib["key"] == "address-form":
                    address_form = op_node.attrib["value"]
                elif op_node.attrib["key"] == "datatype":
                    current_spec.datatype = op_node.attrib["value"]

            if address_form not in ["literal", "base-register", "post-indexed", ""]:
                continue

            current_spec.name = variant.find("asmtemplate/text").text.split(" ")[0]

            # implicit PC operand
            if current_spec.control_flow:
                op_pc = OperandSpec(name="pc", type="REG", values=["PC"],
                                    src=True, dest=False, width=64, signed=False)
                current_spec.implicit_operands.append(op_pc)

            # implicit flags operand
            if flags_op:
                current_spec.implicit_operands.append(flags_op)

            try:
                encodings = Aarch64Transformer.parse_instruction_variant(variant, current_spec, explanations)

                for encoding in encodings:
                    if Aarch64Transformer.postprocessing_instruction(encoding):
                        variants.append(encoding)

            except ParseFailed:
                continue

        return variants

    def parse_tree(self):
       for instruction_node in self.tree.iter('instructionsection'):
            if instruction_node.attrib['type'] != "instruction":
                continue
            explanations = instruction_node.find(".//explanations")
            if explanations is None:
                continue

            encodings = self.parse_instruction(instruction_node, explanations)
            self.instructions.extend(encodings)

    @staticmethod
    def get_flags_from_asl(element: ET.Element) -> Optional[OperandSpec]:
        """ look through the ASL code of the instruction to find if it reads/writes the flags """
        flag_values = {k: [False, False] for k in ["N", "Z", "C", "V", ""]}
        uses_flags = False
        for line in element.find("ps_section/ps/pstext").itertext():
            match = re.search(r"(=?) *PSTATE\.<?([NZCV,]+)>? *(=?)", line)
            if match:
                affected_flags = match.group(2).split(",")
                is_read = (match.group(1) == "=")
                is_write = (match.group(3) == "=")
                if not is_read and not is_write:
                    continue

                uses_flags = True
                for f in affected_flags:
                    flag_values[f][0] |= is_read
                    flag_values[f][1] |= is_write
        if uses_flags:
            flag_op = OperandSpec(name="flags", type="FLAGS", values=[],
                                  src=False, dest=False, width=0, signed=False)

            # the loop maps aarch64 flags to x86 eflags, which is the basis for out flags data structure
            for f in ["C", "", "", "Z", "N", "", "", "", "V"]:
                if flag_values[f][0] and flag_values[f][1]:
                    flag_op.values.append("r/w")
                elif flag_values[f][0] and not flag_values[f][1]:
                    flag_op.values.append("r")
                elif not flag_values[f][0] and flag_values[f][1]:
                    flag_op.values.append("w")
                else:
                    flag_op.values.append("")
        else:
            flag_op = None
        return flag_op

    def save(self, filename: str):
        json_str = "[\n" + ",\n".join([i.to_json() for i in self.instructions]) + "\n]"
        # print(json_str)
        with open(filename, "w+") as f:
            f.write(json_str)

class Downloader:
    def __init__(self, extensions: List[str],  out_file: str) -> None:
        self.extensions = extensions
        self.out_file = out_file

    def run(self):
        file_name = "ISA_A64_xml_A_profile-2024-12"
        print("> Downloading complete instruction spec...")
        subprocess.run(
            "wget "
            f"https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-A64-ISA/ISA_A64/{file_name}.tar.gz",
            shell=True,
            check=True)

        subprocess.run("mkdir -p all", shell=True, check=True)
        subprocess.run("mkdir -p instructions", shell=True, check=True)
        subprocess.run(f"tar xf {file_name}.tar.gz -C all", shell=True, check=True)
        subprocess.run(
            f"mv all/{file_name}/*.xml instructions/", shell=True, check=True)
        subprocess.run(f"rm {file_name}.tar.gz*", shell=True, check=True)
        os.remove("instructions/encodingindex.xml")
        subprocess.run("rm -r all", shell=True, check=True)

        files = glob.glob("instructions/*.xml")

        print("\n> Filtering and transforming the instruction spec...")
        transformer = Aarch64Transformer(files)
        transformer.parse_tree()
        print(f"Produced base.json with {len(transformer.instructions)} instructions")
        transformer.save("base.json")
        subprocess.run("rm -r instructions", shell=True, check=True)

