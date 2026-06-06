"""
File: AArch64 assembly layout + printing — render a TestCase's IR to assembly text.
"""
from typing import List, Dict

from ..interfaces import TestCase, Function, BasicBlock, Instruction
from ..generator import Printer
from .aarch64_target_desc import Aarch64TargetDesc


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

        for inst in list(bb) + bb.terminators:
            self.instruction_address[inst] = self._instruction_counter * 4
            self.content.append(self._instruction_to_str(inst))
            self._instruction_counter += 1

    def _instruction_to_str(self, inst: Instruction) -> str:
        if inst.name == "macro":
            return self._macro_to_str(inst)

        instruction = inst.to_asm_string()

        if inst.is_instrumentation:
            comment = "// instrumentation"
        elif inst.is_noremove:
            comment = "// noremove"
        else:
            comment = ""

        return f"{instruction} {comment}"

    # Macros currently expand to exactly one instruction (a NOP placeholder).
    def _macro_to_str(self, inst: Instruction) -> str:
        macro_placeholder = "NOP"
        if inst.operands[1].value.lower() == ".noarg":
            return f".macro{inst.operands[0].value}: {macro_placeholder}"
        return f".macro{inst.operands[0].value}{inst.operands[1].value}: {macro_placeholder}"



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
