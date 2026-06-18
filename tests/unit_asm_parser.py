"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import unittest

from src.asm_parser import AsmParserGeneric


class DataDirectiveTest(unittest.TestCase):
    """ Regression: every GAS data directive must be recognized as an opcode line. Previously
    `.bcd`/`.value`/`.2byte`/`.4byte`/`.8byte` were silently missed (slice bugs), so test cases
    embedding raw bytes were mis-parsed as instructions. """

    def test_all_data_directives_recognized(self):
        for directive in (".bcd", ".byte", ".long", ".quad", ".value", ".2byte", ".4byte", ".8byte"):
            self.assertTrue(
                AsmParserGeneric._is_data_directive(f"{directive} 0x1234"),
                msg=f"`{directive}` not recognized as a data directive")

    def test_non_directives_rejected(self):
        for line in ("add x0, x1, x2", "mov rax, 1", ".function_main:",
                     ".bb_main.entry:", ".macro foo", ""):
            self.assertFalse(
                AsmParserGeneric._is_data_directive(line),
                msg=f"`{line}` wrongly classified as a data directive")


if __name__ == "__main__":
    unittest.main()
