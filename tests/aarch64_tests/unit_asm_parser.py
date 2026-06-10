"""
Tests for the AArch64 template-driven assembly parser (aarch64_asm_parser).

Regression coverage for the parse->reprint roundtrip, in particular memory addressing
modes (offset / pre-index / post-index), which a previous heuristic parser collapsed
(`[x3, #6212]` was mis-parsed as post-index `[x3], #6212`), and 32- vs 64-bit register
disambiguation driven by each operand's allowed `values`.
"""
import copy
import re
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.config import CONF
from src.isa_loader import InstructionSet
from src.interfaces import OT
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_asm_parser import Aarch64AsmParser

_PARSER = None
_SAVED_CONF = None


def setUpModule():
    global _PARSER, _SAVED_CONF
    # CONF is a Borg singleton; snapshot it so CONF.load() here does not leak
    # config into other test modules during a full run.
    _SAVED_CONF = copy.deepcopy(CONF._borg_shared_state)
    CONF.load("config.yml")
    isa = InstructionSet("base.json", CONF.instruction_categories)
    _PARSER = Aarch64AsmParser(Aarch64RandomGenerator(isa, 0))


def tearDownModule():
    CONF._borg_shared_state.clear()
    CONF._borg_shared_state.update(_SAVED_CONF)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


class Aarch64AsmParserTest(unittest.TestCase):
    def _parse(self, line: str):
        return _PARSER.parse_line(line, 0, _PARSER.instruction_map)

    def _assert_roundtrip(self, line: str):
        """Parsing then reprinting `line` yields the same instruction (modulo whitespace/case)."""
        inst = self._parse(line)
        self.assertEqual(_norm(inst.to_asm_string()), _norm(line))
        return inst

    # --- addressing modes (the roundtrip bug) ---------------------------------------------------
    def test_immediate_offset_not_collapsed_to_post_index(self):
        # The regression: `[x3, #6212]` must stay an offset, NOT become post-index `[x3], #6212`.
        inst = self._assert_roundtrip("LDR  w0, [x3, #6212]")
        self.assertNotIn("], #", inst.to_asm_string())

    def test_pre_index_writeback_preserved(self):
        inst = self._assert_roundtrip("STR  x0, [x1, #-247]!")
        self.assertTrue(inst.to_asm_string().rstrip().endswith("]!"))

    def test_post_index_preserved(self):
        self._assert_roundtrip("LDR  w5, [x3], #100")

    def test_register_offset_preserved(self):
        self._assert_roundtrip("LDR  x5, [x3, x0]")

    def test_bare_base_preserved(self):
        self._assert_roundtrip("LDR  w3, [x2]")

    # --- 32- vs 64-bit disambiguation (driven by operand values) --------------------------------
    def test_w_variant_is_32bit(self):
        inst = self._parse("SUBS  w1, w2, w5")
        self.assertTrue(all(o.width == 32 for o in inst.operands if o.type == OT.REG))

    def test_x_variant_is_64bit(self):
        inst = self._parse("SUBS  x1, x2, x5")
        self.assertTrue(all(o.width == 64 for o in inst.operands if o.type == OT.REG))

    # --- other forms ----------------------------------------------------------------------------
    def test_conditional_branch_captures_cond(self):
        inst = self._parse("B.eq  .bb_0.1")
        conds = [o.value for o in inst.operands if o.type == OT.COND]
        self.assertEqual(conds, ["eq"])

    def test_shifted_register_preserved(self):
        self._assert_roundtrip("AND  w4, w1, w3, lsl #13")

    def test_tbz_preserved(self):
        self._assert_roundtrip("TBZ  w3, #5, .bb_0.2")

    def test_immediate_operand_preserved(self):
        self._assert_roundtrip("ADDS  w4, w3, #1417")


if __name__ == "__main__":
    unittest.main()
