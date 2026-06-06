"""
Unit tests for arm_isa_parser.tags.get_tags:
  - NZCV writers tagged BASE-NZCV / SVE-NZCV (writers only, not readers)
  - SVE instructions never get BASE-* flag tags
  - no name-prefix guesses and no BASE-MISC catch-all (explicit tags only)
"""
import unittest
from types import SimpleNamespace as NS

from src.aarch64.arm_isa_parser.tags import get_tags


def op(type_, values=(), src=False, dest=False):
    return NS(type_=type_, values=list(values), src=src, dest=dest)


def inst(name, operands=(), implicit=(), control_flow=False):
    return NS(name=name, operands=list(operands),
              implicit_operands=list(implicit), control_flow=control_flow)


# FLAGS operand value layouts (per-flag access strings)
FLAGS_W = op("FLAGS", values=["w", "", "", "w", "w", "", "", "", "w"])
FLAGS_R = op("FLAGS", values=["r", "", "", "r", "r", "", "", "", "r"])
SVE_REG = op("REG", values=["z0"])      # an SVE Z register -> has_sve_regs
GP_REG = op("REG", values=["x0"])


class TagsTest(unittest.TestCase):
    def test_base_flag_writer_gets_base_nzcv(self):
        tags = get_tags(inst("adds", operands=[GP_REG], implicit=[FLAGS_W]))
        self.assertIn("BASE-NZCV", tags)
        self.assertNotIn("SVE-NZCV", tags)

    def test_flag_reader_not_tagged_nzcv(self):
        tags = get_tags(inst("adc", operands=[GP_REG], implicit=[FLAGS_R]))
        self.assertNotIn("BASE-NZCV", tags)
        self.assertNotIn("SVE-NZCV", tags)

    def test_sve_flag_writer_gets_sve_nzcv_not_base(self):
        # SVE flag-writing instruction (has SVE register + writes flags)
        tags = get_tags(inst("somesvecmp", operands=[SVE_REG], implicit=[FLAGS_W]))
        self.assertIn("SVE-NZCV", tags)
        self.assertNotIn("BASE-NZCV", tags)

    def test_no_prefix_guess(self):
        # Unknown mnemonic starting with 'b' must NOT be guessed as a branch.
        self.assertEqual(get_tags(inst("bogusxyz", operands=[GP_REG])), [])

    def test_no_misc_catch_all(self):
        # Unmatched instruction gets no tags (no BASE-MISC fallback).
        self.assertEqual(get_tags(inst("totallyunknown", operands=[GP_REG])), [])


if __name__ == "__main__":
    unittest.main()
