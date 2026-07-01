"""
Tests for the AArch64 side of the (now arch-agnostic) violation minimizer.

The minimizer engine (postprocessor.MainMinimizer + passes) is shared across architectures;
the arch-specific parts are sourced from the TargetDesc minimizer hooks:
  - speculation_barrier  : the fence inserted by FenceInsertionPass / skipped by the loop
  - asm_header           : header for a standalone asm file
  - nop_replacement(line): how NopReplacementPass neutralizes an instruction
  - is_branch_line(line) : control-flow detection used by the nop/fence passes

These tests pin the AArch64 hooks (HW-free). End-to-end reduction runs on hardware via
`revizor.py minimize` since it needs /dev/executor.
"""
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # any cwd
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc
from src.aarch64.aarch64_generator import Aarch64DsbSyPass
from src.interfaces import BasicBlock, Function, TestCase, Instruction


def _tc_with(mnemonics):
    """A single-BB test case with the given instructions; returns (test_case, basic_block)."""
    tc = TestCase(0)
    func = Function(".function_0", tc.actors["main"])
    bb = BasicBlock(".bb0")
    prev = None
    for m in mnemonics:
        inst = Instruction(m, False, "", False, template=m)
        bb.insert_after(prev, inst)
        prev = inst
    func.append(bb)
    tc.functions.append(func)
    return tc, bb


class Aarch64MinimizerHooksTest(unittest.TestCase):
    def setUp(self):
        self.t = Aarch64TargetDesc()

    # --- fence / header -------------------------------------------------------------------------
    def test_speculation_barrier_is_dsb_sy(self):
        self.assertEqual(self.t.speculation_barrier, "dsb sy")

    def test_asm_header_empty(self):
        # AArch64 uses GNU as default syntax; no .intel_syntax-style header is needed.
        self.assertEqual(self.t.asm_header, "")

    # --- nop replacement (fixed 4-byte instructions -> single NOP) ------------------------------
    def test_plain_instruction_maps_to_nop(self):
        self.assertEqual(self.t.nop_replacement("add x0, x1, x2"), "nop")

    def test_memory_instruction_maps_to_nop(self):
        self.assertEqual(self.t.nop_replacement("ldr x0, [x1, #8]"), "nop")

    def test_existing_nop_is_not_replaced(self):
        self.assertIsNone(self.t.nop_replacement("nop"))

    def test_branches_are_not_neutralized(self):
        for branch in ("b .bb_0", "bl .f", "br x3", "blr x3", "ret",
                       "cbz x1, .l", "cbnz x1, .l", "tbz x1, #3, .l", "tbnz x1, #3, .l",
                       "b.eq .l", "b.ne .l"):
            self.assertIsNone(self.t.nop_replacement(branch), f"branch neutralized: {branch}")

    def test_indentation_and_case_tolerated(self):
        self.assertEqual(self.t.nop_replacement("    ADD X0, X1, X2"), "nop")
        self.assertIsNone(self.t.nop_replacement("    B.EQ .l"))

    # --- branch detection -----------------------------------------------------------------------
    def test_is_branch_line(self):
        self.assertTrue(self.t.is_branch_line("b .bb_0"))
        self.assertTrue(self.t.is_branch_line("b.eq .l"))
        self.assertTrue(self.t.is_branch_line("cbz x1, .l"))
        self.assertFalse(self.t.is_branch_line("ldr x0, [x1]"))
        self.assertFalse(self.t.is_branch_line("bic x0, x1, x2"))   # 'b'-prefixed but not a branch
        self.assertFalse(self.t.is_branch_line("bfi x0, x1, #2, #4"))


class Aarch64DsbSyPassTest(unittest.TestCase):
    """The full-fencing pass (used by create_fenced_test_case for the observation filter) must put a
    DSB SY after every instruction, i.e. a fence between every two instructions."""

    def test_fence_after_every_instruction(self):
        tc, bb = _tc_with(["add x0, x1, x2", "ldr x3, [x0]", "eor x4, x4, x4"])
        Aarch64DsbSyPass().run_on_test_case(tc)
        self.assertEqual([i.name for i in bb],
                         ["add x0, x1, x2", "DSB SY", "ldr x3, [x0]", "DSB SY", "eor x4, x4, x4", "DSB SY"])

    def test_single_instruction(self):
        tc, bb = _tc_with(["add x0, x1, x2"])
        Aarch64DsbSyPass().run_on_test_case(tc)
        self.assertEqual([i.name for i in bb], ["add x0, x1, x2", "DSB SY"])


if __name__ == "__main__":
    unittest.main()
