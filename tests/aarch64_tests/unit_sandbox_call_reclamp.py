"""Regression: the sandbox-clamp walk must re-clamp a memory base that is reused AFTER a function call.

A call (BL/BLR) transfers to a generated callee that may clobber any caller register, so a base
clamped before the call cannot be assumed in-region afterward. The dataflow used to carry the
"already in-region" fact across the call and elide the re-clamp, so a clobbered base was reused
unclamped -> an architectural out-of-sandbox access that the contract executor fail-louds
([CE FATAL] kaddr2uaddr ... spec_nesting=0). Device-free; drives SandboxWalk directly.
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.interfaces import (TestCase, Function, BasicBlock, Instruction, MemoryOperand,  # noqa: E402
                            RegisterOperand, Actor, ActorMode, ActorPL, OT)
from src.aarch64.aarch64_target_desc import AArch64MemRole, Aarch64TargetDesc  # noqa: E402
from src.aarch64.seal.sealer import SandboxWalk, SandboxSealing  # noqa: E402


def _str_to_base(base_reg: str, data_reg: str) -> Instruction:
    """A plain `STR <data>, [<base>]` (no writeback, no offset), base-role tagged."""
    base = RegisterOperand(base_reg, 64, True, False)
    base.mem_role = AArch64MemRole.BASE
    mem = MemoryOperand(base_reg, 64, False, True, inner=[base])     # store -> mem is a dest
    inst = Instruction("str", False, "", False, has_memory_access=True,
                       template="STR {Xt}, [{Xn}]")
    inst.add_op(RegisterOperand(data_reg, 64, True, False))
    inst.add_op(mem)
    return inst


def _blr(reg: str) -> Instruction:
    inst = Instruction("blr", False, "CALL", True, template="BLR {reg}")
    inst.add_op(RegisterOperand(reg, 64, True, False))
    return inst


def _add(dst: str) -> Instruction:
    """A benign non-call instruction that does NOT clobber the base (dst != base)."""
    inst = Instruction("add", False, "", False, template="ADD {Xd}, {Xn}, {Xm}")
    inst.add_op(RegisterOperand(dst, 64, False, True))
    inst.add_op(RegisterOperand(dst, 64, True, False))
    inst.add_op(RegisterOperand(dst, 64, True, False))
    return inst


def _tc_with(middle: Instruction) -> TestCase:
    """A one-function test case: STR [x4]; <middle>; STR [x4] — same base reused after `middle`."""
    tc = TestCase(0)
    actor = tc.actors["main"]
    func = Function(".function_0", actor)
    bb = BasicBlock(".bb_0.0")
    for inst in (_str_to_base("x4", "x0"), middle, _str_to_base("x4", "x1")):
        bb.insert_after(bb.end, inst)
    func.append(bb)
    tc.functions.append(func)
    return tc


def _walk():
    gen = types.SimpleNamespace(
        target_desc=Aarch64TargetDesc(),
        instruction_set=types.SimpleNamespace(instructions=[]))   # no MTE tag store -> granule floor 1
    return SandboxWalk(gen)


class CallReclampTest(unittest.TestCase):
    def test_reclamp_after_call(self):
        """A base reused after a BLR is clamped a SECOND time (the callee may have clobbered it)."""
        sandbox, data_sites = _walk().sandbox(_tc_with(_blr("x28")))
        self.assertEqual(len(sandbox), 2,
                         "base x4 must be re-clamped after the call, not assumed in-region")
        self.assertEqual(len(data_sites), 2, "both accesses are data sites")

    def test_no_reclamp_without_call(self):
        """Control: with a benign non-call between the two accesses, the base provably survives, so it
        is clamped only ONCE (the fix is specific to calls; it does not over-clamp straight-line code)."""
        sandbox, _ = _walk().sandbox(_tc_with(_add("x5")))
        self.assertEqual(len(sandbox), 1, "a surviving base must not be needlessly re-clamped")

    def test_reclamp_inserts_and_add_before_second_access(self):
        """The re-clamp is the AND/ADD sandbox pair, inserted before the post-call access."""
        tc = _tc_with(_blr("x28"))
        _walk().sandbox(tc)
        templates = [i.template for i in tc.functions[0][0]]
        # two AND x4,... ; ADD x4,...,x29 pairs must now be present (one per access)
        self.assertEqual(sum(t.startswith("ADD x4, x4, x29") for t in templates), 2, templates)
        self.assertEqual(sum(t.startswith("AND x4, x4,") for t in templates), 2, templates)


if __name__ == "__main__":
    unittest.main()
