"""MTE stage-1 taint: a base rewritten by offset_subs (indexed/displaced access) no longer holds
its sandboxed value, so the next same-base access must re-sandbox; a plain [base] access keeps the
optimization (the second access is NOP-only). The helpers are stubbed so no ISA spec is needed --
the taint-set bookkeeping under test is the real code."""
import types
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_seal import Sandbox, CompositeSeal, _SANDBOX_MASK, SealInstrumentation
from src.aarch64.aarch64_mte import MteTag, MTEFixPoint


class _FakeBB:
    def __init__(self, insts):
        self._insts = insts
        self.successors = []

    def __iter__(self):
        return iter(self._insts)


class _FakeFunc:
    def __init__(self, bbs):
        self._bbs = bbs

    def __iter__(self):
        return iter(self._bbs)

    def get_first_bb(self):
        return self._bbs[0]


def _mem_inst():
    return types.SimpleNamespace(name="ldr", has_memory_access=True,
                                 get_mem_operands=lambda: [object()])


def _sandbox_decisions(offset_subs):
    m = SealInstrumentation.__new__(SealInstrumentation)
    m._get_mem_base_reg = lambda inst: "x5"            # same base for both accesses
    m._norm_reg = lambda r: r
    m._make_offset_sub_insts = lambda mem_op: list(offset_subs)
    m._fixpoint_cls = MTEFixPoint
    m._sandbox = Sandbox(_SANDBOX_MASK)
    value_seals = [MteTag()]
    m._value_composite = value_seals[0]
    m._composite = CompositeSeal([m._sandbox] + value_seals)
    m._mte_tag_propagates = lambda inst: None
    m._dest_regs = lambda inst: frozenset()

    insertions = []
    func = _FakeFunc([_FakeBB([_mem_inst(), _mem_inst()])])   # two accesses, same base
    m._build_slots(func, 0, [], insertions, [])
    return [bool(sandbox) for (_i, _b, sandbox, _o, _n) in insertions]


class MTETaintTest(unittest.TestCase):
    def test_indexed_same_base_resandboxes(self):
        first, second = _sandbox_decisions(offset_subs=["SUB"])
        self.assertTrue(first)
        self.assertTrue(second)   # offset_subs left base wild -> must re-sandbox

    def test_plain_same_base_optimized(self):
        first, second = _sandbox_decisions(offset_subs=[])
        self.assertTrue(first)
        self.assertFalse(second)  # base still at its sandboxed value -> NOP-only


if __name__ == "__main__":
    unittest.main()
