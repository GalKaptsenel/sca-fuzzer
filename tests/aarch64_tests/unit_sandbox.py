"""Regression tests for the AArch64 sandbox: role-aware base masking + index/offset/extend
cancellation, and the base!=index constraint. Runnable from any cwd (path bootstrap from __file__)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.interfaces import (MemoryOperand, RegisterOperand, ImmediateOperand,   # noqa: E402
                            Instruction, OT)
from src.aarch64 import aarch64_generator as g   # noqa: E402
from src.aarch64.aarch64_target_desc import AArch64MemRole, Aarch64TargetDesc   # noqa: E402


def _mem(*components, src=True, dest=False):
    """components: (role|None, value, OT, width) -> a MemoryOperand wrapping them as inner operands."""
    inner = []
    for role, val, ot, width in components:
        op = RegisterOperand(val, width, True, False) if ot is OT.REG else ImmediateOperand(val, width)
        op.mem_role = AArch64MemRole(role) if role else None
        inner.append(op)
    return MemoryOperand(", ".join(c[1] for c in components), 64, src, dest, inner=inner)


class _Sandbox(g._SandboxInstrumentationBase):
    _norm = {}
    _sandbox_mask = "#0x1fff"
    _sandbox_base_reg = "x29"


class SandboxCancellationTest(unittest.TestCase):
    """Mask the base into [x29 .. x29+mask], then cancel index/offset/extend so EA == masked base."""

    def setUp(self):
        self.s = _Sandbox()

    def _cancel(self, mem):
        return [i.template for i in self.s._make_offset_sub_insts(mem)]

    def test_base_only_just_masks(self):
        mem = _mem(("base", "x1", OT.REG, 64))
        self.assertEqual([i.template for i in self.s._make_sandbox_insts(self.s._base_reg(mem))],
                         ["AND x1, x1, #0x1fff", "ADD x1, x1, x29"])
        self.assertEqual(self._cancel(mem), [])

    def test_positive_displacement_subtracts(self):
        mem = _mem(("base", "x1", OT.REG, 64), ("offset", "8", OT.IMM, 9))
        self.assertEqual(self._cancel(mem), ["SUB x1, x1, #8"])

    def test_negative_displacement_adds(self):
        mem = _mem(("base", "x1", OT.REG, 64), ("offset", "-16", OT.IMM, 9))
        self.assertEqual(self._cancel(mem), ["ADD x1, x1, #16"])

    def test_index_subtracts_register(self):
        mem = _mem(("base", "x1", OT.REG, 64), ("index", "x2", OT.REG, 64))
        self.assertEqual(self._cancel(mem), ["SUB x1, x1, x2"])

    def test_index_with_extend_and_amount(self):
        mem = _mem(("base", "x1", OT.REG, 64), ("index", "w2", OT.REG, 32),
                   ("extend", "sxtw", OT.IMM, 0), ("offset", "2", OT.IMM, 0))
        self.assertEqual(self._cancel(mem), ["SUB x1, x1, w2, sxtw #2"])

    def test_index_with_lsl_shift(self):
        mem = _mem(("base", "x1", OT.REG, 64), ("index", "x2", OT.REG, 64), ("offset", "3", OT.IMM, 0))
        self.assertEqual(self._cancel(mem), ["SUB x1, x1, x2, lsl #3"])

    def test_large_displacement_is_chunked_to_imm12(self):
        mem = _mem(("base", "x1", OT.REG, 64), ("offset", "5000", OT.IMM, 12))
        self.assertEqual(self._cancel(mem), ["SUB x1, x1, #4095", "SUB x1, x1, #905"])


class BaseIndexCollisionTest(unittest.TestCase):
    """base == index would make `SUB base, base, index` zero the base (sandbox escape); the patch pass
    must force the index register to differ from the base."""

    def test_colliding_index_is_replaced(self):
        patch = g.Aarch64PatchUndefinedLoadsStoresPass(Aarch64TargetDesc())
        base = RegisterOperand("x3", 64, True, False)
        base.mem_role = AArch64MemRole.BASE
        index = RegisterOperand("x3", 64, True, False)
        index.mem_role = AArch64MemRole.INDEX
        mem = MemoryOperand("x3, x3", 64, True, False, inner=[base, index])
        inst = Instruction("ldr", False, "", False, template="LDR {Xt}, [{Xn}, {Xm}]")
        inst.operands = [RegisterOperand("x0", 64, False, True), mem]

        patch._patch_address_register_collision(inst)
        self.assertNotEqual(patch._norm(index.value), patch._norm(base.value))


class PairExclusiveConstraintTest(unittest.TestCase):
    """UNPREDICTABLE register-collision rules for pair / exclusive load-store, patched by
    Aarch64PatchUndefinedLoadsStoresPass. Construct an instruction with a deliberate collision, run
    the pass, and assert the offending register was changed (and innocent ones were not)."""

    def setUp(self):
        self.patch = g.Aarch64PatchUndefinedLoadsStoresPass(Aarch64TargetDesc())

    def _reg(self, name, width=64, src=False, dest=False):
        return RegisterOperand(name, width, src, dest)

    def _mem_base(self, name, writeback=False, store=False):
        base = RegisterOperand(name, 64, True, writeback)   # base.dest == writeback
        base.mem_role = AArch64MemRole.BASE
        return MemoryOperand(name, 64, not store, store, inner=[base])

    def _run(self, name, operands, template):
        inst = Instruction(name, False, "", False, template=template)
        inst.operands = operands
        self.patch._patch_instruction(inst)
        return inst

    def _n(self, v):
        return self.patch._norm(v)

    # ---- LDP (two-dest load) ----
    def test_ldp_equal_dests_are_separated(self):
        d0, d1 = self._reg("x5", dest=True), self._reg("x5", dest=True)
        self._run("ldp", [d0, d1, self._mem_base("x2")], "LDP {Xt1}, {Xt2}, [{Xn}]")
        self.assertNotEqual(self._n(d0.value), self._n(d1.value))

    def test_ldp_no_writeback_allows_dest_equal_base(self):
        # Unsigned-offset LDP: dest aliasing the base is permitted (no writeback) — base untouched.
        d0, d1 = self._reg("x2", dest=True), self._reg("x3", dest=True)
        mem = self._mem_base("x2", writeback=False)
        self._run("ldp", [d0, d1, mem], "LDP {Xt1}, {Xt2}, [{Xn}, #16]")
        self.assertEqual(self._n(mem.inner[0].value), self._n("x2"))
        self.assertEqual(self._n(d0.value), self._n("x2"))   # not forced away from the base

    def test_ldp_writeback_dests_differ_from_base(self):
        d0, d1 = self._reg("x2", dest=True), self._reg("x3", dest=True)
        mem = self._mem_base("x2", writeback=True)
        self._run("ldp", [d0, d1, mem], "LDP {Xt1}, {Xt2}, [{Xn}], #16")
        base = self._n(mem.inner[0].value)
        self.assertNotEqual(self._n(d0.value), base)
        self.assertNotEqual(self._n(d1.value), base)
        self.assertNotEqual(self._n(d0.value), self._n(d1.value))

    # ---- STP writeback (two-src store) ----
    def test_stp_writeback_srcs_differ_from_base(self):
        s0, s1 = self._reg("x2", src=True), self._reg("x3", src=True)
        mem = self._mem_base("x2", writeback=True, store=True)
        self._run("stp", [s0, s1, mem], "STP {Xt1}, {Xt2}, [{Xn}], #16")
        base = self._n(mem.inner[0].value)
        self.assertNotEqual(self._n(s0.value), base)
        self.assertNotEqual(self._n(s1.value), base)

    def test_stp_no_writeback_allows_src_equal_base(self):
        s0, s1 = self._reg("x2", src=True), self._reg("x3", src=True)
        mem = self._mem_base("x2", writeback=False, store=True)
        self._run("stp", [s0, s1, mem], "STP {Xt1}, {Xt2}, [{Xn}, #16]")
        self.assertEqual(self._n(s0.value), self._n("x2"))   # unchanged

    # ---- LDXP (exclusive two-dest load: dests distinct + distinct from base, no writeback field) ----
    def test_ldxp_dests_and_base_all_distinct(self):
        d0, d1 = self._reg("x2", dest=True), self._reg("x2", dest=True)
        mem = self._mem_base("x2")
        self._run("ldxp", [d0, d1, mem], "LDXP {Xt1}, {Xt2}, [{Xn}]")
        norms = {self._n(d0.value), self._n(d1.value), self._n(mem.inner[0].value)}
        self.assertEqual(len(norms), 3)

    # ---- STXR (status must not alias data or base) ----
    def test_stxr_status_distinct_from_data_and_base(self):
        status = self._reg("w1", width=32, dest=True)
        data = self._reg("x1", src=True)
        mem = self._mem_base("x2", store=True)
        self._run("stxr", [status, data, mem], "STXR {Ws}, {Xt}, [{Xn}]")
        self.assertNotIn(self._n(status.value),
                         {self._n(data.value), self._n(mem.inner[0].value)})

    # ---- STXP (status must not alias either data reg or the base) ----
    def test_stxp_status_distinct_from_all(self):
        status = self._reg("w1", width=32, dest=True)
        d0, d1 = self._reg("x1", src=True), self._reg("x2", src=True)
        mem = self._mem_base("x3", store=True)
        self._run("stxp", [status, d0, d1, mem], "STXP {Ws}, {Xt1}, {Xt2}, [{Xn}]")
        self.assertNotIn(self._n(status.value),
                         {self._n(d0.value), self._n(d1.value), self._n(mem.inner[0].value)})

    # ---- single-register writeback (LDR/STR): transferred reg must differ from base ----
    def test_single_load_writeback_dest_differs_from_base(self):
        d = self._reg("x2", dest=True)
        mem = self._mem_base("x2", writeback=True)
        self._run("ldr", [d, mem], "LDR {Xt}, [{Xn}], #8")
        self.assertNotEqual(self._n(d.value), self._n(mem.inner[0].value))

    def test_single_store_writeback_src_differs_from_base(self):
        s = self._reg("x2", src=True)
        mem = self._mem_base("x2", writeback=True, store=True)
        self._run("str", [s, mem], "STR {Xt}, [{Xn}], #8")
        self.assertNotEqual(self._n(s.value), self._n(mem.inner[0].value))

    def test_single_no_writeback_allows_transferred_equal_base(self):
        d = self._reg("x2", dest=True)
        mem = self._mem_base("x2", writeback=False)
        self._run("ldr", [d, mem], "LDR {Xt}, [{Xn}, #8]")
        self.assertEqual(self._n(d.value), self._n("x2"))   # not patched (no writeback)


if __name__ == "__main__":
    unittest.main()
