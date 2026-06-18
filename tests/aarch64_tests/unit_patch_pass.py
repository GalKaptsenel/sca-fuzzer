"""_replace_reg substitutes a register of the same width AND class: a SIMD-pair dest must not be
replaced by a GPR (which the old width-only pool did), and 128-bit SIMD must not KeyError."""
import types
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.interfaces import RegisterOperand
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc
from src.aarch64.aarch64_generator import Aarch64PatchUndefinedLoadsStoresPass


def _pass():
    p = Aarch64PatchUndefinedLoadsStoresPass.__new__(Aarch64PatchUndefinedLoadsStoresPass)
    p.target_desc = types.SimpleNamespace(
        registers=Aarch64TargetDesc.registers,            # class-level dicts (no /proc/cpuinfo init)
        simd_registers=Aarch64TargetDesc.simd_registers,
        reg_normalized=Aarch64TargetDesc.reg_normalized,
    )
    return p


class ReplaceRegClassTest(unittest.TestCase):
    def test_simd_dest_replaced_by_simd(self):
        p = _pass()
        op = RegisterOperand("d0", 64, True, False)
        p._replace_reg(op, forbidden={p.target_desc.reg_normalized["d0"]})
        self.assertIn(op.value, p.target_desc.simd_registers[64])
        self.assertNotEqual(op.value, "d0")

    def test_gpr_dest_replaced_by_gpr(self):
        p = _pass()
        op = RegisterOperand("x0", 64, True, False)
        p._replace_reg(op, forbidden={p.target_desc.reg_normalized["x0"]})
        self.assertIn(op.value, p.target_desc.registers[64])

    def test_128bit_simd_does_not_keyerror(self):
        p = _pass()
        op = RegisterOperand("v0", 128, True, False)
        p._replace_reg(op, forbidden={p.target_desc.reg_normalized["v0"]})
        self.assertIn(op.value, p.target_desc.simd_registers[128])


if __name__ == "__main__":
    unittest.main()
