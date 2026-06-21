"""Branch classification: unconditional branches include b/br/ret and calls include bl/blr,
so the asm-parser CFG (fallthrough edges, BB splits) is correct for the indirect forms too."""
import types
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc
from src.interfaces import CondOperand


def _inst(name, operands=()):
    return types.SimpleNamespace(name=name, operands=list(operands))


class BranchClassificationTest(unittest.TestCase):
    def test_unconditional_branches(self):
        for n in ("b", "br", "ret", "B", "RET", "b.al", "b.nv"):
            self.assertTrue(Aarch64TargetDesc.is_unconditional_branch(_inst(n)), n)
        for n in ("bl", "cbz", "add"):
            self.assertFalse(Aarch64TargetDesc.is_unconditional_branch(_inst(n)), n)

    def test_bcond_al_nv_is_unconditional(self):
        # a B.cond carrying an always/never condition is effectively unconditional
        for cond in ("al", "nv"):
            self.assertTrue(
                Aarch64TargetDesc.is_unconditional_branch(_inst("b.cond", [CondOperand(cond)])), cond)
        for cond in ("eq", "ne", "lt"):
            self.assertFalse(
                Aarch64TargetDesc.is_unconditional_branch(_inst("b.cond", [CondOperand(cond)])), cond)

    def test_calls(self):
        for n in ("bl", "blr", "BL"):
            self.assertTrue(Aarch64TargetDesc.is_call(_inst(n)), n)
        for n in ("b", "br", "ret", "add"):
            self.assertFalse(Aarch64TargetDesc.is_call(_inst(n)), n)


if __name__ == "__main__":
    unittest.main()
