"""branch_mistraining_entries is gated by CONF.enable_branch_mistraining. This pins the gate (off =>
no mistraining even for a trainable branch) and the training polarity (on => the branch is saturated
OPPOSITE its architectural direction, so it mispredicts on first execution).

The gate is off by default while mistraining is WIP, pending hardware confirmation that the training
is effective on this core. Remove the gate (and this test) once mistraining is enabled by default.
"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.config import CONF                                          # noqa: E402
from src.aarch64.aarch64_executor import Aarch64LocalExecutor        # noqa: E402

_CBZ = 0xb4000040  # `cbz x0, .+8` — a conditional branch (verified is_conditional_branch == True)


def _cer_with_branch():
    """Minimal CE trace: a nest-0 conditional branch at 0x1000 whose nest-0 successor is 0x1008
    (!= pc+4), i.e. architecturally taken → branch_mistraining_entries would emit one entry."""
    branch = SimpleNamespace(cpu=SimpleNamespace(pc=0x1000, encoding=_CBZ),
                             metadata=SimpleNamespace(speculation_nesting=0))
    succ = SimpleNamespace(cpu=SimpleNamespace(pc=0x1008, encoding=0),
                           metadata=SimpleNamespace(speculation_nesting=0))
    return [branch, succ]


class TestBranchMistrainingGate(unittest.TestCase):

    def setUp(self):
        self._saved = getattr(CONF, "enable_branch_mistraining", False)

    def tearDown(self):
        CONF.enable_branch_mistraining = self._saved

    def test_gate_off_disables_mistraining(self):
        # The crux of the v1-detection fix: gate off ⇒ no mistraining even for a trainable branch.
        CONF.enable_branch_mistraining = False
        self.assertEqual(
            Aarch64LocalExecutor.branch_mistraining_entries(None, _cer_with_branch()), [])

    def test_gate_on_emits_entry(self):
        # Confirms the gate is the only suppressor: the branch IS trainable when the gate is on.
        CONF.enable_branch_mistraining = True
        self.assertEqual(
            Aarch64LocalExecutor.branch_mistraining_entries(None, _cer_with_branch()), [(0, False)])


if __name__ == "__main__":
    unittest.main()
