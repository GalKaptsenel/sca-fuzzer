"""TEMPORARY test — DELETE when the branch-mistraining bug is fixed and the gate is removed.

`CONF.enable_branch_mistraining` is gated OFF *only* because the current mistraining implementation
has a bug: it trains toward the architectural direction and thereby suppresses the natural
misprediction Spectre-v1 needs (see docs/aarch64 §3.2 and memory project_spectrev1_mistrain_regression).
This pins that the gate actually disables mistraining (`branch_mistraining_entries` returns []), and
that the gate is the *only* thing suppressing it (with the gate on, the same trace yields an entry).

Once the mistraining direction/path is fixed on hardware and mistraining is re-enabled by default,
the gate — and therefore this whole test file — should be removed.
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
        self._saved = CONF.enable_branch_mistraining

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
