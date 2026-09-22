"""The sandbox-only sealer path that makes PTE fuzzing a STANDALONE feature (no code primitive):
make_sealer must accept the empty primitive set and dispatch it to SandboxSealedTestCase, and still
reject an unknown primitive. No hardware needed (construction only).

    python -m unittest tests.aarch64_tests.unit_pte_sealer
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.aarch64.seal.sealer import (make_sealer, Sealer, SealedTestCase,           # noqa: E402
                                     SandboxSealedTestCase)


def _stub_generator():
    """The minimal surface SandboxWalk touches at construction (no capstone, no ISA load)."""
    return types.SimpleNamespace(
        instruction_set=types.SimpleNamespace(instructions=[]),
        target_desc=types.SimpleNamespace(reg_normalized={}))


class SandboxOnlySealerTest(unittest.TestCase):
    def test_empty_primitives_allowed_unknown_rejected(self):
        # empty set = sandbox-only sealing (PTE standalone); must construct without raising.
        sealer = make_sealer(_stub_generator(), None, None, set(), None)
        self.assertIsInstance(sealer, Sealer)
        with self.assertRaises(ValueError):
            make_sealer(_stub_generator(), None, None, {"bogus"}, None)

    def test_sandbox_only_is_a_sealed_test_case(self):
        # it plugs into the same SealedTestCase machinery the executor drives.
        self.assertTrue(issubclass(SandboxSealedTestCase, SealedTestCase))


if __name__ == "__main__":
    unittest.main(verbosity=2)
