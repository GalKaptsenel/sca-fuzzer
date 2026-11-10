"""The sandbox-only builder path that makes PTE fuzzing a STANDALONE feature (no code primitive):
make_gvt_access_builder must accept the empty primitive set and dispatch it to SandboxGvtTestCase, and still
reject an unknown primitive. No hardware needed (construction only).

    python -m unittest tests.aarch64_tests.unit_pte_builder
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.aarch64.gvt.builder import (make_gvt_access_builder, GvtAccessBuilder, GvtTestCase,           # noqa: E402
                                     SandboxGvtTestCase)


def _stub_generator():
    """The minimal surface SandboxWalk touches at construction (no capstone, no ISA load)."""
    return types.SimpleNamespace(
        instruction_set=types.SimpleNamespace(instructions=[]),
        target_desc=types.SimpleNamespace(reg_normalized={}))


class SandboxOnlyBuilderTest(unittest.TestCase):
    def test_empty_primitives_allowed_unknown_rejected(self):
        # empty set = sandbox-only GVT access (PTE standalone); must construct without raising.
        builder = make_gvt_access_builder(_stub_generator(), None, set(), None, None)
        self.assertIsInstance(builder, GvtAccessBuilder)
        with self.assertRaises(ValueError):
            make_gvt_access_builder(_stub_generator(), None, {"bogus"}, None, None)

    def test_sandbox_only_is_a_gvt_test_case(self):
        # it plugs into the same GvtTestCase machinery the executor drives.
        self.assertTrue(issubclass(SandboxGvtTestCase, GvtTestCase))


if __name__ == "__main__":
    unittest.main(verbosity=2)
