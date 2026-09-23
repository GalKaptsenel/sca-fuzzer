"""Generation admission for the atomic / acquire-release families: enabling the BASE-MEM-ATOMIC and
BASE-MEM-ACQREL categories admits the integer LSE atomics + swp + ordered accesses, while the
CE-unsafe (CAS/CASP), FEAT-gated (LSE128/LSUI), floating-point, RCW, and exclusive forms stay out.

Runnable from any cwd; loads the real base.json once. Run from the repo root:
    python -m unittest tests.aarch64_tests.unit_atomic_enablement
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from src.config import CONF                                  # noqa: E402
from src.isa_loader import InstructionSet                    # noqa: E402

_NAMES = None
_SAVED = None


def setUpModule():
    global _NAMES, _SAVED
    _SAVED = copy.deepcopy(CONF._borg_shared_state)
    CONF.load(os.path.join(_ROOT, "config.yml"))
    CONF.instruction_set = "aarch64"
    CONF.instruction_categories = ["BASE-MEM-ATOMIC", "BASE-MEM-ACQREL"]
    isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
    _NAMES = {s.name for s in isa.instructions}


def tearDownModule():
    CONF._borg_shared_state.clear()
    CONF._borg_shared_state.update(_SAVED)


class AtomicEnablementTest(unittest.TestCase):
    def _assert_in(self, *names):
        for n in names:
            self.assertIn(n, _NAMES, f"{n} should be generatable")

    def _assert_out(self, *names):
        for n in names:
            self.assertNotIn(n, _NAMES, f"{n} should NOT be generatable")

    def test_integer_lse_atomics_enabled(self):
        self._assert_in("ldadd", "ldclr", "ldeor", "ldset", "ldsmax", "ldsmin", "ldumax", "ldumin",
                        "swp", "swpa", "swpl", "swpal", "swpb", "swph")

    def test_acquire_release_enabled(self):
        self._assert_in("ldar", "stlr", "ldapr", "ldlar", "stllr")

    def test_cas_and_casp_enabled(self):                # CE models the RMW; generator patches CASP pairs
        self._assert_in("cas", "casa", "casal", "casb", "cash",
                        "casp", "caspa", "caspal", "caspl")

    def test_feat_gated_variants_excluded_by_arch(self):
        self._assert_out("ldsetp", "swpp",              # LSE128 pair (v9.4)
                         "ldtadd", "cast", "swpt")      # LSUI tag variants (v9.6)

    def test_other_families_not_pulled_in(self):
        self._assert_out("ldfadd", "stfadd",            # FP atomics (own tag, not enabled)
                         "rcwcas",                      # RCW (own tag, not enabled)
                         "ldxr", "stxr", "ldaxr", "stlxr")  # exclusives (not enabled)

    def test_non_memory_and_plain_loads_not_pulled_in(self):
        self._assert_out("ldr", "str", "ldp", "add", "adds")


if __name__ == "__main__":
    unittest.main(verbosity=2)
