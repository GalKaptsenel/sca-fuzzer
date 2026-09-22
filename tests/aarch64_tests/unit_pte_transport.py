"""Transport tests for the PTE environment section over REIF (no hardware, no capstone):
the per-variant EnvironmentPlan survives serialize/deserialize, the genuine variant ships NO PTE
section (kernel keeps pristine PTEs) while a decoy does, and the difference between genuine and decoy
is solely that section -- the code relocations are identical.

    python -m unittest tests.aarch64_tests.unit_pte_transport
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.interfaces import Input                                                    # noqa: E402
from src.aarch64.aarch64_executor_input_encoder import (                           # noqa: E402
    ExecutorInput, deserialize, SEC_PTE_SETTINGS)
from src.aarch64.seal.environment import EnvironmentPlan, PteOverride, LEVEL_LEAF   # noqa: E402


def _sections_present(blob: bytes):
    import struct
    _, _, _, n, _, _ = struct.unpack_from("<6Q", blob, 0)
    return {struct.unpack_from("<4Q", blob, 48 + i * 32)[0] for i in range(n)}


class PteTransportTest(unittest.TestCase):
    def setUp(self):
        self.inp = Input(1)                 # zero-filled single-actor input is enough for transport
        self.decoy_plan = EnvironmentPlan(pte_overrides=(
            PteOverride(2, LEVEL_LEAF, mask=0b1, value=0b0),        # clear valid on the spec-only page
            PteOverride(2, LEVEL_LEAF, mask=0xC0, value=0x40)))     # ap = 1

    def test_genuine_ships_no_pte_section(self):
        genuine = ExecutorInput(self.inp)                          # empty env plan (pristine)
        self.assertTrue(genuine.env_plan.is_empty)
        self.assertNotIn(SEC_PTE_SETTINGS, _sections_present(genuine.serialize()))

    def test_decoy_ships_pte_section_and_roundtrips(self):
        decoy = ExecutorInput(self.inp, env_plan=self.decoy_plan)
        blob = decoy.serialize()
        self.assertIn(SEC_PTE_SETTINGS, _sections_present(blob))
        self.assertEqual(deserialize(blob).env_plan, self.decoy_plan)

    def test_genuine_and_decoy_differ_only_in_environment(self):
        genuine = ExecutorInput(self.inp)
        decoy = ExecutorInput(self.inp, env_plan=self.decoy_plan)
        # identical code relocations (the whole point: only the environment differs)
        self.assertEqual(genuine.code_reloc, decoy.code_reloc)
        # and the section sets differ by exactly the PTE section
        self.assertEqual(_sections_present(decoy.serialize()) - _sections_present(genuine.serialize()),
                         {SEC_PTE_SETTINGS})


if __name__ == "__main__":
    unittest.main(verbosity=2)
