"""Per-site seal slot ordering: sandbox -> auth -> offset-cancel SUB -> MTE retag -> access.

The MTE retag (ADDG) MUST be the last instrumentation immediately before its access. If an
address-mutating op (the offset-cancel SUB) is left between the retag and the access, the placeholder
CE trace (where the retag is a NOP, the tag restored only by the after-access correction) diverges
from the genuine run, the PAC resolver signs over the wrong register state, and the genuine AUT*
FPAC-faults. This guards that regression. seal() only inserts instructions, so no kernel module is
needed.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.seal.primitives import index_instructions
from src.aarch64.seal.sealer import make_sealer


class SealOrderingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        CONF.load(os.path.join(_ROOT, "config_pac_mte_basic.yml"))
        cls.isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)

    def _seal_with_mte_sites(self, primitives):
        """Generate (over a few seeds) a sealed TC that has at least one MTE data-access site."""
        for seed in range(12):
            gen = Aarch64RandomGenerator(self.isa, seed)
            sealer = make_sealer(gen, lambda tc, inp: None, primitives, None)
            path = os.path.join(tempfile.mkdtemp(), "t.asm")
            sealed = sealer.seal(gen.create_test_case(path, disable_assembler=True))
            if sealed._mte:
                return sealed
        self.fail(f"no MTE data-access site generated in 12 seeds for {primitives}")

    def _assert_mte_immediately_before_access(self, sealed):
        locs = index_instructions(sealed._tc)
        for m in sealed._mte:
            self.assertEqual(len(m.slot_locs), 1)                 # the MTE slot is one instruction
            fm, bm, im = m.slot_locs[0]
            fa, ba, ia = locs[id(m.access)]
            self.assertEqual((fm, bm), (fa, ba), "MTE slot and its access must share a basic block")
            self.assertEqual(im, ia - 1,
                             "MTE retag must be the instruction immediately before its access "
                             "(nothing — especially the offset-cancel SUB — may sit between them)")

    def test_mte_only_retag_is_last(self):
        self._assert_mte_immediately_before_access(self._seal_with_mte_sites({"mte"}))

    def test_pac_mte_retag_is_last(self):
        self._assert_mte_immediately_before_access(self._seal_with_mte_sites({"pac", "mte"}))

    def test_pac_mte_auth_precedes_retag_at_a_site(self):
        """At a PAC+MTE site the auth (PAC slot) is placed before the retag — auth then (sub) then
        retag. Checks the PAC slot for the same base register sits strictly before the MTE slot."""
        sealed = self._seal_with_mte_sites({"pac", "mte"})
        checked = 0
        for m in sealed._mte:
            fm, bm, im = m.slot_locs[0]
            # the memory-base PAC sealing for this site shares the value_reg and basic block
            pac_before = [p.slot_locs[-1][2] for p in sealed._pac
                          if p.value_reg == m.value_reg and p.slot_locs
                          and p.slot_locs[0][:2] == (fm, bm) and p.slot_locs[-1][2] < im]
            if pac_before:
                self.assertLess(max(pac_before), im)             # auth slot ends before the retag
                checked += 1
        self.assertGreater(checked, 0, "expected at least one PAC-auth-then-MTE site")


if __name__ == "__main__":
    unittest.main()
