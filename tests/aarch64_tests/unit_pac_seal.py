"""PacSign seal corner cases (logical, hardware-free — needs only the ISA, not /dev/executor):
  - genuine  -> [MOVK correct_sig, AUT*]  (auth), or [NOP, XPAC] (strip) with _REMAIN_STRIP_PROB,
                or [NOP, XPAC] when no signature was resolved
  - decoy    -> [MOVK alt_sig, AUT*]      (forged auth), or [NOP, XPAC] (strip)
The rng is controlled so each branch is exercised deterministically.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_pac import PacSign, PACFixPoint, build_pac_specs, _AUTH_TO_XPAC

_PAC_CATS = ["PAC", "PAC-SIGN", "PAC-AUTH", "PAC-STRIP",
             "BASE-ARITH", "BASE-MEM-LOAD", "BASE-MEM-STORE", "BASE-BRANCH"]


class _Rng:
    """Deterministic random() so the strip-vs-auth branch is forced; choice picks index 0."""
    def __init__(self, r): self._r = r
    def random(self): return self._r
    def choice(self, seq): return seq[0]


class TestPacSignSeal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CONF.load(os.path.join(_ROOT, "config.yml"))
        cls.isa = InstructionSet(os.path.join(_ROOT, "base.json"), _PAC_CATS)
        cls.gen = Aarch64RandomGenerator(cls.isa, 0xC0FFEE)
        _, auth_specs, xpac_specs = build_pac_specs(cls.gen)
        cls.seal = PacSign(cls.gen, auth_specs, xpac_specs)

    def _fp(self, correct_sig=0x1234, alt_sigs=(0x5678,)):
        # a fresh fix point per test (no shared mutable state); committed_inst is a real AUT* over x5
        fp = PACFixPoint(slot_id=0, value_reg="x5")
        fp.committed_inst = self.seal._pick_mem_auth("x5")
        fp.correct_sig = correct_sig
        fp.alt_sigs = list(alt_sigs)
        return fp

    _ABOVE = 0.5   # >= _REMAIN_STRIP_PROB -> auth ;  _BELOW < it -> strip
    _BELOW = 0.0

    def _xpac_of(self, fp):
        return _AUTH_TO_XPAC[fp.committed_inst.name.lower()]

    def test_placeholder_is_nop_xpac(self):
        fp = self._fp()
        out = [i.name.lower() for i in self.seal.placeholder(fp)]
        self.assertEqual(out, ["nop", self._xpac_of(fp)])

    def test_genuine_auth_uses_correct_sig(self):
        fp = self._fp(correct_sig=0x1234)
        out = self.seal.genuine(fp, _Rng(self._ABOVE))
        self.assertEqual(out[0].name.lower(), "movk")
        self.assertIn("0x1234", out[0].template)
        self.assertEqual(out[1].name.lower(), fp.committed_inst.name.lower())  # the AUT*

    def test_genuine_strip_branch(self):
        fp = self._fp()
        out = [i.name.lower() for i in self.seal.genuine(fp, _Rng(self._BELOW))]
        self.assertEqual(out, ["nop", self._xpac_of(fp)])

    def test_genuine_no_signature_is_strip(self):
        fp = self._fp(correct_sig=None)
        out = [i.name.lower() for i in self.seal.genuine(fp, _Rng(self._ABOVE))]
        self.assertEqual(out, ["nop", self._xpac_of(fp)])              # None sig -> strip, never auth

    def test_decoy_forges_alt_sig(self):
        fp = self._fp(correct_sig=0x1234, alt_sigs=(0x5678,))
        out = self.seal.decoy(fp, _Rng(self._ABOVE))
        self.assertEqual(out[0].name.lower(), "movk")
        self.assertIn("0x5678", out[0].template)                      # the wrong (alt) signature
        self.assertNotIn("0x1234", out[0].template)
        self.assertEqual(out[1].name.lower(), fp.committed_inst.name.lower())

    def test_decoy_strip_branch(self):
        fp = self._fp()
        out = [i.name.lower() for i in self.seal.decoy(fp, _Rng(self._BELOW))]
        self.assertEqual(out, ["nop", self._xpac_of(fp)])

    def test_memory_pointer_self_provides_auth(self):
        # a memory-pointer fix point has no committed AUT* yet; placeholder picks one (so genuine/
        # decoy can sign+auth the base), staying Sandbox-agnostic.
        fp = PACFixPoint(slot_id=0, value_reg="x5")
        self.assertIsNone(fp.committed_inst)
        out = [i.name.lower() for i in self.seal.placeholder(fp)]
        self.assertIsNotNone(fp.committed_inst)                       # self-provided
        self.assertEqual(out, ["nop", _AUTH_TO_XPAC[fp.committed_inst.name.lower()]])


if __name__ == "__main__":
    unittest.main()
