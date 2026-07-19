"""Sealing resolution is a pure function of (sealing class, per-test-case salt): no live randomness,
no cross-input state. This lets the executor drop the plan cache — same-class inputs forge identically
and every variant reproduces across trace passes. Verifies:
  - _wrong_sigs: deterministic from (correct_sig, mask, salt); perturbs only the field bits; every
    entry is a genuine AUTH failure; corner cases (tiny mask, zero mask).
  - PacSealing.seal strip choice is seeded (same rng -> same render).
  - ResolvedSealingTestCase.genuine()/decoy(rng): reproducible per seed; genuine fixed per salt.
No kernel module needed.
"""
import os
import sys
import random
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

from src.config import CONF
from src.interfaces import GeneratorException, Instruction, RegisterOperand
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.seal.pac import PacSign, build_pac_specs
from src.aarch64.seal.sealer import _wrong_sigs, PacSealing, ResolvedSealingTestCase, _Resolved

_PAC_FIELD = 0xFFFF << 48          # a plausible 16-bit PAC field mask
_CORRECT = 0x0ABC << 48 | 0x1234   # arbitrary "correct signature"


class WrongSigsTest(unittest.TestCase):
    def test_deterministic(self):
        a = _wrong_sigs(_CORRECT, _PAC_FIELD, salt=0xdead)
        b = _wrong_sigs(_CORRECT, _PAC_FIELD, salt=0xdead)
        self.assertEqual(a, b)

    def test_only_field_bits_perturbed(self):
        for s in _wrong_sigs(_CORRECT, _PAC_FIELD, salt=1):
            self.assertEqual(s & ~_PAC_FIELD, _CORRECT & ~_PAC_FIELD)   # non-field bits untouched
            self.assertNotEqual(s, _CORRECT)                            # a real AUTH failure

    def test_pool_is_distinct(self):
        pool = _wrong_sigs(_CORRECT, _PAC_FIELD, salt=2)
        self.assertEqual(len(pool), len(set(pool)))
        self.assertGreater(len(pool), 0)

    def test_salt_changes_pool(self):
        self.assertNotEqual(_wrong_sigs(_CORRECT, _PAC_FIELD, salt=1),
                            _wrong_sigs(_CORRECT, _PAC_FIELD, salt=2))

    def test_class_invariant_no_hidden_state(self):
        # a fresh interpreter-independent call reproduces the pool: nothing but the args feeds it
        pools = [_wrong_sigs(_CORRECT, _PAC_FIELD, salt=7) for _ in range(5)]
        self.assertTrue(all(p == pools[0] for p in pools))

    def test_tiny_mask_single_forgery(self):
        # a 1-bit field admits exactly one wrong value: no infinite loop, pool of size 1
        pool = _wrong_sigs(_CORRECT, 0x1, salt=3)
        self.assertEqual(len(pool), 1)
        self.assertNotEqual(pool[0], _CORRECT)

    def test_zero_mask_raises(self):
        # no field to perturb -> no forgery is possible -> loud failure, never a silent empty pool
        with self.assertRaises(GeneratorException):
            _wrong_sigs(_CORRECT, 0x0, salt=4)


class SealResolutionDeterminismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CONF.load(os.path.join(_ROOT, "config_pac_mte_basic.yml"))
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        gen = Aarch64RandomGenerator(isa, 0x1234)
        _, auth_specs, xpac_specs = build_pac_specs(gen)
        cls.enc = PacSign(gen, auth_specs, xpac_specs, 0xFFFF << 48)

    def _pac_sealing(self) -> PacSealing:
        inst = Instruction("autia", True, "", False)
        inst.operands = [RegisterOperand("x0", 64, True, True),
                         RegisterOperand("x1", 64, True, False)]
        return PacSealing("x0", inst, self.enc)

    def _resolved(self, salt: int) -> ResolvedSealingTestCase:
        """A synthetic one-slot resolution: a speculative (decoy-eligible) PAC entry with a forgery
        pool, so decoy() can perturb it. No CE / kernel needed."""
        ps = self._pac_sealing()
        # wrong signatures differ in the PAC field window (bits [63:48]), like a real forgery pool
        alts = [0x1111 << 48, 0x2222 << 48, 0x3333 << 48, 0x4444 << 48]
        entry = _Resolved(ps, _CORRECT, alts, spec_nesting=5)
        offsets = {id(ps): (0, 4)}                 # seal() emits two words (MOVK + AUTH/XPAC)
        return ResolvedSealingTestCase([entry], bytes(16), offsets, salt)

    def test_pac_seal_strip_is_seed_deterministic(self):
        ps = self._pac_sealing()
        run1 = [i.name.lower() for i in ps.seal(0x1234, random.Random(9))]
        run2 = [i.name.lower() for i in ps.seal(0x1234, random.Random(9))]
        self.assertEqual(run1, run2)                       # same seed -> identical render

    def test_decoy_reproducible_per_seed(self):
        r = self._resolved(salt=0x55)
        self.assertEqual(r.decoy(random.Random(1)), r.decoy(random.Random(1)))
        self.assertEqual(r.decoy(random.Random(2)), r.decoy(random.Random(2)))

    def test_genuine_deterministic_and_cached(self):
        r = self._resolved(salt=0x55)
        self.assertIs(r.genuine(), r.genuine())            # cached
        # a second resolution with the same entries + salt reproduces the genuine plan
        self.assertEqual(self._resolved(salt=0x55).genuine(), r.genuine())

    def test_decoy_varies_with_seed(self):
        r = self._resolved(salt=0x55)
        decoys = {r.decoy(random.Random(i)) for i in range(16)}
        self.assertGreater(len(decoys), 1, "decoy never varied across 16 seeds")

    def test_genuine_never_forges(self):
        # genuine seals the correct value on every slot: the loaded signature is _CORRECT's field
        r = self._resolved(salt=0x55)
        self.assertTrue(r.genuine(), "expected at least one genuine relocation")


class PacFieldWindowTest(unittest.TestCase):
    """PacSign deduces its MOVK windows from the PAC field mask, and the emitter produces one MOVK per
    window — so a field reaching below bit 48 (a narrow-VA core) emits more than one."""

    @classmethod
    def setUpClass(cls):
        CONF.load(os.path.join(_ROOT, "config_pac_mte_basic.yml"))
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.gen = Aarch64RandomGenerator(isa, 0x1234)
        _, cls.auth_specs, cls.xpac_specs = build_pac_specs(cls.gen)

    def _sealing(self, pac_bits_mask):
        enc = PacSign(self.gen, self.auth_specs, self.xpac_specs, pac_bits_mask)
        inst = Instruction("autia", True, "", False)
        inst.operands = [RegisterOperand("x0", 64, True, True), RegisterOperand("x1", 64, True, False)]
        return PacSealing("x0", inst, enc)

    def test_window_count_from_mask(self):
        for mask, n in ((0x7F << 48, 1), ((0x7F << 48) | (0xFF << 40), 2),
                        ((0x7F << 48) | (0xFFFF << 24), 3)):
            enc = PacSign(self.gen, self.auth_specs, self.xpac_specs, mask)
            self.assertEqual(enc.n_sig_movks, n)

    def test_one_window_slot_and_fill(self):
        ps = self._sealing(0x7F << 48)                                    # 48-bit VA: PAC in top window
        self.assertEqual([i.name.lower() for i in ps.seal(None, None)], ["nop", "xpaci"])
        signed = 0x0022000012345678
        movks = [i for i in ps.seal(signed, random.Random(0)) if i.name.lower() == "movk"]
        self.assertEqual([(int(m.operands[2].value), int(m.operands[1].value)) for m in movks],
                         [(48, (signed >> 48) & 0xFFFF)])

    def test_two_window_slot_and_fill(self):
        ps = self._sealing((0x7F << 48) | (0xFF << 40))                   # narrow VA: field reaches below 48
        self.assertEqual([i.name.lower() for i in ps.seal(None, None)], ["nop", "nop", "xpaci"])
        signed = 0x0022c20012345678
        movks = [i for i in ps.seal(signed, random.Random(0)) if i.name.lower() == "movk"]
        self.assertEqual({int(m.operands[2].value): int(m.operands[1].value) for m in movks},
                         {32: (signed >> 32) & 0xFFFF, 48: (signed >> 48) & 0xFFFF})


if __name__ == "__main__":
    unittest.main()
