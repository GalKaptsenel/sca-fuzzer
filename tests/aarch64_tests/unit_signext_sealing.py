"""Sign-extension non-interference sealing (AArch64), self-contained (no /dev/executor, no CE). The
sign-ext flip is only used composed with PAC (PacSignExtSealedTestCase); these cover its building
blocks:
  * _signext_mask_pool draws the bits [63:55] (each a single bit), so EORing one makes that bit disagree
    with the other eight -> non-canonical in either regime and under TBI on/off -> a translation fault;
    a config mask must be one such bit;
  * SignExtSealing (a CanonSealing) flips the base before the access and, when the base is preserved,
    flips it back after (EOR self-inverse) so the decoy re-converges with the baseline.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import CONF
from src.aarch64.aarch64_relocations import eor_imm_word, NOP_WORD
from src.aarch64.seal import sealer as S
from tests.conf_isolation import setUpModule, tearDownModule  # noqa: F401  (restores CONF + cwd)

_SIGNEXT = ((1 << 9) - 1) << 55   # bits [63:55]


class SignExtPoolTest(unittest.TestCase):
    def setUp(self):
        CONF.instruction_set = "aarch64"
        CONF.set_to_arch_defaults()
        CONF.signext_mask = None

    def tearDown(self):
        CONF.set_to_arch_defaults()

    def test_pool_is_single_bits_in_63_55(self):
        pool = S._signext_mask_pool()
        self.assertEqual(pool, [1 << b for b in range(55, 64)])
        for m in pool:
            self.assertEqual(m & (m - 1), 0, f"{m:#x} not a single bit")
            self.assertFalse(m & ~_SIGNEXT, f"{m:#x} outside [63:55]")

    def test_each_bit_breaks_uniformity(self):
        # a canonical (kernel) pointer has bits [63:55] all 1; flipping one makes it disagree with the rest
        canonical = _SIGNEXT
        for m in S._signext_mask_pool():
            flipped = (canonical ^ m) & _SIGNEXT
            bits = {(flipped >> b) & 1 for b in range(55, 64)}
            self.assertEqual(bits, {0, 1}, f"flip {m:#x} must break sign-extension uniformity")

    def test_config_mask_single_bit_passes(self):
        CONF.signext_mask = 1 << 55
        self.assertEqual(S._signext_mask_pool(), [1 << 55])

    def test_config_mask_multi_bit_rejected(self):
        CONF.signext_mask = (1 << 55) | (1 << 60)
        with self.assertRaisesRegex(Exception, "single bit"):
            S._signext_mask_pool()

    def test_config_mask_outside_range_rejected(self):
        CONF.signext_mask = 1 << 54   # below bit 55 -> canonicality's range, not sign-ext
        with self.assertRaisesRegex(Exception, "single bit"):
            S._signext_mask_pool()


class SignExtSealingRenderTest(unittest.TestCase):
    def setUp(self):
        self.pool = [1 << 55, 1 << 63]

    def test_base_preserved_flips_then_reverts(self):
        s = S.SignExtSealing("x3", None, self.pool, revert=True)
        self.assertEqual([i.name for i in s.seal(None, None)], ["nop", "nop"], "genuine = 2 NOPs")
        decoy = s.seal(1 << 55, None)
        self.assertEqual([i.name for i in decoy], ["eor", "eor"], "decoy = flip + revert")
        w0, w1 = S._encode(decoy[0]), S._encode(decoy[1])
        self.assertEqual(w0, eor_imm_word(3, 3, 1 << 55))
        self.assertEqual(w0, w1, "same mask before and after -> EOR self-inverse -> base restored")

    def test_base_not_preserved_flips_only(self):
        s = S.SignExtSealing("x3", None, self.pool, revert=False)
        self.assertEqual([i.name for i in s.seal(None, None)], ["nop"], "genuine = 1 NOP")
        self.assertEqual([i.name for i in s.seal(1 << 63, None)], ["eor"], "decoy = flip only")

    def test_placeholder_encodes_to_nop(self):
        s = S.SignExtSealing("x5", None, self.pool, revert=True)
        self.assertTrue(all(S._encode(i) == NOP_WORD for i in s.slot_insts))


if __name__ == "__main__":
    unittest.main()
