"""Sign-extension speculative-exception sealing (AArch64), self-contained (no /dev/executor, no CE). The
sign-ext flip is only used composed with PAC (PacSignExtSealedTestCase); these cover its building
blocks:
  * _signext_mask_pool is bit 55 alone — the TTBR0/TTBR1 regime selector, the one bit in [63:55] whose
    flip faults under TBI on OR off (top-byte bits [63:56] are TBI-ignored, so they are excluded);
  * SignExtSealing (a CanonSealing) flips the base before the access and, when the base is preserved,
    flips it back after (EOR self-inverse) so the decoy re-converges with the baseline.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aarch64.aarch64_insn_encode import eor_imm_word, NOP_WORD
from src.aarch64.seal import sealer as S
from tests.conf_isolation import setUpModule, tearDownModule  # noqa: F401  (restores CONF + cwd)

_BIT55 = 1 << 55          # regime selector
_TOP_BYTE = 0xFF << 56    # [63:56], TBI-ignored


class SignExtPoolTest(unittest.TestCase):
    def test_pool_is_bit_55_only(self):
        self.assertEqual(S._signext_mask_pool(), [_BIT55])

    def test_pool_excludes_the_tbi_ignored_top_byte(self):
        # the only pool mask is bit 55; it never touches [63:56], so the flip is TBI-robust
        for m in S._signext_mask_pool():
            self.assertFalse(m & _TOP_BYTE, f"{m:#x} is in the TBI-ignored top byte")

    def test_bit_55_flip_disagrees_with_sign_extension(self):
        # a canonical TTBR1 pointer has bit55=1 and sign-extension bits [54:VA] all 1; flipping bit 55
        # makes the regime selector (now 0) disagree with [54:VA] (still 1) -> non-canonical in either
        # regime and whether or not the top byte is ignored (TBI on/off).
        va = 48
        canonical = _BIT55 | (((1 << (55 - va)) - 1) << va)   # bit55=1, [54:va]=1
        flipped = canonical ^ _BIT55
        self.assertEqual((flipped >> 55) & 1, 0, "regime selector flipped to 0")
        self.assertNotEqual((flipped >> va) & 1, (flipped >> 55) & 1, "[54:va] no longer sign-extends bit 55")


@unittest.skip("obsolete: sealer uses GlobalValueTable now; see unit_canon_gst.py")
class SignExtSealingRenderTest(unittest.TestCase):
    def setUp(self):
        self.pool = [1 << 55]

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
        self.assertEqual([i.name for i in s.seal(1 << 55, None)], ["eor"], "decoy = flip only")

    def test_placeholder_encodes_to_nop(self):
        s = S.SignExtSealing("x5", None, self.pool, revert=True)
        self.assertTrue(all(S._encode(i) == NOP_WORD for i in s.slot_insts))


if __name__ == "__main__":
    unittest.main()
