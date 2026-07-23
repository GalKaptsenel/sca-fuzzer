"""Checks for the software QARMA3/QARMA5 pointer-auth (src/aarch64/aarch64_qarma.py).

The known-answer vector is real Pixel-8 hardware output (pacia under a fixed key), so a match proves the
Python model is bit-exact with QARMA5 hardware and agrees with the CE's C port (which checks the same
vector in test_qarma.c)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aarch64 import aarch64_qarma as q

KEYS = [0x0123456789abcdef, 0xfedcba9876543210] + [0] * 8   # apia = {lo, hi}
QARMA5 = q.PacProfile(iterations=4, tsz=25, tbi=1, pauth2=True)   # VA 39
QARMA3 = q.PacProfile(iterations=2, tsz=16, tbi=1, pauth2=True)   # VA 48


class QarmaTest(unittest.TestCase):
    def test_qarma5_matches_pixel_hardware(self):
        signed = q.sign(0x0000000012345000, 0x1122334455667788, "pacia", KEYS, QARMA5)
        self.assertEqual(signed, 0x002a920012345000)

    def test_sign_then_strip_roundtrips(self):
        user = 0x0000000000abc000
        for ctx in range(4):
            for p in (QARMA5, QARMA3):
                self.assertEqual(q.strip(q.addpac(user, ctx, KEYS[0], KEYS[1], p), p), user)

    def test_context_and_key_sensitivity(self):
        user = 0x0000000000abc000
        a = q.addpac(user, 0x11, KEYS[0], KEYS[1], QARMA5)
        self.assertNotEqual(a, q.addpac(user, 0x12, KEYS[0], KEYS[1], QARMA5))     # wrong ctx
        self.assertNotEqual(a, q.addpac(user, 0x11, KEYS[0] ^ 1, KEYS[1], QARMA5)) # wrong key
        self.assertEqual(a, q.addpac(user, 0x11, KEYS[0], KEYS[1], QARMA5))        # deterministic

    def test_qarma3_differs_from_qarma5(self):
        user = 0x0000000000abc000
        self.assertNotEqual(q.addpac(user, 0x11, KEYS[0], KEYS[1], QARMA3),
                            q.addpac(user, 0x11, KEYS[0], KEYS[1], QARMA5))

    def test_key_selection_by_mnemonic(self):
        keys = list(range(1, 11))   # apia={1,2} apib={3,4} apda={5,6} apdb={7,8} apga={9,10}
        ptr, ctx = 0x0000000000abc000, 0x99
        self.assertEqual(q.sign(ptr, ctx, "pacib", keys, QARMA5),
                         q.addpac(ptr, ctx, keys[2], keys[3], QARMA5))
        self.assertEqual(q.sign(ptr, ctx, "pacdb", keys, QARMA5),
                         q.addpac(ptr, ctx, keys[6], keys[7], QARMA5))


if __name__ == "__main__":
    unittest.main()
