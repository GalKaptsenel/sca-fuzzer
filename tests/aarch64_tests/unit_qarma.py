"""Checks for the software QARMA3/QARMA5 pointer-auth (src/aarch64/aarch64_qarma.py).

The known-answer vectors are real hardware output (pacia under a fixed key), so a match proves the
Python model is bit-exact with QARMA5 hardware and agrees with the CE's C port (which checks the same
vectors in test_qarma.c)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aarch64 import aarch64_qarma as q

KEYS = [0x0123456789abcdef, 0xfedcba9876543210] + [0] * 8   # apia = {lo, hi}
CTX = 0x1122334455667788
QARMA5 = q.PacProfile(iterations=4, tsz=25, tbi0=1, tbi1=0, pauth2=True)   # VA 39
QARMA3 = q.PacProfile(iterations=2, tsz=16, tbi0=1, tbi1=1, pauth2=True)   # VA 48


class QarmaTest(unittest.TestCase):
    def test_qarma5_matches_hardware(self):
        # pacia outputs measured on real hardware. HW selects TBI per pointer by bit 55: a low-half
        # (user) pointer uses TBI on (tbi0=1), a high-half (kernel) pointer uses TBI off (tbi1=0). The
        # one profile reproduces both -- the executor signs kernel pointers, and TBI1 wrong there
        # yields a wrong signature that FPAC-faults (regression).
        for ptr, want in ((0x0000000012345000, 0x002a920012345000),   # user,   tbi0=1
                          (0xffffffc012345000, 0xb1e67d4012345000),   # kernel, tbi1=0
                          (0xffffff8000abc000, 0xd5a28d0000abc000)):  # kernel, tbi1=0
            self.assertEqual(q.sign(ptr, CTX, "pacia", KEYS, QARMA5), want)
        bad = q.PacProfile(iterations=4, tsz=25, tbi0=1, tbi1=1, pauth2=True)   # TBI1 on -> the FPAC bug
        self.assertNotEqual(q.sign(0xffffffc012345000, CTX, "pacia", KEYS, bad), 0xb1e67d4012345000)

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
