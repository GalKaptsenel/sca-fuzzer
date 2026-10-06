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
# profile(version, generic version, auth level, va0, va1, tbi0, tbi1, tbid0, tbid1, constpacfield, mtx0, mtx1)
QARMA5 = q.profile(5, 5, 3, 39, 39, True, False, False, False, False, False, False)
QARMA3 = q.profile(3, 3, 3, 48, 48, True, True, False, False, False, False, False)
N3 = q.profile(3, 3, 3, 48, 48, True, True, False, True, False, False, False)


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
        bad = q.profile(5, 5, 3, 39, 39, True, True, False, False, False, False, False)  # TBI1 on
        self.assertNotEqual(q.sign(0xffffffc012345000, CTX, "pacia", KEYS, bad), 0xb1e67d4012345000)

    def test_noncanonical_selbit_matches_hardware(self):
        # PACIZB measured on N3 (QARMA3, VA 48, TBI0=TBI1=1, TBID0=0, TBID1=1, PAuth2): a non-canonical
        # pointer (bit63 != bit55). ARM AddPAC selbit = ptr<55> because the LOW half has effective TBI,
        # even though this high-half pointer has none; selbit = ptr<63> gave 0xfe21... (FPAC regression).
        apib = (0x4de594c526f0f1cc, 0xd94d88c0873cc2b6)
        self.assertEqual(q.addpac(0x0ae500000ae50000, 0, *apib, N3, True), 0xe4bf00000ae50000)

    def test_sign_then_strip_roundtrips(self):
        user = 0x0000000000abc000
        for ctx in range(4):
            for p in (QARMA5, QARMA3):
                self.assertEqual(q.strip(q.addpac(user, ctx, KEYS[0], KEYS[1], p, False), p, False), user)

    def test_context_and_key_sensitivity(self):
        user = 0x0000000000abc000
        a = q.addpac(user, 0x11, KEYS[0], KEYS[1], QARMA5, False)
        self.assertNotEqual(a, q.addpac(user, 0x12, KEYS[0], KEYS[1], QARMA5, False))     # wrong ctx
        self.assertNotEqual(a, q.addpac(user, 0x11, KEYS[0] ^ 1, KEYS[1], QARMA5, False)) # wrong key
        self.assertEqual(a, q.addpac(user, 0x11, KEYS[0], KEYS[1], QARMA5, False))        # deterministic

    def test_qarma3_differs_from_qarma5(self):
        user = 0x0000000000abc000
        self.assertNotEqual(q.addpac(user, 0x11, KEYS[0], KEYS[1], QARMA3, False),
                            q.addpac(user, 0x11, KEYS[0], KEYS[1], QARMA5, False))

    def test_key_selection_by_mnemonic(self):
        keys = list(range(1, 11))   # apia={1,2} apib={3,4} apda={5,6} apdb={7,8} apga={9,10}
        ptr, ctx = 0x0000000000abc000, 0x99
        self.assertEqual(q.sign(ptr, ctx, "pacib", keys, QARMA5),
                         q.addpac(ptr, ctx, keys[2], keys[3], QARMA5, True))
        self.assertEqual(q.sign(ptr, ctx, "pacdb", keys, QARMA5),
                         q.addpac(ptr, ctx, keys[6], keys[7], QARMA5, False))


if __name__ == "__main__":
    unittest.main()
