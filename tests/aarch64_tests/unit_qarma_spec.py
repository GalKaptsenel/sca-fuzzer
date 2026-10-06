"""The software PAC model (aarch64_qarma.py) against the ARM ARM shared pseudocode (aarch64/functions/pac):
AddPAC selbit / field / non-canonical rules per feature level, Strip, and the CE's AUT* model (strip,
re-sign, compare) against a literal transcription of the PAuth2 Auth pseudocode."""
import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aarch64 import aarch64_qarma as q

M = (1 << 64) - 1
KLO, KHI = 0x0123456789abcdef, 0xfedcba9876543210


def _prof(va0=48, va1=48, tbi0=1, tbi1=1, tbid0=0, tbid1=0, level=3, cpf=False):
    return q.profile(3, 3, level, va0, va1, tbi0, tbi1, tbid0, tbid1, cpf, False, False)


def _bits(hi: int, lo: int) -> int:
    return ((1 << (hi - lo + 1)) - 1) << lo


def _spec_auth(ptr, mod, p, is_instr):
    """Literal ARM Auth (FEAT_PAuth2, no MTX): returns (result, would_pass)."""
    half = (ptr >> 55) & 1
    tbi_h, tbid_h = (p.tbi1, p.tbid1) if half else (p.tbi0, p.tbid0)
    tbi = bool(tbi_h) and not (is_instr and tbid_h)
    bottom = 64 - (p.t1sz if half else p.t0sz)
    ext = M if half else 0
    low = ptr & _bits(bottom - 1, 0)
    if tbi:
        original = (ptr & _bits(63, 56)) | (ext & _bits(55, bottom)) | low
    else:
        original = (ext & _bits(63, bottom)) | low
    pac = q.computepac(original, mod, KLO, KHI, p.iterations)
    result = ptr ^ (pac & _bits(54, bottom))
    if not tbi:
        result ^= pac & _bits(63, 56)
    rep = M if (result >> 55) & 1 else 0
    span = _bits(54, bottom) if tbi else _bits(63, bottom)
    return result, (result & span) == (rep & span)


class AddPacSelbitTest(unittest.TestCase):
    """selbit = ptr<55> if (data: TBI0|TBI1; instr: TBI0&!TBID0 | TBI1&!TBID1) else ptr<63>; always
    ptr<55> under PAuth2+CONSTPACFIELD. Observable as result bit 55 of a bit63 != bit55 pointer."""

    def test_selbit_table(self):
        ptrs = (0x0ae500000ae50000, 0xf000000012345000)   # bit55=1,bit63=0 / bit55=0,bit63=1
        for tbi0, tbi1, tbid0, tbid1, cpf in itertools.product((0, 1), repeat=5):
            p = _prof(tbi0=tbi0, tbi1=tbi1, tbid0=tbid0, tbid1=tbid1, cpf=bool(cpf))
            for is_instr, ptr in itertools.product((False, True), ptrs):
                if cpf:
                    src = 55
                elif is_instr:
                    src = 55 if (tbi1 and not tbid1) or (tbi0 and not tbid0) else 63
                else:
                    src = 55 if (tbi1 or tbi0) else 63
                got = (q.addpac(ptr, 0, KLO, KHI, p, is_instr) >> 55) & 1
                self.assertEqual(got, (ptr >> src) & 1, f"{p} instr={is_instr} ptr={ptr:#x}")


class AddPacFieldTest(unittest.TestCase):
    def test_field_width_per_half(self):
        p = _prof(va0=39, va1=48)
        self.assertEqual(q.pac_field_mask(0x0000000012345000, p, False), _bits(54, 39))
        self.assertEqual(q.pac_field_mask(0xffff000012345000, p, False), _bits(54, 48))
        p = _prof(va0=39, va1=48, tbi1=0)
        self.assertEqual(q.pac_field_mask(0xffff000012345000, p, False), _bits(54, 48) | _bits(63, 56))

    def test_addpac_writes_only_the_field(self):
        rng = random.Random(1)
        for va0, va1, tbi0, tbi1, tbid0, tbid1 in itertools.product((39, 48), (39, 48), *((0, 1),) * 4):
            p = _prof(va0, va1, tbi0, tbi1, tbid0, tbid1)
            for _ in range(8):
                for half in (0, 1):
                    ptr = rng.randrange(1 << 64)
                    ptr = q.strip(ptr | (half << 55) if half else ptr & ~(1 << 55), p, False)
                    out = q.addpac(ptr, rng.randrange(1 << 64), KLO, KHI, p, False)
                    self.assertEqual((out ^ ptr) & ~q.pac_field_mask(ptr, p, False) & M, 0)

    def test_noncanonical_rule_per_feature_level(self):
        ptr = 0x0012345612345000          # tbi half 0: [54:48] = 0x12 -> non-canonical
        legacy, epac = _prof(level=1), _prof(level=2)
        ext_ptr = ptr & ~_bits(55, 48)
        pac = q.computepac(ext_ptr, 0, KLO, KHI, legacy.iterations)
        self.assertEqual(q.addpac(ptr, 0, KLO, KHI, legacy, False) & _bits(54, 48),
                         (pac ^ (1 << 54)) & _bits(54, 48))            # PAuth: flip PAC<top_bit-1>
        self.assertEqual(q.addpac(ptr, 0, KLO, KHI, epac, False) & _bits(54, 48), 0)   # EPAC: PAC = 0
        self.assertEqual(q.addpac(ptr, 0, KLO, KHI, _prof(), False) & _bits(54, 48),
                         (pac ^ ptr) & _bits(54, 48))                   # PAuth2: XOR, no corruption


class UnusedBitsMaskTest(unittest.TestCase):
    """Non-PAuth2 corruption test per the spec's unusedbits_mask: [54:bottom], plus [63:56] when TBI is
    on; bit 55 never part of it."""

    def test_spec_unusedbits_mask(self):
        legacy = _prof(level=1)                       # TBI0=1 for this low-half pointer
        clean, tagged = 0x0000000012345000, 0xa500000012345000
        pac = q.computepac(clean, 0, KLO, KHI, legacy.iterations)
        self.assertEqual(q.addpac(clean, 0, KLO, KHI, legacy, False) & _bits(54, 48), pac & _bits(54, 48))
        self.assertEqual(q.addpac(tagged, 0, KLO, KHI, legacy, False) & _bits(54, 48),
                         (q.computepac(tagged, 0, KLO, KHI, legacy.iterations) ^ (1 << 54)) & _bits(54, 48))
        # No TBI: mask is [54:48] only (zero here) -> not corrupted; selbit = ptr<63> = 1 extends [63:48].
        no_tbi = _prof(level=1, tbi0=0, tbi1=0)
        pac = q.computepac(0xffff000012345000, 0, KLO, KHI, no_tbi.iterations)
        self.assertEqual(q.addpac(tagged, 0, KLO, KHI, no_tbi, False) & _bits(54, 48), pac & _bits(54, 48))


class StripTest(unittest.TestCase):
    def test_strip_fills_field_with_bit55(self):
        p = _prof(va0=39, va1=48, tbi1=0)
        low, high = 0xa512_34ff_ffff_1000, 0x5580_ab00_1234_5000      # bit55 = 0 / 1
        self.assertEqual(q.strip(low, p, False), 0xa500_007f_ffff_1000)    # TBI0: keep top byte, [55:39]=0
        self.assertEqual(q.strip(high, p, False), 0xffff_ab00_1234_5000)   # no TBI1: [63:48]=1


class AuthModelEquivalenceTest(unittest.TestCase):
    """The CE models an architectural AUT* as: pass iff addpac(strip(ptr)) == ptr, result strip(ptr).
    That must equal the PAuth2 Auth pseudocode for every pointer (signed, forged, random)."""

    def test_strip_resign_equals_spec_auth(self):
        rng = random.Random(7)
        checked = passes = 0
        for va0, va1, tbi0, tbi1, tbid0, tbid1, cpf in itertools.product(
                (39, 48), (39, 48), *((0, 1),) * 4, (False, True)):
            p = _prof(va0, va1, tbi0, tbi1, tbid0, tbid1, cpf=cpf)
            for is_instr in (False, True):
                for _ in range(12):
                    mod = rng.randrange(1 << 64)
                    base = rng.randrange(1 << 64)
                    signed = q.addpac(q.strip(base, p, is_instr), mod, KLO, KHI, p, is_instr)
                    for ptr in (signed, signed ^ (1 << 50), base):
                        spec_res, spec_ok = _spec_auth(ptr, mod, p, is_instr)
                        canonical = q.strip(ptr, p, is_instr)
                        model_ok = q.addpac(canonical, mod, KLO, KHI, p, is_instr) == ptr
                        self.assertEqual(model_ok, spec_ok, f"{p} instr={is_instr} ptr={ptr:#x}")
                        if spec_ok:
                            self.assertEqual(canonical, spec_res)
                            passes += 1
                        checked += 1
        self.assertGreater(passes, checked // 4)


class ProfileValidationTest(unittest.TestCase):
    def test_unmodeled_state_fails_loud(self):
        for args in ((3, 3, 3, 48, 48, 1, 1, 0, 1, False, True, False),    # MTX0
                     (3, 3, 3, 52, 48, 1, 1, 0, 1, False, False, False),   # T0SZ=12 (LVA)
                     (3, 3, 6, 48, 48, 1, 1, 0, 1, False, False, False),   # reserved auth level
                     (3, 4, 3, 48, 48, 1, 1, 0, 1, False, False, False),   # no QARMA4 generic
                     (4, 3, 3, 48, 48, 1, 1, 0, 1, False, False, False)):  # no QARMA4
            with self.assertRaises(ValueError, msg=str(args)):
                q.profile(*args)

    def test_pacga_is_not_an_addpac_mnemonic(self):
        with self.assertRaises(KeyError):
            q.sign(0, 0, "pacga", [0] * 10, _prof())
        with self.assertRaises(ValueError):
            q.is_instr_key("pacga")


class DecodeRegistersTest(unittest.TestCase):
    """decode_registers reads the ARM ARM field positions and rejects what the model cannot cover."""
    N3_TCR = (16 << 0) | (16 << 16) | (1 << 37) | (1 << 38) | (1 << 52)          # T0SZ=T1SZ=16, TBI0/1, TBID1
    QARMA3_PAUTH2 = (3 << 12) | (1 << 8)                                            # APA3=3, GPA3=1

    def test_n3_like_registers(self):
        r = q.decode_registers(self.N3_TCR, 0, self.QARMA3_PAUTH2)
        self.assertEqual(r, q.PacRegisters(3, 3, 3, 48, 48, True, True, False, True, False, False, False))
        self.assertTrue(q.profile(**r._asdict()).pauth2)

    def test_field_positions(self):
        tcr = (25 << 0) | (16 << 16) | (1 << 51) | (1 << 60)
        isar1 = (5 << 4) | (1 << 24)                                                 # APA=5, GPA=1
        r = q.decode_registers(tcr, isar1, 1 << 24)                                  # PAC_frac=1
        self.assertEqual((r.va_size0, r.va_size1, r.tbid0, r.mtx0), (39, 48, True, True))
        self.assertEqual((r.qarma_version, r.generic_qarma_version, r.auth_level, r.constpacfield),
                         (5, 5, 5, True))

    def test_rejects_unmodeled_or_inconsistent(self):
        for isar1, isar2 in ((1 << 8, 0),                    # API: IMPDEF address auth
                             (0, 0),                         # no address auth
                             (3 << 4, 3 << 12),              # APA and APA3
                             ((3 << 4) | (1 << 24), 1 << 8), # GPA and GPA3
                             (3 << 4, 2 << 24)):             # reserved PAC_frac
            with self.assertRaises(ValueError, msg=f"{isar1:#x} {isar2:#x}"):
                q.decode_registers(self.N3_TCR, isar1, isar2)


if __name__ == "__main__":
    unittest.main()
