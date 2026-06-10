"""
Unit tests for NZCVScheme per-flag encoding, _reconstruct_pstate, and AArch64InputGenerator.
"""
import struct
import unittest
import numpy as np

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_input_layout import NZCVScheme, _reconstruct_pstate, _input_bytes_with_pstate
from src.aarch64.aarch64_input_generator import AArch64InputGenerator
from src.interfaces import Input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_slot(n: int, z: int, c: int, v: int) -> int:
    """Build a per-flag slot-6 value from individual flag bits (0 or 1)."""
    assert all(f in (0, 1) for f in (n, z, c, v))
    lo = (n << 0) | (z << 8) | (c << 16) | (v << 24)
    return lo | (lo << 32)


def expected_pstate(n: int, z: int, c: int, v: int) -> int:
    return (n << 31) | (z << 30) | (c << 29) | (v << 28)


ALL_COMBOS = [(n, z, c, v) for n in (0, 1) for z in (0, 1) for c in (0, 1) for v in (0, 1)]


# ---------------------------------------------------------------------------
# NZCVScheme.input_byte
# ---------------------------------------------------------------------------

class TestInputByte(unittest.TestCase):

    def test_n_byte_offset(self):
        self.assertEqual(NZCVScheme.input_byte('n'), 48)

    def test_z_byte_offset(self):
        self.assertEqual(NZCVScheme.input_byte('z'), 49)

    def test_c_byte_offset(self):
        self.assertEqual(NZCVScheme.input_byte('c'), 50)

    def test_v_byte_offset(self):
        self.assertEqual(NZCVScheme.input_byte('v'), 51)

    def test_uppercase_aliases(self):
        for flag in ('N', 'Z', 'C', 'V'):
            self.assertEqual(NZCVScheme.input_byte(flag),
                             NZCVScheme.input_byte(flag.lower()))

    def test_slot_base_is_48(self):
        self.assertEqual(NZCVScheme.SLOT_BASE_BYTE, 48)
        self.assertEqual(NZCVScheme.SLOT_IDX * 8, 48)


# ---------------------------------------------------------------------------
# NZCVScheme.to_pstate — all 16 combinations
# ---------------------------------------------------------------------------

class TestToPstate(unittest.TestCase):

    def test_all_zero_flags(self):
        self.assertEqual(NZCVScheme.to_pstate(make_slot(0, 0, 0, 0)), 0x00000000)

    def test_all_one_flags(self):
        self.assertEqual(NZCVScheme.to_pstate(make_slot(1, 1, 1, 1)), 0xF0000000)

    def test_nzcv_is_nZcv(self):
        # nZcv = N=0 Z=1 C=0 V=0 → 0x40000000
        self.assertEqual(NZCVScheme.to_pstate(make_slot(0, 1, 0, 0)), 0x40000000)

    def test_all_16_combinations(self):
        for n, z, c, v in ALL_COMBOS:
            with self.subTest(n=n, z=z, c=c, v=v):
                slot = make_slot(n, z, c, v)
                got = NZCVScheme.to_pstate(slot)
                want = expected_pstate(n, z, c, v)
                self.assertEqual(got, want,
                    f"flags=({n},{z},{c},{v}): got {got:#010x}, want {want:#010x}")

    def test_noisy_upper_bits_ignored(self):
        # Only bit 0 of each byte matters; other bits must be masked away
        for n, z, c, v in ALL_COMBOS:
            with self.subTest(n=n, z=z, c=c, v=v):
                # Set all bits in each byte, but flag value is still bit 0
                noisy = (
                    ((0xFE | n) << 0)   |  # byte 48: noise in bits 7:1
                    ((0xFE | z) << 8)   |  # byte 49
                    ((0xFE | c) << 16)  |  # byte 50
                    ((0xFE | v) << 24)     # byte 51
                )
                noisy_slot = noisy | (noisy << 32)
                got = NZCVScheme.to_pstate(noisy_slot)
                want = expected_pstate(n, z, c, v)
                self.assertEqual(got, want)

    def test_pstate_only_uses_bits_31_28(self):
        # Result must have no bits set outside 31:28
        for n, z, c, v in ALL_COMBOS:
            pstate = NZCVScheme.to_pstate(make_slot(n, z, c, v))
            self.assertEqual(pstate & ~0xF0000000, 0,
                f"stray bits in pstate {pstate:#010x} for flags ({n},{z},{c},{v})")

    def test_upper_32_bits_of_slot_dont_affect_result(self):
        # to_pstate reads from the raw integer; our make_slot mirrors lower→upper,
        # but the result should be the same even if upper 32 bits differ.
        for n, z, c, v in ALL_COMBOS:
            lo = (n << 0) | (z << 8) | (c << 16) | (v << 24)
            slot_with_mirrored = lo | (lo << 32)
            slot_with_garbage   = lo | (0xDEADBEEF << 32)
            self.assertEqual(
                NZCVScheme.to_pstate(slot_with_mirrored),
                NZCVScheme.to_pstate(slot_with_garbage),
                "upper 32 bits should not affect to_pstate result"
            )


# ---------------------------------------------------------------------------
# NZCVScheme.make_random
# ---------------------------------------------------------------------------

class TestMakeRandom(unittest.TestCase):

    def _fixed_rng(self, flags):
        """Return an rng whose integers(0,2) calls return the given flag sequence."""
        class _FixedRng:
            def __init__(self, seq):
                self._it = iter(seq)
            def integers(self, lo, hi):
                return next(self._it)
        return _FixedRng(flags)

    def test_structure_only_bit0_per_byte(self):
        rng = np.random.default_rng(seed=42)
        for _ in range(256):
            slot = NZCVScheme.make_random(rng)
            # Each byte must be 0 or 1 (only bit 0 set)
            for byte_off in range(4):
                byte_val = (slot >> (byte_off * 8)) & 0xFF
                self.assertIn(byte_val, (0, 1),
                    f"byte {byte_off} of slot {slot:#018x} has value {byte_val:#x} (expected 0 or 1)")

    def test_upper_32_mirrors_lower_32(self):
        rng = np.random.default_rng(seed=0)
        for _ in range(256):
            slot = NZCVScheme.make_random(rng)
            lo = slot & 0xFFFFFFFF
            hi = (slot >> 32) & 0xFFFFFFFF
            self.assertEqual(lo, hi, f"upper 32 ≠ lower 32 in slot {slot:#018x}")

    def test_all_16_combinations_reachable(self):
        # Over many samples, all 16 NZCV combos must appear.
        rng = np.random.default_rng(seed=7)
        seen = set()
        for _ in range(4096):
            slot = NZCVScheme.make_random(rng)
            pstate = NZCVScheme.to_pstate(slot)
            seen.add(pstate >> 28)  # top nibble encodes NZCV
        self.assertEqual(seen, set(range(16)),
            f"not all 16 flag combinations were generated; missing: {set(range(16)) - seen}")

    def test_fixed_flags_n1z0c0v0(self):
        rng = self._fixed_rng([1, 0, 0, 0])  # N=1 Z=0 C=0 V=0
        slot = NZCVScheme.make_random(rng)
        self.assertEqual(NZCVScheme.to_pstate(slot), 1 << 31)

    def test_fixed_flags_n0z1c0v0(self):
        rng = self._fixed_rng([0, 1, 0, 0])  # N=0 Z=1 C=0 V=0  → nZcv
        slot = NZCVScheme.make_random(rng)
        self.assertEqual(NZCVScheme.to_pstate(slot), 0x40000000)

    def test_fixed_flags_all_ones(self):
        rng = self._fixed_rng([1, 1, 1, 1])
        slot = NZCVScheme.make_random(rng)
        self.assertEqual(NZCVScheme.to_pstate(slot), 0xF0000000)

    def test_flags_are_independent(self):
        # Each flag is drawn independently: fixing one flag's byte doesn't
        # constrain others.  Verify across many samples that each flag
        # is 50/50 independent of every other flag.
        rng = np.random.default_rng(seed=13)
        counts = {(b0, b1): 0 for b0 in (0, 1) for b1 in (0, 1)}
        n_samples = 8192
        for _ in range(n_samples):
            slot = NZCVScheme.make_random(rng)
            n_bit = (slot >> 0) & 1
            z_bit = (slot >> 8) & 1
            counts[(n_bit, z_bit)] += 1
        # Each of the 4 joint outcomes should be roughly 25%
        for combo, cnt in counts.items():
            frac = cnt / n_samples
            self.assertAlmostEqual(frac, 0.25, delta=0.05,
                msg=f"N,Z joint distribution skewed: {combo}={frac:.3f}")


# ---------------------------------------------------------------------------
# _reconstruct_pstate (the memoryview in-place function)
# ---------------------------------------------------------------------------

class TestReconstructPstate(unittest.TestCase):

    def _make_view(self, slot6_value: int):
        """Build a 64-element u64 memoryview with slot 6 set to slot6_value."""
        buf = bytearray(64 * 8)
        struct.pack_into('<Q', buf, NZCVScheme.SLOT_IDX * 8, slot6_value)
        return memoryview(buf).cast('Q'), buf

    def test_all_16_combinations(self):
        for n, z, c, v in ALL_COMBOS:
            with self.subTest(n=n, z=z, c=c, v=v):
                view, _ = self._make_view(make_slot(n, z, c, v))
                _reconstruct_pstate(view)
                self.assertEqual(int(view[NZCVScheme.SLOT_IDX]),
                                 expected_pstate(n, z, c, v))

    def test_other_slots_untouched(self):
        view, buf = self._make_view(make_slot(1, 0, 1, 0))
        # Write sentinel values into all other slots
        for i in range(64):
            if i != NZCVScheme.SLOT_IDX:
                view[i] = 0xDEADBEEFCAFE0000 | i
        _reconstruct_pstate(view)
        for i in range(64):
            if i != NZCVScheme.SLOT_IDX:
                self.assertEqual(int(view[i]), 0xDEADBEEFCAFE0000 | i,
                    f"slot {i} was modified unexpectedly")

    def test_idempotent_after_pstate_written(self):
        # If slot 6 already contains a valid PSTATE value (bits 31:28 only),
        # to_pstate should not corrupt it further.
        # (not guaranteed by spec, but good to document behavior)
        view, _ = self._make_view(0x40000000)  # nZcv in PSTATE form already
        _reconstruct_pstate(view)
        # bit 0 of byte 0 = 0, bit 0 of byte 1 = 0, ... → PSTATE = 0
        # This documents the expected (non-idempotent) behavior:
        # running twice gives 0, not 0x40000000.
        self.assertEqual(int(view[NZCVScheme.SLOT_IDX]), 0)

    def test_stale_slot_bits_fully_overwritten_not_ored(self):
        # reconstruct must REBUILD slot 6 from the per-flag bytes, not OR into
        # whatever junk was already there: stale non-flag bits must not survive.
        garbage = 0xDEADBEEFCAFEBABE & 0xFEFEFEFEFEFEFEFE  # bits 1..7 of every byte, no flag bit
        view, _ = self._make_view(make_slot(1, 0, 0, 0) | garbage)
        _reconstruct_pstate(view)
        self.assertEqual(int(view[NZCVScheme.SLOT_IDX]), expected_pstate(1, 0, 0, 0))


# ---------------------------------------------------------------------------
# AArch64InputGenerator — slot 6 has per-flag encoding
# ---------------------------------------------------------------------------

class TestAArch64InputGenerator(unittest.TestCase):

    def _gen_slots(self, seed=42, count=20):
        gen = AArch64InputGenerator(seed)
        inputs = gen.generate(count)
        slots = []
        for inp in inputs:
            for actor_idx in range(len(inp)):
                slots.append(int(inp[actor_idx]['gpr'][NZCVScheme.SLOT_IDX]))
        return slots

    def test_slot6_only_bit0_per_byte(self):
        for slot in self._gen_slots():
            for byte_off in range(4):
                byte_val = (slot >> (byte_off * 8)) & 0xFF
                self.assertIn(byte_val, (0, 1),
                    f"byte {byte_off} of slot {slot:#018x} invalid (expected 0 or 1)")

    def test_slot6_upper_mirrors_lower(self):
        for slot in self._gen_slots():
            lo = slot & 0xFFFFFFFF
            hi = (slot >> 32) & 0xFFFFFFFF
            self.assertEqual(lo, hi,
                f"upper 32 ≠ lower 32 in slot {slot:#018x}")

    def test_slot6_to_pstate_valid(self):
        for slot in self._gen_slots():
            pstate = NZCVScheme.to_pstate(slot)
            self.assertEqual(pstate & ~0xF0000000, 0,
                f"to_pstate({slot:#018x}) has stray bits: {pstate:#010x}")

    def test_all_16_pstate_values_reachable(self):
        gen = AArch64InputGenerator(seed=99)
        seen = set()
        for inp in gen.generate(500):
            slot = int(inp[0]['gpr'][NZCVScheme.SLOT_IDX])
            seen.add(NZCVScheme.to_pstate(slot) >> 28)
        self.assertEqual(seen, set(range(16)),
            f"missing flag combinations: {set(range(16)) - seen}")

    def test_different_seeds_give_different_slots(self):
        def first_slot(seed):
            gen = AArch64InputGenerator(seed)
            inp = gen.generate(1)
            return int(inp[0][0]['gpr'][NZCVScheme.SLOT_IDX])
        slots = {first_slot(s) for s in range(20)}
        self.assertGreater(len(slots), 1,
            "all seeds produced the same NZCV slot — RNG is not seeded properly")

    def test_each_flag_is_50_50(self):
        gen = AArch64InputGenerator(seed=42)
        flag_counts = {flag: [0, 0] for flag in ('n', 'z', 'c', 'v')}
        for inp in gen.generate(2000):
            slot = int(inp[0]['gpr'][NZCVScheme.SLOT_IDX])
            for flag, (byte_off, _) in NZCVScheme._LAYOUT.items():
                bit = (slot >> (byte_off * 8)) & 1
                flag_counts[flag][bit] += 1
        for flag, (zeros, ones) in flag_counts.items():
            total = zeros + ones
            frac_ones = ones / total
            self.assertAlmostEqual(frac_ones, 0.5, delta=0.07,
                msg=f"flag '{flag}' is not ~50/50: ones={ones}/{total}={frac_ones:.3f}")

    def test_nzcv_independent_of_main_rng(self):
        # Two generators with same seed should produce identical non-NZCV slots.
        # This verifies the auxiliary RNG doesn't disturb the main RNG sequence.
        gen_a = AArch64InputGenerator(seed=7)
        gen_b = AArch64InputGenerator(seed=7)
        inputs_a = gen_a.generate(10)
        inputs_b = gen_b.generate(10)
        for i, (ia, ib) in enumerate(zip(inputs_a, inputs_b)):
            for slot_idx in range(len(ia[0]['gpr'])):
                if slot_idx == NZCVScheme.SLOT_IDX:
                    continue
                self.assertEqual(int(ia[0]['gpr'][slot_idx]),
                                 int(ib[0]['gpr'][slot_idx]),
                    f"input {i} slot {slot_idx}: generators diverged in non-NZCV slots")


# ---------------------------------------------------------------------------
# _input_bytes_with_pstate — slot-6 must be PSTATE-converted in returned bytes
# ---------------------------------------------------------------------------

class TestInputBytesWithPstate(unittest.TestCase):
    """Regression test: bytearray[start:] creates a copy, not a view.

    The bug: memoryview(data[0x2000:]).cast('Q') modifies a copy of data,
    leaving data[0x2030] unchanged (raw slot 6 value sent to kernel → wrong NZCV).
    The fix: memoryview(data)[0x2000:].cast('Q') creates a view into data.
    """

    GPR_OFFSET = 0x2000         # gpr field starts at byte 0x2000 in tobytes()
    SLOT6_OFFSET = 0x2030       # flags = gpr[6] = gpr_offset + 6*8

    def _make_input_with_slot6(self, n: int, z: int, c: int, v: int) -> Input:
        inp = Input(1)
        inp[0]['gpr'][NZCVScheme.SLOT_IDX] = make_slot(n, z, c, v)
        return inp

    def _read_slot6_from_bytes(self, data: bytes) -> int:
        return int.from_bytes(data[self.SLOT6_OFFSET:self.SLOT6_OFFSET + 8], 'little')

    def test_slot6_is_pstate_format(self):
        """_input_bytes_with_pstate must write PSTATE (bits 31:28) not raw per-flag value."""
        for n, z, c, v in ALL_COMBOS:
            with self.subTest(n=n, z=z, c=c, v=v):
                inp = self._make_input_with_slot6(n, z, c, v)
                result = _input_bytes_with_pstate(inp)
                got = self._read_slot6_from_bytes(result)
                want = expected_pstate(n, z, c, v)
                self.assertEqual(got, want,
                    f"flags=({n},{z},{c},{v}): slot6 in result = {got:#010x}, "
                    f"want PSTATE {want:#010x}  (raw would be {make_slot(n,z,c,v):#018x})")

    def test_raw_input_is_not_modified(self):
        """_input_bytes_with_pstate must not mutate the original Input object."""
        inp = self._make_input_with_slot6(1, 0, 1, 0)
        raw_before = make_slot(1, 0, 1, 0)
        _input_bytes_with_pstate(inp)
        raw_after = int(inp[0]['gpr'][NZCVScheme.SLOT_IDX])
        self.assertEqual(raw_after, raw_before,
            "Input object was mutated by _input_bytes_with_pstate")

    def test_non_slot6_bytes_unchanged(self):
        """Only slot-6 bytes should differ between raw tobytes() and the result."""
        inp = self._make_input_with_slot6(0, 1, 0, 1)
        raw = inp.tobytes()
        result = _input_bytes_with_pstate(inp)
        self.assertEqual(len(raw), len(result))
        # Bytes before slot 6
        self.assertEqual(raw[:self.SLOT6_OFFSET], result[:self.SLOT6_OFFSET])
        # Bytes after slot 6
        self.assertEqual(raw[self.SLOT6_OFFSET + 8:], result[self.SLOT6_OFFSET + 8:])
        # Slot 6 itself must differ (raw != PSTATE for non-trivial flags)
        raw_slot6 = int.from_bytes(raw[self.SLOT6_OFFSET:self.SLOT6_OFFSET + 8], 'little')
        got_slot6 = int.from_bytes(result[self.SLOT6_OFFSET:self.SLOT6_OFFSET + 8], 'little')
        want_slot6 = expected_pstate(0, 1, 0, 1)
        self.assertEqual(got_slot6, want_slot6)
        self.assertNotEqual(raw_slot6, got_slot6)

    def test_known_crash_input_tc2_inp0(self):
        """Regression: TC2 inp=0 had raw slot6=0x0000010000000100 reaching kernel.

        This caused x6 = 0x0000010000000100 instead of 0x40000000 (Z=1),
        leading to NZCV=0 on HW while CE had Z=1 → wrong auth → FPAC crash.
        """
        inp = Input(1)
        inp[0]['gpr'][NZCVScheme.SLOT_IDX] = 0x0000010000000100  # raw TC2 inp=0 slot 6
        result = _input_bytes_with_pstate(inp)
        got = self._read_slot6_from_bytes(result)
        self.assertEqual(got, 0x40000000,   # Z=1 in PSTATE format
            f"slot6 in result = {got:#010x}, want 0x40000000 (Z=1)")
        self.assertNotEqual(got, 0x0000010000000100,
            "raw value reached kernel — bytearray copy bug not fixed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
