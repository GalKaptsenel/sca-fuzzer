"""
Unit tests for the speculative-exception carrier register file (src/aarch64/seal/carrier.py). Pure
logic, no hardware/CE: top-down allocation from v31, byte-aligned lanes that never straddle a
register, the SIMD-image byte mapping, config-driven window size, and loud overflow. Run from root:
    python -m unittest tests.aarch64_tests.unit_carrier_allocator
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.aarch64.seal.carrier import (                                        # noqa: E402
    CarrierAllocator, CarrierContribution, CarrierField, CarrierOverflow,
    CARRIER_REG_BYTES, CARRIER_SIMD_BASE_REG, MAX_CARRIER_REGS, TOP_VECTOR_REG)
from src.interfaces import SIMD_SUBREGION_SIZE                                 # noqa: E402


class AllocatorTest(unittest.TestCase):

    def test_allocates_top_down_from_v31(self):
        a = CarrierAllocator(reg_count=3)
        regs = [a.allocate(8).reg_index for _ in range(6)]   # two 8-byte lanes per 16-byte reg
        self.assertEqual(regs, [31, 31, 30, 30, 29, 29])
        self.assertEqual(a.reserved_registers(), [31, 30, 29])

    def test_byte_aligned_and_no_straddle(self):
        a = CarrierAllocator(reg_count=1)
        f1 = a.allocate(1)                                   # lane 1 at byte 0
        f2 = a.allocate(2)                                   # width 2 -> aligned up to byte 2
        f3 = a.allocate(8)                                   # width 8 -> aligned up to byte 8
        self.assertEqual((f1.reg_index, f1.byte_offset, f1.width), (31, 0, 1))
        self.assertEqual((f2.reg_index, f2.byte_offset, f2.width), (31, 2, 2))
        self.assertEqual((f3.reg_index, f3.byte_offset, f3.width), (31, 8, 8))
        for f in (f1, f2, f3):                               # never spills the register
            self.assertLessEqual(f.byte_offset + f.width, CARRIER_REG_BYTES)

    def test_field_does_not_fit_moves_to_next_register(self):
        a = CarrierAllocator(reg_count=2)
        a.allocate(8)                                        # v31 bytes 0..7
        f = a.allocate(8)                                    # v31 bytes 8..15
        self.assertEqual(f.reg_index, 31)
        g = a.allocate(2)                                    # v31 full -> v30
        self.assertEqual((g.reg_index, g.byte_offset), (30, 0))

    def test_simd_byte_offset_maps_top_slot_to_v31(self):
        self.assertEqual(CARRIER_SIMD_BASE_REG, TOP_VECTOR_REG + 1 - MAX_CARRIER_REGS)
        top = CarrierField(TOP_VECTOR_REG, 0, 8)             # v31, byte 0
        self.assertEqual(top.simd_byte_offset, (MAX_CARRIER_REGS - 1) * CARRIER_REG_BYTES)
        self.assertEqual(top.simd_byte_offset + top.width <= SIMD_SUBREGION_SIZE, True)
        base = CarrierField(CARRIER_SIMD_BASE_REG, 0, 1)     # lowest carrier reg -> image byte 0
        self.assertEqual(base.simd_byte_offset, 0)

    def test_field_taint_offsets_match_image_span(self):
        a = CarrierAllocator(reg_count=1)
        f = a.allocate(2)
        self.assertEqual(list(a.field_taint_offsets(f)),
                         [f.simd_byte_offset, f.simd_byte_offset + 1])

    def test_umov_source_operand(self):
        self.assertEqual(CarrierField(31, 0, 1).umov_source, "v31.b[0]")
        self.assertEqual(CarrierField(31, 4, 2).umov_source, "v31.h[2]")
        self.assertEqual(CarrierField(30, 8, 8).umov_source, "v30.d[1]")


class OverflowTest(unittest.TestCase):

    def test_overflow_raises_not_falls_back(self):
        a = CarrierAllocator(reg_count=1)                    # one register = 16 one-byte lanes
        for _ in range(CARRIER_REG_BYTES):
            a.allocate(1)
        with self.assertRaises(CarrierOverflow):
            a.allocate(1)

    def test_window_size_is_config_driven(self):
        # Same workload, different reserved counts -> overflow iff the window is too small.
        workload = [8, 8, 8, 8]                              # 4 lanes of 8 bytes = 2 full registers
        a2 = CarrierAllocator(reg_count=2)
        self.assertEqual([a2.allocate(w).reg_index for w in workload], [31, 31, 30, 30])
        a1 = CarrierAllocator(reg_count=1)
        a1.allocate(8)
        a1.allocate(8)
        with self.assertRaises(CarrierOverflow):
            a1.allocate(8)

    def test_reg_count_bounds_validated(self):
        with self.assertRaises(ValueError):
            CarrierAllocator(reg_count=0)
        with self.assertRaises(ValueError):
            CarrierAllocator(reg_count=MAX_CARRIER_REGS + 1)


class ContributionTest(unittest.TestCase):

    def test_write_field_little_endian_into_image(self):
        a = CarrierAllocator(reg_count=2)
        f = a.allocate(2)
        c = CarrierContribution()
        c.write_field(f, 0xBEEF)
        img = c.to_simd_bytes()
        self.assertEqual(len(img), SIMD_SUBREGION_SIZE)
        off = f.simd_byte_offset
        self.assertEqual(img[off:off + 2], b"\xef\xbe")
        self.assertEqual(c.written_ranges(), ((off, 2),))

    def test_distinct_fields_do_not_overlap(self):
        a = CarrierAllocator(reg_count=2)
        f1, f2 = a.allocate(8), a.allocate(8)
        c = CarrierContribution()
        c.write_field(f1, 0x1111111111111111)
        c.write_field(f2, 0x2222222222222222)
        img = c.to_simd_bytes()
        self.assertEqual(img[f1.simd_byte_offset:f1.simd_byte_offset + 8], b"\x11" * 8)
        self.assertEqual(img[f2.simd_byte_offset:f2.simd_byte_offset + 8], b"\x22" * 8)

    def test_value_wider_than_field_is_truncated(self):
        f = CarrierField(31, 0, 1)
        c = CarrierContribution()
        c.write_field(f, 0x1FF)                              # only the low byte lands
        self.assertEqual(c.to_simd_bytes()[f.simd_byte_offset], 0xFF)


if __name__ == "__main__":
    unittest.main()
