"""
Unit tests for global_seal_table.py (GST) and seal_values.py. No hardware/CE: top-down allocation,
capacity-in-bits (overflow raises), the SIMD byte mapping, little-endian image writes, the backend
read (UMOV), and each value's apply op (BFI/EOR). Run from root:
    python -m unittest tests.aarch64_tests.unit_global_seal_table
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.aarch64.seal.global_seal_table import (                                     # noqa: E402
    VectorGlobalSealTable, GlobalSealTableCapacityExceeded,
    VECTOR_REG_BYTES, SIMD_BASE_REG, TOP_REG)
from src.aarch64.seal.seal_values import PacSignature, MteTag, MaskFlip       # noqa: E402
from src.interfaces import SIMD_SUBREGION_SIZE                                # noqa: E402


def _reg_of(location):
    return int(location.load("x9")[0].template.split("v")[1].split(".")[0])


class VectorGlobalSealTableTest(unittest.TestCase):
    def test_capacity_bits_rounds_to_registers_top_down(self):
        f = VectorGlobalSealTable(capacity_bits=2 * VECTOR_REG_BYTES * 8)   # 2 registers
        regs = [_reg_of(f.allocate(8)) for _ in range(4)]             # two 8-byte values/register
        self.assertEqual(regs, [31, 31, 30, 30])

    def test_overflow_raises(self):
        f = VectorGlobalSealTable(capacity_bits=VECTOR_REG_BYTES * 8)       # 1 register = 16 bytes
        f.allocate(8)
        f.allocate(8)
        with self.assertRaises(GlobalSealTableCapacityExceeded):
            f.allocate(8)

    def test_byte_range_maps_top_register_to_end_of_image(self):
        f = VectorGlobalSealTable(capacity_bits=VECTOR_REG_BYTES * 8)
        loc = f.allocate(2)                                           # v31, bytes 0..1
        base = (TOP_REG - SIMD_BASE_REG) * VECTOR_REG_BYTES
        self.assertEqual(list(loc.byte_range()), [base, base + 1])
        self.assertLessEqual(loc.byte_range().stop, SIMD_SUBREGION_SIZE)

    def test_byte_aligned_lanes(self):
        f = VectorGlobalSealTable(capacity_bits=VECTOR_REG_BYTES * 8)
        self.assertEqual(f.allocate(1).byte_range().start % 1, 0)
        self.assertEqual(f.allocate(2).byte_range().start % 2, 0)    # width-2 aligned up to byte 2
        self.assertEqual(f.allocate(8).byte_range().start % 8, 0)

    def test_image_store_little_endian(self):
        f = VectorGlobalSealTable(capacity_bits=VECTOR_REG_BYTES * 8)
        loc = f.allocate(2)
        img = f.blank_image()
        img.store(loc, 0xBEEF)
        data = img.to_section().content
        self.assertEqual(len(data), SIMD_SUBREGION_SIZE)
        off = loc.byte_range().start
        self.assertEqual(data[off:off + 2], b"\xef\xbe")


class LoadTest(unittest.TestCase):
    def test_umov_uses_w_reg_for_subword_and_x_for_doubleword(self):
        f = VectorGlobalSealTable(capacity_bits=VECTOR_REG_BYTES * 8)
        byte_loc = f.allocate(1)
        self.assertEqual(byte_loc.load("x9")[0].template, "UMOV w9, v31.b[0]")
        f2 = VectorGlobalSealTable(capacity_bits=VECTOR_REG_BYTES * 8)
        dw_loc = f2.allocate(8)
        self.assertEqual(dw_loc.load("x9")[0].template, "UMOV x9, v31.d[0]")


class SealValueApplyTest(unittest.TestCase):
    def setUp(self):
        self.f = VectorGlobalSealTable(capacity_bits=VECTOR_REG_BYTES * 8)

    def test_pac_signature_loads_then_bfi(self):
        v = PacSignature(lsb=48, nbits=16)
        self.assertEqual(v.width, 2)
        templates = [i.template for i in v.apply(self.f.allocate(v.width), "x0", "x9")]
        self.assertEqual(templates, ["UMOV w9, v31.h[0]", "BFI x0, x9, #48, #16"])

    def test_mte_tag_bfi_writes_only_tag_bits(self):
        v = MteTag()
        self.assertEqual(v.width, 1)
        templates = [i.template for i in v.apply(self.f.allocate(v.width), "x0", "x9")]
        self.assertEqual(templates, ["UMOV w9, v31.b[0]", "BFI x0, x9, #56, #4"])

    def test_mask_flip_window_loads_then_shifted_eor(self):
        # A mask confined to [54:48] packs into one byte; the EOR shifts it back into place.
        v = MaskFlip(lo=48, nbits=7)
        self.assertEqual(v.width, 1)
        self.assertEqual(v.encode(0x7F << 48), 0x7F)
        templates = [i.template for i in v.apply(self.f.allocate(v.width), "x0", "x9")]
        self.assertEqual(templates, ["UMOV w9, v31.b[0]", "EOR x0, x0, x9, lsl #48"])

    def test_mask_flip_low_window_needs_no_shift(self):
        v = MaskFlip(lo=0, nbits=8)
        self.assertEqual(v.encode(0xAB), 0xAB)
        templates = [i.template for i in v.apply(self.f.allocate(v.width), "x0", "x9")]
        self.assertEqual(templates, ["UMOV w9, v31.b[0]", "EOR x0, x0, x9"])


if __name__ == "__main__":
    unittest.main()
