import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import unittest
from src.aarch64.seal.relocation import (
    RelocType, Relocation, apply_relocations, read_word32,
    is_movk64, set_movk_imm16, get_movk_imm16, NOP_WORD)


class RelocationMechanismTest(unittest.TestCase):
    def test_word32_overwrites_and_leaves_base_untouched(self):
        base = bytes(range(16))
        out = apply_relocations(base, [Relocation(4, 0xAABBCCDD)])
        self.assertEqual(out[4:8], (0xAABBCCDD).to_bytes(4, "little"))
        self.assertEqual(out[:4], base[:4])
        self.assertEqual(out[8:], base[8:])
        self.assertEqual(base, bytes(range(16)))  # base not mutated

    def test_multiple_relocations_apply_in_place(self):
        base = bytes(12)
        out = apply_relocations(base, [Relocation(0, 0x11111111), Relocation(8, 0x22222222)])
        self.assertEqual(read_word32(out, 0), 0x11111111)
        self.assertEqual(read_word32(out, 4), 0)
        self.assertEqual(read_word32(out, 8), 0x22222222)

    def test_out_of_range_offset_raises(self):
        with self.assertRaises(ValueError):
            apply_relocations(bytes(8), [Relocation(6, 0)])

    def test_unknown_type_raises(self):
        bad = Relocation(0, 0, rtype=object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            apply_relocations(bytes(4), [bad])


class MovkBitSurgeryTest(unittest.TestCase):
    # `MOVK X0, #0x1234, LSL #48` assembles to 0xF2E24680 (imm16=0x1234 in bits [20:5]).
    MOVK_X0_1234_LSL48 = 0xF2E24680

    def test_recognizes_movk64(self):
        self.assertTrue(is_movk64(self.MOVK_X0_1234_LSL48))
        self.assertFalse(is_movk64(NOP_WORD))

    def test_reads_immediate(self):
        self.assertEqual(get_movk_imm16(self.MOVK_X0_1234_LSL48), 0x1234)

    def test_rewrites_only_the_immediate(self):
        patched = set_movk_imm16(self.MOVK_X0_1234_LSL48, 0xBEEF)
        self.assertEqual(get_movk_imm16(patched), 0xBEEF)
        # Rd (bits [4:0]) and hw (bits [22:21]) preserved.
        self.assertEqual(patched & 0x1F, self.MOVK_X0_1234_LSL48 & 0x1F)
        self.assertEqual((patched >> 21) & 0x3, (self.MOVK_X0_1234_LSL48 >> 21) & 0x3)

    def test_set_on_non_movk_asserts(self):
        with self.assertRaises(AssertionError):
            set_movk_imm16(NOP_WORD, 0)


if __name__ == "__main__":
    unittest.main()
