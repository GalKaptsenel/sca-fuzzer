import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import unittest
from src.aarch64.aarch64_relocations import (
    RelocType, Relocation, apply_relocations, read_word32,
    get_imm_field, set_imm_field, is_movk64, set_movk_imm16, get_movk_imm16,
    xpac_word, addg_word, movk_word, aut_word, NOP_WORD)
from src.aarch64.aarch64_generator import Aarch64Generator


def _asm_word(text: str) -> int:
    return int.from_bytes(Aarch64Generator.in_memory_assemble(text)[:4], "little")


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

    def test_value_too_wide_raises(self):
        with self.assertRaises(ValueError):
            apply_relocations(bytes(8), [Relocation(0, 0x1_0000_0000)])

    def test_overlapping_relocations_raise(self):
        with self.assertRaises(ValueError):
            apply_relocations(bytes(8), [Relocation(0, 1), Relocation(2, 2)])

    def test_read_word32_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            read_word32(bytes(6), 4)


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

    def test_set_on_non_movk_raises(self):
        with self.assertRaises(ValueError):
            set_movk_imm16(NOP_WORD, 0)

    def test_imm_field_roundtrip_and_width_check(self):
        w = set_imm_field(0, 5, 16, 0xBEEF)
        self.assertEqual(get_imm_field(w, 5, 16), 0xBEEF)
        with self.assertRaises(ValueError):
            set_imm_field(0, 5, 16, 0x1_0000)   # does not fit 16 bits


class ComputedEncodingTest(unittest.TestCase):
    """The computed strip/retag words must equal a real assembly — the byte-check the runtime used to
    do lives here instead, so a wrong encoding can never reach hardware and FPAC-fault the box."""

    def test_xpaci_matches_assembly(self):
        for rd in (0, 5, 17, 30):
            self.assertEqual(xpac_word(False, rd), _asm_word(f"xpaci x{rd}"))

    def test_xpacd_matches_assembly(self):
        for rd in (0, 5, 17, 30):
            self.assertEqual(xpac_word(True, rd), _asm_word(f"xpacd x{rd}"))

    def test_addg_matches_assembly(self):
        for rd in (0, 5, 17):
            for delta in (1, 7, 15):
                self.assertEqual(addg_word(rd, delta), _asm_word(f"addg x{rd}, x{rd}, #0, #{delta}"))

    def test_movk_sig_matches_assembly(self):
        for rd in (0, 5, 30):
            ref = _asm_word(f"movk x{rd}, #0x0, lsl #48")
            self.assertEqual(set_movk_imm16(ref, 0xBEEF), _asm_word(f"movk x{rd}, #0xbeef, lsl #48"))

    def test_movk_word_matches_assembly(self):
        for rd in (0, 5, 30):
            for shift in (0, 16, 32, 48):
                self.assertEqual(movk_word(rd, 0xBEEF, shift), _asm_word(f"movk x{rd}, #0xbeef, lsl #{shift}"))

    def test_aut_addressed_matches_assembly(self):
        for mn in ("autia", "autib", "autda", "autdb"):
            self.assertEqual(aut_word(mn, 5, 3), _asm_word(f"{mn} x5, x3"))

    def test_aut_zero_matches_assembly(self):
        for mn in ("autiza", "autizb", "autdza", "autdzb"):
            self.assertEqual(aut_word(mn, 5, 31), _asm_word(f"{mn} x5"))


if __name__ == "__main__":
    unittest.main()
