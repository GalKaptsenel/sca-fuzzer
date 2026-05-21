"""
Integration tests for PAC TC1/TC2/TC3 generation using the real ISA and generator.

Unlike unit_pac_generator.py (which mocks the ISA), these tests load the real base.json
instruction set, instantiate Aarch64RandomGenerator and PACInstrumentation, and run the
full instrument_stage1 → inject values → instrument_stage2 pipeline.  They verify the
same structural invariants but on realistic test cases, exercising instruction selection
and taint analysis that the unit tests cannot reach.

Groups:
  1. TestPacE2ESlotStructure     — TC1/TC2/TC3 structural invariants across 100 real TCs
  2. TestPacE2ETc3SpecCombos     — all 4 (ctx,ptr) combos appear for spec/non-arch slots
  3. TestPacE2ETc3NonArchPath    — spec_nesting=None treated as spec (TC3 ≠ TC2)
  4. TestPacE2EValueRoundTrip    — MOVZ+MOVK×3 encodes signed_value and ctx_value losslessly
  5. TestPacE2EArchSlotsCorrect  — arch slots: TC3 == TC2, AUTH always succeeds semantically
"""
import re
import copy
import os
import random
import sys
import tempfile
import unittest
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import (
    Aarch64RandomGenerator, PACInstrumentation, FixPoint, SignedRegInfo,
    SLOT_SIZE, FIX_COUNT_CTX, FIX_COUNT_PTR,
    CTX_SLOT_START, PTR_SLOT_START, AUTH_SLOT_POS,
    _PAC_TO_AUTH,
)

# ===========================================================================
# Module-level setup — load ISA once for all tests
# ===========================================================================

_ISA: Optional[InstructionSet] = None
_TMPDIR = None

_SIGN_TO_AUTH = {
    "pacia": "autia",   "pacib": "autib",
    "pacda": "autda",   "pacdb": "autdb",
    "paciza": "autiza", "pacizb": "autizb",
    "pacdza": "autdza", "pacdzb": "autdzb",
}
_SIGN_TO_XPAC = {
    "pacia": "xpaci",  "pacib": "xpaci",
    "pacda": "xpacd",  "pacdb": "xpacd",
    "paciza": "xpaci", "pacizb": "xpaci",
    "pacdza": "xpacd", "pacdzb": "xpacd",
}
_AUTH_NAMES = set(_SIGN_TO_AUTH.values())
_XPAC_NAMES = {"xpaci", "xpacd"}


def setUpModule():
    global _ISA, _TMPDIR
    CONF.load("config.yml")
    _ISA = InstructionSet("base.json", CONF.instruction_categories)
    _TMPDIR = tempfile.mkdtemp()


def tearDownModule():
    import shutil
    if _TMPDIR:
        shutil.rmtree(_TMPDIR, ignore_errors=True)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_gen(seed: int) -> Aarch64RandomGenerator:
    return Aarch64RandomGenerator(_ISA, seed)


def _make_pac(gen: Aarch64RandomGenerator) -> PACInstrumentation:
    return PACInstrumentation(gen, xpac_weight=2, auth_weight=3, sign_weight=3)


def _slot_insts(tc, slot_id: int) -> Optional[List]:
    pos_map: Dict[int, object] = {}
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                if getattr(inst, "_pac_slot_id", None) == slot_id:
                    pos_map[inst._pac_slot_pos] = inst
    if len(pos_map) != SLOT_SIZE:
        return None
    return [pos_map[p] for p in range(SLOT_SIZE)]


def _decode_value(insts) -> Optional[int]:
    val = 0
    for inst in insts:
        m_imm = re.search(r'#0x([0-9a-fA-F]+)', inst.template or "")
        m_lsl = re.search(r'LSL #(\d+)', inst.template or "")
        if not m_imm:
            return None
        val |= int(m_imm.group(1), 16) << (int(m_lsl.group(1)) if m_lsl else 0)
    return val


def _inject_values(fix_points: List[FixPoint], rng: random.Random):
    """Inject realistic fake CE-captured values into fix_points."""
    for fp in fix_points:
        # Realistic signed pointer: canonical address | PAC signature in top 16 bits.
        canonical = (rng.randint(0, 0xFFFF_FFFF_FFFF) & ~0xF) | 0x4000_0000_0000
        pac_sig   = rng.randint(0, 0xFFFF)
        fp.signed_value = canonical | (pac_sig << 48)
        if fp.info.ctx_reg is not None:
            fp.ctx_value = rng.randint(0, 0xFFFF_FFFF_FFFF_FFFF)
        fp.spec_nesting = 0  # default: arch


def _gen_tc_with_slots(seed: int):
    """Return (fix_points, prep_tc, pac) for a TC with at least one PAC slot, or None."""
    gen = _make_gen(seed)
    pac = _make_pac(gen)
    asm_path = os.path.join(_TMPDIR, f"tc_{seed}.asm")
    try:
        tc = gen.create_test_case(asm_path, disable_assembler=True)
        prep_tc, fix_points = pac.instrument_stage1(copy.deepcopy(tc))
    except Exception:
        return None
    if not fix_points:
        return None
    return fix_points, prep_tc, pac


# ===========================================================================
# 1. TestPacE2ESlotStructure
# ===========================================================================

class TestPacE2ESlotStructure(unittest.TestCase):
    """Structural invariants of TC1/TC2/TC3 using real ISA and real TCs."""

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        rng = random.Random(42)
        for seed in range(300):
            result = _gen_tc_with_slots(seed)
            if result is None:
                continue
            fix_points, prep_tc, pac = result
            _inject_values(fix_points, rng)
            tc1, tc2, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            cls.cases.append((fix_points, tc1, tc2, tc3))
            if len(cls.cases) >= 100:
                break
        assert len(cls.cases) >= 20, f"Too few TCs with PAC slots: {len(cls.cases)}"

    def _all_slots(self):
        for fix_points, tc1, tc2, tc3 in self.cases:
            for fp in fix_points:
                s1 = _slot_insts(tc1, fp.slot_id)
                s2 = _slot_insts(tc2, fp.slot_id)
                s3 = _slot_insts(tc3, fp.slot_id)
                if s1 and s2 and s3:
                    yield fp, s1, s2, s3

    # ── TC1 invariants ────────────────────────────────────────────────────

    def test_tc1_pos8_is_xpac_never_auth(self):
        for fp, s1, _, _ in self._all_slots():
            with self.subTest(slot=fp.slot_id, pac=fp.info.pac_mnemonic):
                self.assertIn(s1[AUTH_SLOT_POS].name, _XPAC_NAMES,
                              f"TC1 pos 8 is not XPAC: {s1[AUTH_SLOT_POS].name}")
                self.assertNotIn(s1[AUTH_SLOT_POS].name, _AUTH_NAMES)

    def test_tc1_no_auth_anywhere(self):
        for fp, s1, _, _ in self._all_slots():
            for pos, inst in enumerate(s1):
                with self.subTest(slot=fp.slot_id, pos=pos):
                    self.assertNotIn(inst.name, _AUTH_NAMES,
                                     f"AUTH found in TC1 at pos {pos}: {inst.name}")

    def test_tc1_xpac_matches_signing_variant(self):
        for fp, s1, _, _ in self._all_slots():
            expected = _SIGN_TO_XPAC.get(fp.info.pac_mnemonic.lower())
            with self.subTest(slot=fp.slot_id, pac=fp.info.pac_mnemonic):
                self.assertEqual(s1[AUTH_SLOT_POS].name, expected)

    def test_tc1_ptr_encodes_signed_value(self):
        for fp, s1, _, _ in self._all_slots():
            encoded = _decode_value(s1[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
            with self.subTest(slot=fp.slot_id):
                self.assertEqual(encoded, fp.signed_value)

    def test_tc1_ctx_encodes_ctx_value_or_nop(self):
        for fp, s1, _, _ in self._all_slots():
            if fp.info.ctx_reg is not None:
                encoded = _decode_value(s1[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX])
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(encoded, fp.ctx_value)
            else:
                for i in range(FIX_COUNT_CTX):
                    with self.subTest(slot=fp.slot_id, pos=i):
                        self.assertEqual(s1[i].name, "nop")

    # ── TC2 invariants ────────────────────────────────────────────────────

    def test_tc2_pos8_is_auth_matching_variant(self):
        for fp, _, s2, _ in self._all_slots():
            expected = _SIGN_TO_AUTH.get(fp.info.pac_mnemonic.lower())
            with self.subTest(slot=fp.slot_id, pac=fp.info.pac_mnemonic):
                self.assertEqual(s2[AUTH_SLOT_POS].name, expected)

    def test_tc2_ptr_encodes_signed_value(self):
        for fp, _, s2, _ in self._all_slots():
            encoded = _decode_value(s2[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
            with self.subTest(slot=fp.slot_id):
                self.assertEqual(encoded, fp.signed_value)

    # ── TC3 arch slots == TC2 ─────────────────────────────────────────────

    def test_tc3_arch_slots_equal_tc2(self):
        """spec_nesting=0 (arch): TC3 must be identical to TC2."""
        for fp, _, s2, s3 in self._all_slots():
            if fp.spec_nesting != 0:
                continue
            with self.subTest(slot=fp.slot_id):
                for pos in range(SLOT_SIZE):
                    self.assertEqual(s2[pos].name, s3[pos].name,
                                     f"pos {pos}: TC2={s2[pos].name} TC3={s3[pos].name}")


# ===========================================================================
# 2. TestPacE2ETc3SpecCombos
# ===========================================================================

class TestPacE2ETc3SpecCombos(unittest.TestCase):
    """TC3 speculative slots use AUTH + one of 4 (ctx,ptr) combos randomly."""

    def _classify_combo(self, s3, fp: FixPoint):
        """Return (ctx_restored, ptr_restored) for a TC3 spec slot."""
        ctx_restored = (s3[CTX_SLOT_START].name == "movz") if fp.info.ctx_reg is not None else False
        ptr_restored  = (s3[PTR_SLOT_START].name == "movz")
        return ctx_restored, ptr_restored

    def test_auth_always_at_pos8_for_spec_slots(self):
        """All 4 combos share: AUTH at pos 8 regardless of ctx/ptr choice."""
        result = _gen_tc_with_slots(0)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result
        rng = random.Random(7)
        _inject_values(fix_points, rng)
        for fp in fix_points:
            fp.spec_nesting = 1  # force spec

        for seed in range(40):
            random.seed(seed)
            _, _, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            for fp in fix_points:
                s3 = _slot_insts(tc3, fp.slot_id)
                if s3:
                    with self.subTest(seed=seed, slot=fp.slot_id):
                        self.assertIn(s3[AUTH_SLOT_POS].name, _AUTH_NAMES,
                                      f"pos 8 is {s3[AUTH_SLOT_POS].name}, expected AUTH")

    def test_all_four_combos_appear_for_spec_slots(self):
        """Over enough seeds, all four (ctx_restored, ptr_restored) combos must appear."""
        result = _gen_tc_with_slots(0)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result

        # Find a slot with a ctx_reg (needed to see ctx=True combos)
        ctx_fp = next((fp for fp in fix_points if fp.info.ctx_reg is not None), None)
        if ctx_fp is None:
            self.skipTest("No slot with ctx_reg in seed=0 TC; can't test ctx combos")

        rng = random.Random(9)
        _inject_values(fix_points, rng)
        for fp in fix_points:
            fp.spec_nesting = 1  # all spec

        seen = set()
        for seed in range(300):
            random.seed(seed)
            _, _, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            s3 = _slot_insts(tc3, ctx_fp.slot_id)
            if s3:
                seen.add(self._classify_combo(s3, ctx_fp))
            if seen == {(False, False), (True, False), (False, True), (True, True)}:
                break

        self.assertEqual(seen, {(False, False), (True, False), (False, True), (True, True)},
                         f"Only saw combos: {seen} after 300 seeds")

    def test_combo_false_false_nops_in_ctx_and_ptr(self):
        """(False,False): positions 0–3 and 4–7 are all NOPs."""
        result = _gen_tc_with_slots(1)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result
        rng = random.Random(3)
        _inject_values(fix_points, rng)
        fp = fix_points[0]
        fp.spec_nesting = 1

        found = False
        for seed in range(200):
            random.seed(seed)
            _, _, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            s3 = _slot_insts(tc3, fp.slot_id)
            if s3 and self._classify_combo(s3, fp) == (False, False):
                for i in range(FIX_COUNT_CTX):
                    self.assertEqual(s3[i].name, "nop")
                for i in range(FIX_COUNT_PTR):
                    self.assertEqual(s3[PTR_SLOT_START + i].name, "nop")
                found = True
                break
        self.assertTrue(found, "(False,False) combo not observed in 200 seeds")

    def test_combo_false_true_has_ptr_restore(self):
        """(False,True): ctx positions are NOPs, ptr positions are MOVZ+MOVK×3."""
        result = _gen_tc_with_slots(2)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result
        rng = random.Random(5)
        _inject_values(fix_points, rng)
        fp = fix_points[0]
        fp.spec_nesting = 1

        found = False
        for seed in range(200):
            random.seed(seed)
            _, _, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            s3 = _slot_insts(tc3, fp.slot_id)
            if s3 and self._classify_combo(s3, fp) == (False, True):
                for i in range(FIX_COUNT_CTX):
                    if fp.info.ctx_reg is not None:
                        self.assertEqual(s3[i].name, "nop")
                # ptr: MOVZ+MOVK×3
                self.assertEqual(s3[PTR_SLOT_START].name, "movz")
                for i in range(1, FIX_COUNT_PTR):
                    self.assertEqual(s3[PTR_SLOT_START + i].name, "movk")
                # value round-trip
                encoded = _decode_value(s3[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
                self.assertEqual(encoded, fp.signed_value)
                found = True
                break
        self.assertTrue(found, "(False,True) combo not observed in 200 seeds")


# ===========================================================================
# 3. TestPacE2ETc3NonArchPath
# ===========================================================================

class TestPacE2ETc3NonArchPath(unittest.TestCase):
    """spec_nesting=None means CE never executed the signing → treated as spec in TC3."""

    def test_non_arch_tc3_has_auth_at_8(self):
        result = _gen_tc_with_slots(10)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result
        rng = random.Random(11)
        _inject_values(fix_points, rng)
        for fp in fix_points:
            fp.spec_nesting = None  # simulate non-arch path

        random.seed(0)
        _, _, tc3 = pac.instrument_stage2(prep_tc, fix_points)
        for fp in fix_points:
            s3 = _slot_insts(tc3, fp.slot_id)
            if s3:
                with self.subTest(slot=fp.slot_id):
                    self.assertIn(s3[AUTH_SLOT_POS].name, _AUTH_NAMES)

    def test_non_arch_tc3_differs_from_tc2_in_at_least_one_slot(self):
        """spec_nesting=None makes TC3 use random combo — at least sometimes ≠ TC2."""
        result = _gen_tc_with_slots(20)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result
        rng = random.Random(21)
        _inject_values(fix_points, rng)
        for fp in fix_points:
            fp.spec_nesting = None

        found_diff = False
        for seed in range(60):
            random.seed(seed)
            _, tc2, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            for fp in fix_points:
                s2 = _slot_insts(tc2, fp.slot_id)
                s3 = _slot_insts(tc3, fp.slot_id)
                if s2 and s3:
                    if any(s2[i].name != s3[i].name or s2[i].template != s3[i].template
                           for i in range(SLOT_SIZE)):
                        found_diff = True
                        break
            if found_diff:
                break
        self.assertTrue(found_diff, "TC3 was always identical to TC2 for spec_nesting=None")

    def test_non_arch_signed_none_no_crash(self):
        """spec_nesting=None + signed_value=None: no crash, TC3 has AUTH."""
        result = _gen_tc_with_slots(30)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result
        for fp in fix_points:
            fp.signed_value = None
            fp.ctx_value    = None
            fp.spec_nesting = None

        random.seed(0)
        _, _, tc3 = pac.instrument_stage2(prep_tc, fix_points)
        for fp in fix_points:
            s3 = _slot_insts(tc3, fp.slot_id)
            if s3:
                self.assertIn(s3[AUTH_SLOT_POS].name, _AUTH_NAMES)
                # No restore at all — all non-auth positions must be NOP
                for i in range(SLOT_SIZE - 1):
                    self.assertEqual(s3[i].name, "nop",
                                     f"pos {i} should be NOP (no values captured), got {s3[i].name}")


# ===========================================================================
# 4. TestPacE2EValueRoundTrip
# ===========================================================================

class TestPacE2EValueRoundTrip(unittest.TestCase):
    """MOVZ+MOVK×3 encodes all 64-bit values losslessly."""

    def test_signed_value_roundtrip_many_tcs(self):
        rng = random.Random(99)
        checked = 0
        for seed in range(200):
            result = _gen_tc_with_slots(seed)
            if result is None:
                continue
            fix_points, prep_tc, pac = result
            _inject_values(fix_points, rng)
            tc1, tc2, _ = pac.instrument_stage2(prep_tc, fix_points)
            for fp in fix_points:
                for tc, label in ((tc1, "TC1"), (tc2, "TC2")):
                    s = _slot_insts(tc, fp.slot_id)
                    if s is None:
                        continue
                    encoded = _decode_value(s[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
                    with self.subTest(seed=seed, slot=fp.slot_id, variant=label):
                        self.assertEqual(encoded, fp.signed_value,
                                         f"signed_value round-trip failed: "
                                         f"expected {fp.signed_value:#018x}, got {encoded:#018x}")
                    checked += 1
            if checked >= 500:
                break
        self.assertGreater(checked, 0, "No slots checked")

    def test_ctx_value_roundtrip_many_tcs(self):
        rng = random.Random(77)
        checked = 0
        for seed in range(200):
            result = _gen_tc_with_slots(seed)
            if result is None:
                continue
            fix_points, prep_tc, pac = result
            _inject_values(fix_points, rng)
            _, tc2, _ = pac.instrument_stage2(prep_tc, fix_points)
            for fp in fix_points:
                if fp.info.ctx_reg is None:
                    continue
                s2 = _slot_insts(tc2, fp.slot_id)
                if s2 is None:
                    continue
                encoded = _decode_value(s2[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX])
                with self.subTest(seed=seed, slot=fp.slot_id):
                    self.assertEqual(encoded, fp.ctx_value)
                checked += 1
            if checked >= 200:
                break
        self.assertGreater(checked, 0, "No ctx slots checked")


# ===========================================================================
# 5. TestPacE2EArchSlotsCorrect
# ===========================================================================

class TestPacE2EArchSlotsCorrect(unittest.TestCase):
    """Arch slots: all three TCs use correct values; TC3 == TC2."""

    def test_tc3_equals_tc2_all_arch_many_tcs(self):
        rng = random.Random(55)
        mismatches = []
        tcs_checked = 0
        for seed in range(300):
            result = _gen_tc_with_slots(seed)
            if result is None:
                continue
            fix_points, prep_tc, pac = result
            _inject_values(fix_points, rng)  # all spec_nesting=0

            random.seed(seed)
            _, tc2, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            for fp in fix_points:
                s2 = _slot_insts(tc2, fp.slot_id)
                s3 = _slot_insts(tc3, fp.slot_id)
                if s2 is None or s3 is None:
                    continue
                for pos in range(SLOT_SIZE):
                    if s2[pos].name != s3[pos].name or s2[pos].template != s3[pos].template:
                        mismatches.append(
                            f"seed={seed} slot={fp.slot_id} pos={pos}: "
                            f"TC2={s2[pos].name!r} TC3={s3[pos].name!r}")
            tcs_checked += 1
            if tcs_checked >= 100:
                break

        self.assertEqual(mismatches, [],
                         f"TC3 ≠ TC2 for arch slots in {len(mismatches)} positions:\n" +
                         "\n".join(mismatches[:10]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
