"""
Random execution tests verifying PAC instrumentation promises across many scenarios.

These tests generate TCs with randomly assigned spec_nesting ∈ {0, 1, None} and
verify that all PAC promises hold for every slot type. Unlike unit_pac_e2e.py
(which uses fixed scenario assignments), each test here uses a fresh random
assignment to exercise all code paths simultaneously and find edge-case failures.

Promises verified per slot:

  spec_nesting=0  (ARCH):
    TC1: XPAC at pos8, signed_value encoded in [4:8], ctx_value in [0:4] (or NOPs)
    TC2: AUTH at pos8, signed_value encoded in [4:8], ctx_value in [0:4]
    TC3: Identical byte-for-byte to TC2

  spec_nesting=1  (SPEC, values captured):
    TC1: XPAC at pos8, signed_value in [4:8]
    TC2: AUTH at pos8, signed_value in [4:8], ctx_value in [0:4]
    TC3: AUTH at pos8, random (ctx,ptr) combo — may differ from TC2

  spec_nesting=None (NON-ARCH, CE never visited this signing):
    With values: same as spec_nesting=1 for TC1/TC2; TC3 uses random combo
    Without values: TC1/TC2 keep stage-1 XPAC untouched; TC3 has AUTH + NOPs only

Reachability: every stage1 slot_id appears in all three TCs after instrument_stage2.
"""
import copy
import os
import random
import re
import sys
import tempfile
import unittest
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import (
    Aarch64RandomGenerator, PACInstrumentation, FixPoint,
    SLOT_SIZE, FIX_COUNT_CTX, FIX_COUNT_PTR,
    CTX_SLOT_START, PTR_SLOT_START, AUTH_SLOT_POS,
    _PAC_TO_AUTH,
)

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
    """Collect all instructions for a slot, keyed by _pac_slot_pos. Returns None if incomplete."""
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
    """Decode a 64-bit value from MOVZ+MOVK×3 encoding."""
    val = 0
    for inst in insts:
        m_imm = re.search(r'#0x([0-9a-fA-F]+)', inst.template or "")
        m_lsl = re.search(r'LSL #(\d+)', inst.template or "")
        if not m_imm:
            return None
        val |= int(m_imm.group(1), 16) << (int(m_lsl.group(1)) if m_lsl else 0)
    return val


def _inject_values(fix_points: List[FixPoint], rng: random.Random):
    """Inject realistic fake CE-captured values (signed ptr in top 16 bits + canonical addr)."""
    for fp in fix_points:
        canonical = (rng.randint(0, 0xFFFF_FFFF_FFFF) & ~0xF) | 0x4000_0000_0000
        pac_sig   = rng.randint(0, 0xFFFF)
        fp.signed_value = canonical | (pac_sig << 48)
        if fp.info.ctx_reg is not None:
            fp.ctx_value = rng.randint(0, 0xFFFF_FFFF_FFFF_FFFF)


def _gen_tc_with_slots(seed: int):
    """Return (fix_points, prep_tc, pac) for a TC with ≥1 PAC slot, or None."""
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
# 1. TestPacRandomPromises — all promises, all spec_nesting values, random TCs
# ===========================================================================

class TestPacRandomPromises(unittest.TestCase):
    """
    Generates 100+ TCs with randomly assigned spec_nesting ∈ {0, 1, None} and
    verifies every promise for every slot simultaneously.
    """

    _SCENARIOS = None  # list of (fix_points, tc1, tc2, tc3, error_str|None)

    @classmethod
    def setUpClass(cls):
        cls._SCENARIOS = []
        rng = random.Random(12345)
        tc_count = 0

        for seed in range(500):
            result = _gen_tc_with_slots(seed)
            if result is None:
                continue
            fix_points, prep_tc, pac = result
            _inject_values(fix_points, rng)

            for fp in fix_points:
                fp.spec_nesting = [0, 1, None][rng.randint(0, 2)]

            try:
                random.seed(rng.randint(0, 10000))
                tc1, tc2, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            except Exception as e:
                cls._SCENARIOS.append((fix_points, None, None, None, str(e)))
                continue

            cls._SCENARIOS.append((fix_points, tc1, tc2, tc3, None))
            tc_count += 1
            if tc_count >= 100:
                break

        assert tc_count >= 20, f"Too few valid TCs: {tc_count}"

    def _iter_slots(self):
        """Yield (fp, s1, s2, s3) for every complete, valid slot."""
        for fix_points, tc1, tc2, tc3, err in self._SCENARIOS:
            if tc1 is None:
                continue
            for fp in fix_points:
                s1 = _slot_insts(tc1, fp.slot_id)
                s2 = _slot_insts(tc2, fp.slot_id)
                s3 = _slot_insts(tc3, fp.slot_id)
                if s1 and s2 and s3:
                    yield fp, s1, s2, s3

    # ── No unexpected crashes ────────────────────────────────────────────────

    def test_instrument_stage2_never_crashes(self):
        """instrument_stage2 must not raise for any spec_nesting / signed_value combination."""
        errors = [e for _, _, _, _, e in self._SCENARIOS if e is not None]
        self.assertEqual(errors, [],
                         f"{len(errors)} unexpected crashes:\n" + "\n".join(errors[:5]))

    # ── Reachability ─────────────────────────────────────────────────────────

    def test_every_stage1_slot_appears_in_all_three_tcs(self):
        """Every slot_id inserted in stage1 must exist in TC1, TC2, and TC3."""
        missing = []
        for fix_points, tc1, tc2, tc3, err in self._SCENARIOS:
            if tc1 is None:
                continue
            for fp in fix_points:
                for label, tc in (("TC1", tc1), ("TC2", tc2), ("TC3", tc3)):
                    if _slot_insts(tc, fp.slot_id) is None:
                        missing.append(f"slot={fp.slot_id} missing from {label} "
                                       f"(spec_nesting={fp.spec_nesting})")
        self.assertEqual(missing, [], "\n".join(missing[:10]))

    # ── TC1 promises ──────────────────────────────────────────────────────────

    def test_tc1_never_contains_auth(self):
        """TC1 must never have an AUTH instruction anywhere (only XPAC)."""
        violations = []
        for fp, s1, _, _ in self._iter_slots():
            if fp.signed_value is None:
                continue
            for pos, inst in enumerate(s1):
                if inst.name in _AUTH_NAMES:
                    violations.append(f"slot={fp.slot_id} pos={pos}: AUTH {inst.name!r} in TC1")
        self.assertEqual(violations, [], "\n".join(violations[:10]))

    def test_tc1_pos8_is_xpac_matching_variant_when_values_captured(self):
        """When signed_value is not None, TC1 pos8 is the XPAC matching the signing variant."""
        for fp, s1, _, _ in self._iter_slots():
            if fp.signed_value is None:
                continue
            expected = _SIGN_TO_XPAC.get(fp.info.pac_mnemonic.lower())
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting, pac=fp.info.pac_mnemonic):
                self.assertEqual(s1[AUTH_SLOT_POS].name, expected)

    def test_tc1_ptr_encodes_signed_value_when_captured(self):
        """TC1 ptr positions [4:8] encode signed_value losslessly for all spec_nesting."""
        for fp, s1, _, _ in self._iter_slots():
            if fp.signed_value is None:
                continue
            encoded = _decode_value(s1[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(encoded, fp.signed_value,
                                 f"TC1 ptr: expected {fp.signed_value:#018x}, got {encoded!r}")

    def test_tc1_ctx_encodes_ctx_value_when_present(self):
        """TC1 ctx positions [0:4] encode ctx_value when ctx_reg is not None."""
        for fp, s1, _, _ in self._iter_slots():
            if fp.signed_value is None or fp.info.ctx_reg is None or fp.ctx_value is None:
                continue
            encoded = _decode_value(s1[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX])
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(encoded, fp.ctx_value)

    # ── TC2 promises ──────────────────────────────────────────────────────────

    def test_tc2_pos8_is_auth_matching_variant_when_values_captured(self):
        """When signed_value is not None, TC2 pos8 is the correct AUTH variant."""
        for fp, _, s2, _ in self._iter_slots():
            if fp.signed_value is None:
                continue
            expected = _SIGN_TO_AUTH.get(fp.info.pac_mnemonic.lower())
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting, pac=fp.info.pac_mnemonic):
                self.assertEqual(s2[AUTH_SLOT_POS].name, expected)

    def test_tc2_ptr_encodes_signed_value_when_captured(self):
        """TC2 ptr positions always encode signed_value for all spec_nesting values."""
        for fp, _, s2, _ in self._iter_slots():
            if fp.signed_value is None:
                continue
            encoded = _decode_value(s2[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(encoded, fp.signed_value)

    def test_tc2_ctx_encodes_ctx_value_when_present(self):
        """TC2 ctx positions encode ctx_value for all spec_nesting values."""
        for fp, _, s2, _ in self._iter_slots():
            if fp.signed_value is None or fp.info.ctx_reg is None or fp.ctx_value is None:
                continue
            encoded = _decode_value(s2[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX])
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(encoded, fp.ctx_value)

    # ── TC3 arch promise: TC3 == TC2 ─────────────────────────────────────────

    def test_tc3_arch_equals_tc2_exactly(self):
        """spec_nesting=0 (arch): TC3 must be identical to TC2 at every position."""
        mismatches = []
        for fp, _, s2, s3 in self._iter_slots():
            if fp.spec_nesting != 0:
                continue
            for pos in range(SLOT_SIZE):
                if s2[pos].name != s3[pos].name or s2[pos].template != s3[pos].template:
                    mismatches.append(
                        f"slot={fp.slot_id} pos={pos}: "
                        f"TC2={s2[pos].name!r}/{s2[pos].template!r} "
                        f"TC3={s3[pos].name!r}/{s3[pos].template!r}")
        self.assertEqual(mismatches, [], "\n".join(mismatches[:10]))

    # ── TC3 spec promises ─────────────────────────────────────────────────────

    def test_tc3_spec_always_has_auth_at_pos8(self):
        """For any spec_nesting≠0, TC3 must have AUTH at pos8 regardless of combo."""
        for fp, _, _, s3 in self._iter_slots():
            if fp.spec_nesting == 0:
                continue
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertIn(s3[AUTH_SLOT_POS].name, _AUTH_NAMES,
                              f"TC3 pos8={s3[AUTH_SLOT_POS].name!r}, expected AUTH")

    def test_tc3_spec_auth_matches_signing_variant(self):
        """TC3 spec AUTH variant matches the signing instruction."""
        for fp, _, _, s3 in self._iter_slots():
            if fp.spec_nesting == 0:
                continue
            expected = _SIGN_TO_AUTH.get(fp.info.pac_mnemonic.lower())
            with self.subTest(slot=fp.slot_id):
                self.assertEqual(s3[AUTH_SLOT_POS].name, expected)

    def test_tc3_spec_ptr_restored_encodes_signed_value(self):
        """When TC3 spec includes a ptr restore (MOVZ at pos 4), it must encode signed_value."""
        for fp, _, _, s3 in self._iter_slots():
            if fp.spec_nesting == 0 or fp.signed_value is None:
                continue
            if s3[PTR_SLOT_START].name != "movz":
                continue  # ptr not restored in this combo
            encoded = _decode_value(s3[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(encoded, fp.signed_value,
                                 "TC3 ptr restore encodes wrong value")

    def test_tc3_spec_ctx_restored_encodes_ctx_value(self):
        """When TC3 spec includes a ctx restore (MOVZ at pos 0), it must encode ctx_value."""
        for fp, _, _, s3 in self._iter_slots():
            if fp.spec_nesting == 0 or fp.info.ctx_reg is None or fp.ctx_value is None:
                continue
            if s3[CTX_SLOT_START].name != "movz":
                continue  # ctx not restored in this combo
            encoded = _decode_value(s3[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX])
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(encoded, fp.ctx_value,
                                 "TC3 ctx restore encodes wrong value")

    # ── spec_nesting=None treated as spec ────────────────────────────────────

    def test_spec_nesting_none_tc3_treated_as_spec(self):
        """spec_nesting=None: TC3 must have AUTH at pos8 (same as spec_nesting=1)."""
        checked = 0
        for fp, _, _, s3 in self._iter_slots():
            if fp.spec_nesting is not None:
                continue
            with self.subTest(slot=fp.slot_id):
                self.assertIn(s3[AUTH_SLOT_POS].name, _AUTH_NAMES,
                              "spec_nesting=None: TC3 pos8 not AUTH")
            checked += 1
        # At least some None slots should appear given random assignment over 100 TCs
        self.assertGreater(checked, 0, "No spec_nesting=None slots encountered")

    def test_spec_nesting_none_tc2_has_auth_when_values_captured(self):
        """spec_nesting=None with values: TC2 must have AUTH (not keep the stage1 XPAC)."""
        for fp, _, s2, _ in self._iter_slots():
            if fp.spec_nesting is not None or fp.signed_value is None:
                continue
            expected = _SIGN_TO_AUTH.get(fp.info.pac_mnemonic.lower())
            with self.subTest(slot=fp.slot_id):
                self.assertEqual(s2[AUTH_SLOT_POS].name, expected)


# ===========================================================================
# 2. TestPacRandomReachability — stage1 slot tags survive into stage2
# ===========================================================================

class TestPacRandomReachability(unittest.TestCase):
    """
    Verifies that stage1 NOP placeholders are correctly wired to stage2 outputs.
    """

    def test_stage1_slot_id_set_matches_fix_points(self):
        """Unique slot_ids in stage1 prep_tc == {fp.slot_id for fp in fix_points}."""
        for seed in range(100):
            result = _gen_tc_with_slots(seed)
            if result is None:
                continue
            fix_points, prep_tc, _ = result

            seen_ids = set()
            for func in prep_tc.functions:
                for bb in func:
                    for inst in bb:
                        sid = getattr(inst, "_pac_slot_id", None)
                        if sid is not None and getattr(inst, "_pac_slot_pos", None) == 0:
                            seen_ids.add(sid)

            expected = {fp.slot_id for fp in fix_points}
            with self.subTest(seed=seed):
                self.assertEqual(seen_ids, expected)


# ===========================================================================
# 3. TestPacRandomTc3Discriminating — TC3 is statistically different from TC2
# ===========================================================================

class TestPacRandomTc3Discriminating(unittest.TestCase):
    """
    The key NI property: TC3 ≠ TC2 for spec slots (statistically), ensuring that
    different hardware speculation behaviors produce distinguishable outputs.
    """

    def test_tc3_differs_from_tc2_for_spec_slots_statistically(self):
        """Over 30 runs, TC3 must differ from TC2 in at least one spec slot."""
        rng = random.Random(777)
        found_diff = False

        for attempt in range(30):
            seed = rng.randint(0, 300)
            result = _gen_tc_with_slots(seed)
            if result is None:
                continue
            fix_points, prep_tc, pac = result
            _inject_values(fix_points, rng)
            for fp in fix_points:
                fp.spec_nesting = 1  # force all spec

            random.seed(attempt)
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

        self.assertTrue(found_diff,
                        "TC3 was always identical to TC2 for spec slots across 30 runs "
                        "— random combo selection is not working")

    def test_all_four_combos_seen_over_many_runs(self):
        """The 4 (ctx,ptr) combos all appear for spec slots with values across many seeds."""
        result = _gen_tc_with_slots(0)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result

        ctx_fp = next((fp for fp in fix_points if fp.info.ctx_reg is not None), None)
        if ctx_fp is None:
            self.skipTest("No slot with ctx_reg in seed=0 TC")

        rng = random.Random(42)
        _inject_values(fix_points, rng)
        for fp in fix_points:
            fp.spec_nesting = 1

        seen = set()
        for seed in range(300):
            random.seed(seed)
            _, _, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            s3 = _slot_insts(tc3, ctx_fp.slot_id)
            if s3:
                ctx_restored = s3[CTX_SLOT_START].name == "movz"
                ptr_restored  = s3[PTR_SLOT_START].name == "movz"
                seen.add((ctx_restored, ptr_restored))
            if len(seen) == 4:
                break

        self.assertEqual(seen,
                         {(False, False), (True, False), (False, True), (True, True)},
                         f"Only saw combos: {seen} after 300 seeds")


# ===========================================================================
# 4. TestPacRandomSignedValueNone — edge case: CE never captured values
# ===========================================================================

class TestPacRandomSignedValueNone(unittest.TestCase):
    """
    spec_nesting=None with signed_value=None means CE never visited the signing
    instruction. TC1/TC2 must keep the stage1 XPAC; TC3 must have AUTH with NOPs.
    """

    def test_signed_none_spec_none_no_crash(self):
        """No crash when signed_value=None and spec_nesting=None."""
        result = _gen_tc_with_slots(42)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result
        for fp in fix_points:
            fp.signed_value = None
            fp.ctx_value    = None
            fp.spec_nesting = None

        random.seed(0)
        tc1, tc2, tc3 = pac.instrument_stage2(prep_tc, fix_points)

        for fp in fix_points:
            s3 = _slot_insts(tc3, fp.slot_id)
            if s3:
                with self.subTest(slot=fp.slot_id):
                    self.assertIn(s3[AUTH_SLOT_POS].name, _AUTH_NAMES,
                                  "TC3 pos8 must be AUTH even with no values captured")
                    for i in range(SLOT_SIZE - 1):
                        self.assertEqual(s3[i].name, "nop",
                                         f"TC3 pos {i} should be NOP (no values), got {s3[i].name}")

    def test_signed_none_spec_none_tc3_has_auth_across_many_tcs(self):
        """TC3 always has AUTH at pos8 when both values are None and spec_nesting=None."""
        rng = random.Random(55)
        found = 0

        for seed in range(100):
            result = _gen_tc_with_slots(seed)
            if result is None:
                continue
            fix_points, prep_tc, pac = result
            for fp in fix_points:
                fp.signed_value = None
                fp.ctx_value    = None
                fp.spec_nesting = None

            random.seed(seed)
            _, _, tc3 = pac.instrument_stage2(prep_tc, fix_points)

            for fp in fix_points:
                s3 = _slot_insts(tc3, fp.slot_id)
                if s3:
                    with self.subTest(seed=seed, slot=fp.slot_id):
                        self.assertIn(s3[AUTH_SLOT_POS].name, _AUTH_NAMES)
                    found += 1
            if found >= 30:
                break

        self.assertGreater(found, 0, "No slots checked")

    def test_signed_none_spec_none_ptr_positions_all_nop(self):
        """With no values captured, TC3 ptr positions [4:7] must all be NOP."""
        result = _gen_tc_with_slots(10)
        self.assertIsNotNone(result)
        fix_points, prep_tc, pac = result
        for fp in fix_points:
            fp.signed_value = None
            fp.ctx_value    = None
            fp.spec_nesting = None

        random.seed(1)
        _, _, tc3 = pac.instrument_stage2(prep_tc, fix_points)

        for fp in fix_points:
            s3 = _slot_insts(tc3, fp.slot_id)
            if s3:
                with self.subTest(slot=fp.slot_id):
                    for i in range(FIX_COUNT_PTR):
                        self.assertEqual(s3[PTR_SLOT_START + i].name, "nop",
                                         f"ptr pos {i} should be NOP, got {s3[PTR_SLOT_START+i].name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
