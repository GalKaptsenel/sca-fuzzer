"""
Random execution tests verifying MTE instrumentation promises across many scenarios.

These tests generate TCs with randomly assigned spec_nesting ∈ {0, 1, None} and
randomly chosen sandbox_base values, then verify all MTE promises simultaneously.

Promises verified per slot:

  spec_nesting=0  (ARCH):
    TC1: NOP
    TC2: NOP  (correct tag preserved — sandbox setup already applied the right tag)
    TC3: NOP  (same, deterministic)

  spec_nesting≠0  (SPEC or NON-ARCH, spec_nesting=1 or None):
    TC1: NOP  (baseline — no tag manipulation)
    TC2: IRG Xd, Xd  (randomizes tag — non-interference candidate)
    TC3: MOVK Xd, #wrong_upper16, LSL #48  (deterministically wrong tag)

  wrong_upper16 formula:
    arch_tag      = (sandbox_base >> 56) & 0xF
    wrong_tag     = arch_tag ^ 1
    wrong_upper16 = (sandbox_upper16 & ~(0xF << 8)) | (wrong_tag << 8)

Reachability: every stage1 slot_id survives in TC1, TC2, TC3 after instrument_stage2.
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
    Aarch64RandomGenerator, MTEInstrumentation, MTEFixPoint, MTE_SLOT_SIZE,
)

_ISA: Optional[InstructionSet] = None
_TMPDIR = None

_SANDBOX_BASES = [
    0x0000_0000_4000_0000,  # arch_tag=0,  wrong_tag=1
    0x0100_0000_4000_0000,  # arch_tag=1,  wrong_tag=0
    0x0500_0000_4000_0000,  # arch_tag=5,  wrong_tag=4
    0x0700_BEEF_4000_0000,  # arch_tag=7,  non-zero lower bits
    0x0A00_0000_4000_0000,  # arch_tag=A,  wrong_tag=B
    0x0F00_0000_4000_0000,  # arch_tag=F,  wrong_tag=E
]


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


def _make_mte(gen: Aarch64RandomGenerator) -> MTEInstrumentation:
    return MTEInstrumentation(gen)


def _find_slot(tc, slot_id: int):
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                if getattr(inst, '_mte_slot_id', None) == slot_id:
                    return inst
    return None


def _all_slot_ids(tc) -> set:
    ids = set()
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                sid = getattr(inst, '_mte_slot_id', None)
                if sid is not None:
                    ids.add(sid)
    return ids


def _gen_tc_with_mem(seed: int):
    """Return (fix_points, prep_tc, mte) or None if TC has no memory accesses."""
    gen = _make_gen(seed)
    mte = _make_mte(gen)
    asm_path = os.path.join(_TMPDIR, f"mte_{seed}.asm")
    try:
        tc = gen.create_test_case(asm_path, disable_assembler=True)
        prep_tc, fix_points = mte.instrument_stage1(copy.deepcopy(tc))
    except Exception:
        return None
    if not fix_points:
        return None
    return fix_points, prep_tc, mte


def _expected_wrong_upper16(sandbox_base: int) -> int:
    sandbox_upper16 = (sandbox_base >> 48) & 0xFFFF
    arch_tag = (sandbox_base >> 56) & 0xF
    wrong_tag = arch_tag ^ 1
    return (sandbox_upper16 & ~(0xF << 8)) | (wrong_tag << 8)


def _parse_movk(template: str):
    m = re.match(r'MOVK\s+(\w+)\s*,\s*#0x([0-9a-fA-F]+)\s*,\s*LSL\s*#(\d+)',
                 template or "", re.IGNORECASE)
    return (m.group(1).lower(), int(m.group(2), 16), int(m.group(3))) if m else None


def _parse_irg(template: str):
    m = re.match(r'IRG\s+(\w+)\s*,\s*(\w+)', template or "", re.IGNORECASE)
    return (m.group(1).lower(), m.group(2).lower()) if m else None


# ===========================================================================
# 1. TestMteRandomPromises — all promises, random spec_nesting and sandbox_base
# ===========================================================================

class TestMteRandomPromises(unittest.TestCase):
    """
    Generates 100+ TCs with randomly assigned spec_nesting ∈ {0, 1, None} and
    randomly chosen sandbox_base values, then verifies every MTE promise.
    """

    _SCENARIOS = None  # list of (fix_points, tc1, tc2, tc3, sandbox_base, error|None)

    @classmethod
    def setUpClass(cls):
        cls._SCENARIOS = []
        rng = random.Random(54321)
        tc_count = 0

        for seed in range(500):
            result = _gen_tc_with_mem(seed)
            if result is None:
                continue
            fix_points, prep_tc, mte = result

            for fp in fix_points:
                fp.spec_nesting = [0, 1, None][rng.randint(0, 2)]

            sandbox_base = rng.choice(_SANDBOX_BASES)

            try:
                tc1, tc2, tc3 = mte.instrument_stage2(prep_tc, fix_points, sandbox_base)
            except Exception as e:
                cls._SCENARIOS.append((fix_points, None, None, None, sandbox_base, str(e)))
                continue

            cls._SCENARIOS.append((fix_points, tc1, tc2, tc3, sandbox_base, None))
            tc_count += 1
            if tc_count >= 100:
                break

        assert tc_count >= 20, f"Too few valid TCs: {tc_count}"

    def _iter_slots(self):
        """Yield (fp, i1, i2, i3, sandbox_base) for every valid slot."""
        for fix_points, tc1, tc2, tc3, sb, err in self._SCENARIOS:
            if tc1 is None:
                continue
            for fp in fix_points:
                i1 = _find_slot(tc1, fp.slot_id)
                i2 = _find_slot(tc2, fp.slot_id)
                i3 = _find_slot(tc3, fp.slot_id)
                if i1 and i2 and i3:
                    yield fp, i1, i2, i3, sb

    # ── No unexpected crashes ────────────────────────────────────────────────

    def test_instrument_stage2_never_crashes(self):
        errors = [e for _, _, _, _, _, e in self._SCENARIOS if e is not None]
        self.assertEqual(errors, [],
                         f"{len(errors)} unexpected crashes:\n" + "\n".join(errors[:5]))

    # ── Reachability ─────────────────────────────────────────────────────────

    def test_every_stage1_slot_appears_in_all_three_tcs(self):
        """Every slot_id from stage1 must exist in TC1, TC2, and TC3."""
        missing = []
        for fix_points, tc1, tc2, tc3, sb, err in self._SCENARIOS:
            if tc1 is None:
                continue
            for fp in fix_points:
                for label, tc in (("TC1", tc1), ("TC2", tc2), ("TC3", tc3)):
                    if _find_slot(tc, fp.slot_id) is None:
                        missing.append(f"slot={fp.slot_id} missing from {label} "
                                       f"(spec_nesting={fp.spec_nesting})")
        self.assertEqual(missing, [], "\n".join(missing[:10]))

    # ── TC1: always NOP ──────────────────────────────────────────────────────

    def test_tc1_always_nop_for_all_spec_nesting(self):
        """TC1 is NOP regardless of spec_nesting (0, 1, or None)."""
        for fp, i1, _, _, _ in self._iter_slots():
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(i1.name.lower(), 'nop')

    # ── TC2: arch→NOP, spec→IRG ──────────────────────────────────────────────

    def test_tc2_arch_is_nop(self):
        """spec_nesting=0: TC2 is NOP (correct tag preserved)."""
        for fp, _, i2, _, _ in self._iter_slots():
            if fp.spec_nesting != 0:
                continue
            with self.subTest(slot=fp.slot_id):
                self.assertEqual(i2.name.lower(), 'nop')

    def test_tc2_spec_is_irg(self):
        """spec_nesting≠0: TC2 is IRG (randomizes the tag)."""
        checked = 0
        for fp, _, i2, _, _ in self._iter_slots():
            if fp.spec_nesting == 0:
                continue
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(i2.name.lower(), 'irg')
            checked += 1
        self.assertGreater(checked, 0, "No spec slots encountered")

    def test_tc2_irg_same_src_and_dest_reg(self):
        """IRG Xd, Xd must use the same register for both src and dest."""
        for fp, _, i2, _, _ in self._iter_slots():
            if fp.spec_nesting == 0:
                continue
            parsed = _parse_irg(i2.template)
            with self.subTest(slot=fp.slot_id, reg=fp.reg):
                self.assertIsNotNone(parsed, f"IRG not parseable: {i2.template!r}")
                dest, src = parsed
                self.assertEqual(dest, fp.reg.lower())
                self.assertEqual(src, fp.reg.lower())

    # ── TC3: arch→NOP, spec→MOVK wrong_upper16 ───────────────────────────────

    def test_tc3_arch_is_nop(self):
        """spec_nesting=0: TC3 is NOP (correct tag preserved, deterministic)."""
        for fp, _, _, i3, _ in self._iter_slots():
            if fp.spec_nesting != 0:
                continue
            with self.subTest(slot=fp.slot_id):
                self.assertEqual(i3.name.lower(), 'nop')

    def test_tc3_spec_is_movk(self):
        """spec_nesting≠0: TC3 is MOVK with wrong_upper16."""
        checked = 0
        for fp, _, _, i3, _ in self._iter_slots():
            if fp.spec_nesting == 0:
                continue
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(i3.name.lower(), 'movk')
            checked += 1
        self.assertGreater(checked, 0, "No spec slots encountered")

    def test_tc3_movk_encodes_correct_wrong_upper16(self):
        """MOVK immediate must be wrong_upper16 = (sandbox_upper16 & ~tag_mask) | (wrong_tag<<8)."""
        for fp, _, _, i3, sb in self._iter_slots():
            if fp.spec_nesting == 0:
                continue
            expected_wrong = _expected_wrong_upper16(sb)
            parsed = _parse_movk(i3.template)
            with self.subTest(slot=fp.slot_id, sandbox=hex(sb), spec=fp.spec_nesting):
                self.assertIsNotNone(parsed, f"MOVK not parseable: {i3.template!r}")
                reg, imm, lsl = parsed
                self.assertEqual(reg, fp.reg.lower())
                self.assertEqual(lsl, 48,
                                 f"MOVK must shift into bits[63:48] — got LSL#{lsl}")
                self.assertEqual(imm, expected_wrong,
                                 f"wrong_upper16: expected 0x{expected_wrong:04x}, got 0x{imm:04x}")

    # ── spec_nesting=None treated as spec ────────────────────────────────────

    def test_spec_nesting_none_tc2_is_irg(self):
        """spec_nesting=None → TC2 = IRG (non-arch treated same as spec)."""
        checked = 0
        for fp, _, i2, _, _ in self._iter_slots():
            if fp.spec_nesting is not None:
                continue
            with self.subTest(slot=fp.slot_id):
                self.assertEqual(i2.name.lower(), 'irg')
            checked += 1
        self.assertGreater(checked, 0, "No spec_nesting=None slots encountered")

    def test_spec_nesting_none_tc3_is_movk(self):
        """spec_nesting=None → TC3 = MOVK (non-arch treated same as spec)."""
        checked = 0
        for fp, _, _, i3, _ in self._iter_slots():
            if fp.spec_nesting is not None:
                continue
            with self.subTest(slot=fp.slot_id):
                self.assertEqual(i3.name.lower(), 'movk')
            checked += 1
        self.assertGreater(checked, 0, "No spec_nesting=None slots encountered")

    # ── TC2 and TC3 are different for spec slots ─────────────────────────────

    def test_tc2_and_tc3_differ_for_spec_slots(self):
        """TC2=IRG (random tag) and TC3=MOVK (wrong tag) are always different instructions."""
        for fp, _, i2, i3, _ in self._iter_slots():
            if fp.spec_nesting == 0:
                continue
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertNotEqual(i2.name.lower(), i3.name.lower(),
                                    "TC2 and TC3 have same instruction name for spec slot")


# ===========================================================================
# 2. TestMteRandomReachability — stage1 slot tags survive into all three TCs
# ===========================================================================

class TestMteRandomReachability(unittest.TestCase):
    """
    Verifies that every stage1 NOP placeholder is reachable as TC1/TC2/TC3.
    """

    def test_stage1_slot_id_set_matches_fix_points(self):
        """Unique _mte_slot_ids in stage1 prep_tc == {fp.slot_id for fp in fix_points}."""
        for seed in range(100):
            result = _gen_tc_with_mem(seed)
            if result is None:
                continue
            fix_points, prep_tc, _ = result
            seen = _all_slot_ids(prep_tc)
            expected = {fp.slot_id for fp in fix_points}
            with self.subTest(seed=seed):
                self.assertEqual(seen, expected)


# ===========================================================================
# 3. TestMteRandomWrongTagFormula — wrong_upper16 formula for all sandbox bases
# ===========================================================================

class TestMteRandomWrongTagFormula(unittest.TestCase):
    """
    Verifies the wrong_upper16 formula against live MOVK output for all sandbox
    bases, across many seeds and spec_nesting values.
    """

    def test_wrong_upper16_correct_for_all_sandbox_bases_and_seeds(self):
        """For every (sandbox_base, seed) combination, MOVK imm == expected_wrong_upper16."""
        errors = []
        for sb in _SANDBOX_BASES:
            expected_wrong = _expected_wrong_upper16(sb)
            for seed in range(30):
                result = _gen_tc_with_mem(seed)
                if result is None:
                    continue
                fix_points, prep_tc, mte = result
                for fp in fix_points:
                    fp.spec_nesting = 1  # all spec so MOVK is generated
                tc1, tc2, tc3 = mte.instrument_stage2(prep_tc, fix_points, sb)
                for fp in fix_points:
                    inst = _find_slot(tc3, fp.slot_id)
                    if inst is None or inst.name.lower() != 'movk':
                        errors.append(f"seed={seed} sb={sb:#x}: TC3 not MOVK")
                        continue
                    p = _parse_movk(inst.template)
                    if p is None:
                        errors.append(f"seed={seed} sb={sb:#x}: MOVK not parseable")
                        continue
                    _, imm, lsl = p
                    if lsl != 48:
                        errors.append(f"seed={seed} sb={sb:#x}: LSL={lsl} (expected 48)")
                    if imm != expected_wrong:
                        errors.append(
                            f"seed={seed} sb={sb:#x}: imm=0x{imm:04x} "
                            f"expected=0x{expected_wrong:04x}")

        self.assertEqual(errors, [], f"{len(errors)} formula errors:\n" + "\n".join(errors[:15]))


# ===========================================================================
# 4. TestMteRandomAllCombinations
# ===========================================================================

class TestMteRandomAllCombinations(unittest.TestCase):

    def test_spec_nesting_none_same_behavior_as_spec_nesting_1(self):
        """spec_nesting=None and spec_nesting=1 must produce the same instructions."""
        errors = []
        for seed in range(30):
            result = _gen_tc_with_mem(seed)
            if result is None:
                continue
            fix_points, prep_tc, mte = result
            sb = _SANDBOX_BASES[seed % len(_SANDBOX_BASES)]

            # Run with spec_nesting=1
            for fp in fix_points:
                fp.spec_nesting = 1
            _, tc2_1, tc3_1 = mte.instrument_stage2(prep_tc, fix_points, sb)

            # Run with spec_nesting=None
            for fp in fix_points:
                fp.spec_nesting = None
            _, tc2_n, tc3_n = mte.instrument_stage2(prep_tc, fix_points, sb)

            for fp in fix_points:
                i2_1 = _find_slot(tc2_1, fp.slot_id)
                i2_n = _find_slot(tc2_n, fp.slot_id)
                i3_1 = _find_slot(tc3_1, fp.slot_id)
                i3_n = _find_slot(tc3_n, fp.slot_id)
                ctx = f"seed={seed} slot={fp.slot_id}"

                if i2_1 and i2_n and i2_1.name.lower() != i2_n.name.lower():
                    errors.append(f"{ctx}: TC2 spec=1 {i2_1.name!r} != spec=None {i2_n.name!r}")
                if i3_1 and i3_n and i3_1.template != i3_n.template:
                    errors.append(f"{ctx}: TC3 spec=1 {i3_1.template!r} != spec=None {i3_n.template!r}")

        self.assertEqual(errors, [], "\n".join(errors[:10]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
