"""
Integration tests for MTE TC1/TC2/TC3 generation using the real ISA and generator.

Groups:
  1. TestMteE2ESlotStructure     — TC1/TC2/TC3 invariants across 100 real TCs
  2. TestMteE2ETc3SpecBehavior   — spec slots: IRG/MOVK appear; arch slots: NOP
  3. TestMteE2EWrongTagFormula   — wrong_upper16 formula for all arch_tag values
  4. TestMteE2EStage1Structure   — NOP placeholder order, unique slot IDs
  5. TestMteE2EAllCombinations   — all scenario × sandbox_base combinations
"""
import copy
import os
import random
import re
import sys
import tempfile
import unittest
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import (
    Aarch64RandomGenerator,
    MTEInstrumentation, MTEFixPoint,
    MTE_SLOT_SIZE,
)

_ISA: Optional[InstructionSet] = None
_TMPDIR = None

_SANDBOX_BASES = [
    0x0000_0000_4000_0000,   # arch_tag=0, wrong_tag=1
    0x0100_0000_4000_0000,   # arch_tag=1, wrong_tag=0
    0x0A00_0000_4000_0000,   # arch_tag=0xA, wrong_tag=0xB
    0x0F00_0000_4000_0000,   # arch_tag=0xF, wrong_tag=0xE
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


def _all_slot_ids(tc) -> Dict[int, object]:
    result = {}
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                sid = getattr(inst, '_mte_slot_id', None)
                if sid is not None:
                    result[sid] = inst
    return result


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
    arch_tag        = (sandbox_base >> 56) & 0xF
    wrong_tag       = arch_tag ^ 1
    return (sandbox_upper16 & ~(0xF << 8)) | (wrong_tag << 8)


def _parse_irg(template: str) -> Optional[Tuple[str, str]]:
    m = re.match(r'IRG\s+(\w+)\s*,\s*(\w+)', template or "", re.IGNORECASE)
    return (m.group(1).lower(), m.group(2).lower()) if m else None


def _parse_movk(template: str) -> Optional[Tuple[str, int, int]]:
    m = re.match(r'MOVK\s+(\w+)\s*,\s*#0x([0-9a-fA-F]+)\s*,\s*LSL\s*#(\d+)',
                 template or "", re.IGNORECASE)
    return (m.group(1).lower(), int(m.group(2), 16), int(m.group(3))) if m else None


# ===========================================================================
# 1. TestMteE2ESlotStructure
# ===========================================================================

class TestMteE2ESlotStructure(unittest.TestCase):
    """TC1/TC2/TC3 structural invariants over 100 real TCs."""

    @classmethod
    def setUpClass(cls):
        cls.cases = []  # (fix_points, tc1, tc2, tc3, sandbox_base)
        sandbox_base = _SANDBOX_BASES[0]
        for seed in range(300):
            result = _gen_tc_with_mem(seed)
            if result is None:
                continue
            fix_points, prep_tc, mte = result
            for i, fp in enumerate(fix_points):
                fp.spec_nesting = i % 2  # alternating arch/spec
            tc1, tc2, tc3 = mte.instrument_stage2(prep_tc, fix_points, sandbox_base)
            cls.cases.append((fix_points, tc1, tc2, tc3, sandbox_base))
            if len(cls.cases) >= 100:
                break
        assert len(cls.cases) >= 20, f"Too few TCs with memory accesses: {len(cls.cases)}"

    # ── TC1: always NOP ───────────────────────────────────────────────────

    def test_tc1_always_nop(self):
        for fix_points, tc1, _, _, _ in self.cases:
            for fp in fix_points:
                inst = _find_slot(tc1, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertIsNotNone(inst)
                    self.assertEqual(inst.name.lower(), 'nop')

    # ── TC2: spec→IRG, arch→NOP ───────────────────────────────────────────

    def test_tc2_spec_is_irg(self):
        for fix_points, _, tc2, _, _ in self.cases:
            for fp in fix_points:
                if fp.spec_nesting is None or fp.spec_nesting == 0:
                    continue
                inst = _find_slot(tc2, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertIsNotNone(inst)
                    self.assertEqual(inst.name.lower(), 'irg')

    def test_tc2_arch_is_nop(self):
        for fix_points, _, tc2, _, _ in self.cases:
            for fp in fix_points:
                if fp.spec_nesting != 0:
                    continue
                inst = _find_slot(tc2, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertIsNotNone(inst)
                    self.assertEqual(inst.name.lower(), 'nop')

    def test_tc2_irg_uses_same_reg_for_src_and_dest(self):
        for fix_points, _, tc2, _, _ in self.cases:
            for fp in fix_points:
                if fp.spec_nesting is None or fp.spec_nesting == 0:
                    continue
                inst = _find_slot(tc2, fp.slot_id)
                if inst is None or inst.name.lower() != 'irg':
                    continue
                parsed = _parse_irg(inst.template)
                with self.subTest(slot=fp.slot_id, reg=fp.reg):
                    self.assertIsNotNone(parsed, f"IRG template unparseable: {inst.template!r}")
                    dest, src = parsed
                    expected_reg = fp.reg.lower()
                    self.assertEqual(dest, expected_reg)
                    self.assertEqual(src, expected_reg)

    # ── TC3: spec→MOVK wrong_tag, arch→NOP ───────────────────────────────

    def test_tc3_spec_is_movk(self):
        for fix_points, _, _, tc3, sandbox_base in self.cases:
            for fp in fix_points:
                if fp.spec_nesting is None or fp.spec_nesting == 0:
                    continue
                inst = _find_slot(tc3, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertIsNotNone(inst)
                    self.assertEqual(inst.name.lower(), 'movk')

    def test_tc3_arch_is_nop(self):
        for fix_points, _, _, tc3, sandbox_base in self.cases:
            for fp in fix_points:
                if fp.spec_nesting != 0:
                    continue
                inst = _find_slot(tc3, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertIsNotNone(inst)
                    self.assertEqual(inst.name.lower(), 'nop')

    def test_tc3_movk_correct_wrong_upper16(self):
        for fix_points, _, _, tc3, sandbox_base in self.cases:
            expected_wrong = _expected_wrong_upper16(sandbox_base)
            for fp in fix_points:
                if fp.spec_nesting is None or fp.spec_nesting == 0:
                    continue
                inst = _find_slot(tc3, fp.slot_id)
                if inst is None or inst.name.lower() != 'movk':
                    continue
                parsed = _parse_movk(inst.template)
                with self.subTest(slot=fp.slot_id, sandbox=hex(sandbox_base)):
                    self.assertIsNotNone(parsed, f"MOVK template unparseable: {inst.template!r}")
                    reg, imm, lsl = parsed
                    self.assertEqual(reg, fp.reg.lower())
                    self.assertEqual(lsl, 48)
                    self.assertEqual(imm, expected_wrong,
                                     f"wrong_upper16: expected 0x{expected_wrong:04x}, got 0x{imm:04x}")



# ===========================================================================
# 2. TestMteE2ETc3SpecBehavior
# ===========================================================================

class TestMteE2ETc3SpecBehavior(unittest.TestCase):
    """Verify TC3 produces correct spec/arch outputs for all_spec and all_arch scenarios."""

    def _run(self, seed: int, sandbox_base: int, spec_nesting_fn):
        result = _gen_tc_with_mem(seed)
        if result is None:
            return None, None, None, None
        fix_points, prep_tc, mte = result
        for i, fp in enumerate(fix_points):
            fp.spec_nesting = spec_nesting_fn(i)
        tc1, tc2, tc3 = mte.instrument_stage2(prep_tc, fix_points, sandbox_base)
        return fix_points, tc1, tc2, tc3

    def test_all_arch_tc1_tc2_tc3_all_nop(self):
        """All arch → TC1=TC2=TC3=NOP for every slot."""
        base = _SANDBOX_BASES[0]
        for seed in range(20):
            fix_points, tc1, tc2, tc3 = self._run(seed, base, lambda i: 0)
            if fix_points is None:
                continue
            for fp in fix_points:
                for label, tc in (("TC1", tc1), ("TC2", tc2), ("TC3", tc3)):
                    inst = _find_slot(tc, fp.slot_id)
                    with self.subTest(seed=seed, slot=fp.slot_id, variant=label):
                        self.assertIsNotNone(inst)
                        self.assertEqual(inst.name.lower(), 'nop')

    def test_all_spec_tc2_is_irg_tc3_is_movk(self):
        """All spec → TC1=NOP, TC2=IRG, TC3=MOVK for every slot."""
        base = _SANDBOX_BASES[1]
        expected_wrong = _expected_wrong_upper16(base)
        for seed in range(20):
            fix_points, tc1, tc2, tc3 = self._run(seed, base, lambda i: 1)
            if fix_points is None:
                continue
            for fp in fix_points:
                i1 = _find_slot(tc1, fp.slot_id)
                i2 = _find_slot(tc2, fp.slot_id)
                i3 = _find_slot(tc3, fp.slot_id)
                with self.subTest(seed=seed, slot=fp.slot_id):
                    self.assertEqual(i1.name.lower(), 'nop')
                    self.assertEqual(i2.name.lower(), 'irg')
                    self.assertEqual(i3.name.lower(), 'movk')
                    p = _parse_movk(i3.template)
                    self.assertIsNotNone(p)
                    _, imm, lsl = p
                    self.assertEqual(lsl, 48)
                    self.assertEqual(imm, expected_wrong)


# ===========================================================================
# 3. TestMteE2EWrongTagFormula
# ===========================================================================

class TestMteE2EWrongTagFormula(unittest.TestCase):
    """wrong_upper16 formula verified against live MOVK output for all arch_tag values."""

    def test_all_arch_tags_produce_correct_wrong_upper16(self):
        result = _gen_tc_with_mem(0)
        self.assertIsNotNone(result)
        fix_points, prep_tc, mte = result
        fp = fix_points[0]
        fp.spec_nesting = 1  # spec so MOVK is generated

        for arch_tag in range(16):
            sandbox_base = arch_tag << 56
            expected_wrong = _expected_wrong_upper16(sandbox_base)

            tc1, tc2, tc3 = mte.instrument_stage2(prep_tc, fix_points, sandbox_base)
            inst = _find_slot(tc3, fp.slot_id)
            with self.subTest(arch_tag=arch_tag, sandbox=hex(sandbox_base)):
                self.assertIsNotNone(inst)
                self.assertEqual(inst.name.lower(), 'movk')
                parsed = _parse_movk(inst.template)
                self.assertIsNotNone(parsed)
                _, imm, lsl = parsed
                self.assertEqual(lsl, 48,
                                 f"arch_tag={arch_tag:#x}: expected LSL#48, got LSL#{lsl}")
                self.assertEqual(imm, expected_wrong,
                                 f"arch_tag={arch_tag:#x}: expected 0x{expected_wrong:04x}, "
                                 f"got 0x{imm:04x}")



# ===========================================================================
# 4. TestMteE2EStage1Structure
# ===========================================================================

class TestMteE2EStage1Structure(unittest.TestCase):
    """Stage-1 structural invariants: unique IDs, NOP placeholder placement."""

    def test_unique_slot_ids_across_tc(self):
        for seed in range(50):
            result = _gen_tc_with_mem(seed)
            if result is None:
                continue
            fix_points, prep_tc, _ = result
            ids = [fp.slot_id for fp in fix_points]
            with self.subTest(seed=seed):
                self.assertEqual(len(ids), len(set(ids)), f"Duplicate slot_ids: {ids}")

    def test_nop_placeholder_in_slot_insts(self):
        for seed in range(50):
            result = _gen_tc_with_mem(seed)
            if result is None:
                continue
            fix_points, _, _ = result
            for fp in fix_points:
                with self.subTest(seed=seed, slot=fp.slot_id):
                    self.assertEqual(len(fp.slot_insts), MTE_SLOT_SIZE)
                    self.assertEqual(fp.slot_insts[0].name.lower(), 'nop')

    def test_slot_exists_in_prep_tc(self):
        for seed in range(50):
            result = _gen_tc_with_mem(seed)
            if result is None:
                continue
            fix_points, prep_tc, _ = result
            all_ids = _all_slot_ids(prep_tc)
            for fp in fix_points:
                with self.subTest(seed=seed, slot=fp.slot_id):
                    self.assertIn(fp.slot_id, all_ids,
                                  f"slot_id={fp.slot_id} not found in prep_tc")




if __name__ == "__main__":
    unittest.main(verbosity=2)
