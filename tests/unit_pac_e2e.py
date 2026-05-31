"""
PAC stage-1 and stage-2 end-to-end tests using the real ISA and generator.

Groups:
  1. TestPacE2EStage1Structure   — stage-1: slot layout, committed_inst, XPAC family, unique IDs
  2. TestPacE2EStage2Structure   — stage-2: TC1/TC2/TC3 invariants using real fix_points
  3. TestPacE2ENoOpOptimisation  — ptr-match → NOP at 4-7; ctx-match → NOP at 0-3
  4. TestPacE2EPipeline          — key-family consistency across all three variants
"""
import copy
import os
import sys
import tempfile
import unittest
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import (
    Aarch64RandomGenerator, PACInstrumentation,
    Aarch64Printer, Aarch64ASMLayout, Aarch64SandboxPass,
    PACVariant, FixPoint, PoolEntry, PACKey,
    _AUTH_TO_KEY, _AUTH_TO_XPAC, _PAC_INFO,
    SLOT_SIZE, AUTH_SLOT_POS, SLOT_SIG_POS,
    CTX_SLOT_START, PTR_SLOT_START, POST_AUTH_CTX_START,
    FIX_COUNT_CTX, FIX_COUNT_PTR,
)
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc
from src.generator import ConfigurableGenerator
from src.aarch64.aarch64_executor import pass_on_test_case

_ISA: Optional[InstructionSet] = None
_TMPDIR: Optional[str] = None

_AUTH_NAMES = frozenset(_AUTH_TO_KEY.keys())
_XPAC_NAMES = frozenset({'xpaci', 'xpacd'})

_SIGNED_VAL = 0xDEAD_BEEF_CAFE_1234
_CTX_VAL    = 0x0000_ABCD_1234_5678
_PRE_PTR    = 0x1111_2222_3333_4444  # guaranteed != _SIGNED_VAL


def setUpModule():
    global _ISA, _TMPDIR
    if not os.path.exists('/dev/executor'):
        raise unittest.SkipTest(
            "kernel module not loaded — run "
            "'sudo insmod revizor-executor.ko && sudo chmod 777 /dev/executor' "
            "to run these tests")
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
    return PACInstrumentation(gen, xpac_weight=1, auth_weight=1, sign_weight=1)


def _gen_stage1(seed: int):
    """Return (fix_points, stage1_tc, pac) or None if no slots were generated."""
    gen = _make_gen(seed)
    pac = _make_pac(gen)
    asm_path = os.path.join(_TMPDIR, f"pac_{seed}.asm")
    try:
        tc = gen.create_test_case(asm_path, disable_assembler=True)
        stage1_tc, fix_points = pac.instrument_stage1(copy.deepcopy(tc))
    except Exception:
        return None
    if not fix_points:
        return None
    return fix_points, stage1_tc, pac


def _slot_insts(tc, slot_id: int) -> Dict[int, object]:
    """Return {pos: Instruction} for every tagged instruction in the given slot."""
    result = {}
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                if getattr(inst, '_pac_slot_id', None) == slot_id:
                    result[inst._pac_slot_pos] = inst
    return result


def _arch_entry(key: PACKey, signed: int = _SIGNED_VAL, ctx: int = _CTX_VAL) -> PoolEntry:
    return PoolEntry(signed_value=signed, ctx_value=ctx, key=key, is_arch=True)


def _setup_fp_for_stage2(fp: FixPoint, spec_nesting: int, want_loads: bool = True) -> PoolEntry:
    """Fill fp's executor-side fields; return the chosen entry."""
    key   = _AUTH_TO_KEY[fp.committed_inst.name.lower()]
    entry = _arch_entry(key)
    fp.chosen_pool_entry      = entry
    fp.spec_nesting           = spec_nesting
    fp.pre_slot_ptr_reg_value = _PRE_PTR if want_loads else _SIGNED_VAL
    if len(fp.committed_inst.operands) > 1:
        fp.pre_slot_ctx_reg_value = _CTX_VAL + 1 if want_loads else _CTX_VAL
    else:
        fp.pre_slot_ctx_reg_value = None
    return entry


def _collect_cases(n: int = 50):
    """Generate up to n stage-1 results, trying seeds 0..499."""
    cases = []
    for seed in range(500):
        result = _gen_stage1(seed)
        if result is not None:
            cases.append(result)
            if len(cases) >= n:
                break
    return cases


# ===========================================================================
# 1. Stage-1 structural invariants
# ===========================================================================

class TestPacE2EStage1Structure(unittest.TestCase):
    """Stage-1 output must be well-formed: slot size, XPAC position, committed_inst."""

    @classmethod
    def setUpClass(cls):
        cls.cases = _collect_cases(50)
        assert len(cls.cases) >= 20, f"Too few PAC TCs: {len(cls.cases)}"

    def test_slot_ids_unique(self):
        """Each TC has no duplicate slot_ids across its fix_points."""
        for fix_points, _, _ in self.cases:
            ids = [fp.slot_id for fp in fix_points]
            self.assertEqual(len(ids), len(set(ids)))

    def test_every_slot_has_slot_size_instructions(self):
        """Every slot in the stage-1 TC has exactly SLOT_SIZE tagged instructions."""
        for fix_points, stage1_tc, _ in self.cases:
            for fp in fix_points:
                pos_map = _slot_insts(stage1_tc, fp.slot_id)
                with self.subTest(slot_id=fp.slot_id):
                    self.assertEqual(len(pos_map), SLOT_SIZE)
                    for p in range(SLOT_SIZE):
                        self.assertIn(p, pos_map, f"position {p} missing in slot {fp.slot_id}")

    def test_auth_slot_pos_has_xpac(self):
        """Position AUTH_SLOT_POS in every stage-1 slot holds an XPAC placeholder."""
        for fix_points, stage1_tc, _ in self.cases:
            for fp in fix_points:
                pos_map = _slot_insts(stage1_tc, fp.slot_id)
                inst    = pos_map.get(AUTH_SLOT_POS)
                with self.subTest(slot_id=fp.slot_id):
                    self.assertIsNotNone(inst)
                    self.assertIn(inst.name.lower(), _XPAC_NAMES)

    def test_all_non_auth_positions_are_nop(self):
        """All stage-1 slot positions except AUTH_SLOT_POS are NOP."""
        for fix_points, stage1_tc, _ in self.cases:
            for fp in fix_points:
                pos_map = _slot_insts(stage1_tc, fp.slot_id)
                for pos, inst in pos_map.items():
                    if pos == AUTH_SLOT_POS:
                        continue
                    with self.subTest(slot_id=fp.slot_id, pos=pos):
                        self.assertEqual(inst.name.lower(), 'nop')

    def test_committed_inst_is_auth_mnemonic(self):
        """committed_inst on every FixPoint is an AUT* instruction."""
        for fix_points, _, _ in self.cases:
            for fp in fix_points:
                with self.subTest(slot_id=fp.slot_id):
                    self.assertIsNotNone(fp.committed_inst)
                    self.assertIn(fp.committed_inst.name.lower(), _AUTH_NAMES)

    def test_xpac_family_matches_committed_inst(self):
        """Stage-1 XPAC mnemonic matches the key family of committed_inst (IA→xpaci, DA→xpacd)."""
        for fix_points, stage1_tc, _ in self.cases:
            for fp in fix_points:
                pos_map      = _slot_insts(stage1_tc, fp.slot_id)
                xpac_inst    = pos_map[AUTH_SLOT_POS]
                expected_xpac = _AUTH_TO_XPAC[fp.committed_inst.name.lower()]
                with self.subTest(slot_id=fp.slot_id):
                    self.assertEqual(xpac_inst.name.lower(), expected_xpac)

    def test_xpac_reg_matches_committed_ptr_reg(self):
        """XPAC operates on the same register as committed_inst's ptr_reg (operands[0])."""
        for fix_points, stage1_tc, _ in self.cases:
            for fp in fix_points:
                pos_map      = _slot_insts(stage1_tc, fp.slot_id)
                xpac_inst    = pos_map[AUTH_SLOT_POS]
                expected_reg = fp.committed_inst.operands[0].value
                with self.subTest(slot_id=fp.slot_id):
                    self.assertEqual(xpac_inst.operands[0].value, expected_reg)


# ===========================================================================
# 2. Stage-2 structural invariants
# ===========================================================================

class TestPacE2EStage2Structure(unittest.TestCase):
    """TC1/TC2/TC3 must satisfy structural invariants when produced from real fix_points."""

    @classmethod
    def setUpClass(cls):
        raw = _collect_cases(50)
        assert len(raw) >= 20, f"Too few PAC TCs: {len(raw)}"
        cls.cases = []  # (fix_points, variants)
        for fix_points, stage1_tc, pac in raw:
            pool: Dict[PACKey, List[PoolEntry]] = {}
            for i, fp in enumerate(fix_points):
                entry = _setup_fp_for_stage2(fp, spec_nesting=i % 2, want_loads=True)
                pool.setdefault(entry.key, []).append(entry)
            variants = pac.instrument_stage2(stage1_tc, fix_points, pool)
            cls.cases.append((fix_points, variants))

    def _at(self, tc, slot_id: int, pos: int):
        return _slot_insts(tc, slot_id).get(pos)

    def test_tc1_auth_pos_has_xpac(self):
        """TC1 (STRIP_ONLY): AUTH_SLOT_POS always holds XPAC."""
        for fix_points, variants in self.cases:
            tc1 = variants[PACVariant.STRIP_ONLY]
            for fp in fix_points:
                inst = self._at(tc1, fp.slot_id, AUTH_SLOT_POS)
                with self.subTest(slot_id=fp.slot_id):
                    self.assertIsNotNone(inst)
                    self.assertIn(inst.name.lower(), _XPAC_NAMES)

    def test_tc2_auth_pos_has_auth(self):
        """TC2 (AUTH_CORRECT): AUTH_SLOT_POS always holds AUT*."""
        for fix_points, variants in self.cases:
            tc2 = variants[PACVariant.AUTH_CORRECT]
            for fp in fix_points:
                inst = self._at(tc2, fp.slot_id, AUTH_SLOT_POS)
                with self.subTest(slot_id=fp.slot_id):
                    self.assertIsNotNone(inst)
                    self.assertIn(inst.name.lower(), _AUTH_NAMES)

    def test_tc3_auth_pos_has_auth(self):
        """TC3 (AUTH_WRONG): AUTH_SLOT_POS always holds AUT* (arch or spec)."""
        for fix_points, variants in self.cases:
            tc3 = variants[PACVariant.AUTH_WRONG]
            for fp in fix_points:
                inst = self._at(tc3, fp.slot_id, AUTH_SLOT_POS)
                with self.subTest(slot_id=fp.slot_id):
                    self.assertIsNotNone(inst)
                    self.assertIn(inst.name.lower(), _AUTH_NAMES)

    def test_tc1_sig_pos_has_movk_when_sig_known(self):
        """TC1 (STRIP_ONLY): SLOT_SIG_POS holds MOVK when correct_sig is known, else NOP."""
        for fix_points, variants in self.cases:
            tc1 = variants[PACVariant.STRIP_ONLY]
            for fp in fix_points:
                inst = self._at(tc1, fp.slot_id, SLOT_SIG_POS)
                with self.subTest(slot_id=fp.slot_id, correct_sig=fp.correct_sig):
                    self.assertIsNotNone(inst)
                    if fp.correct_sig is not None:
                        self.assertEqual(inst.name.lower(), "movk")
                    else:
                        self.assertEqual(inst.name.lower(), "nop")

    def test_tc1_xpac_family_matches_committed(self):
        """TC1 XPAC mnemonic at AUTH_SLOT_POS matches committed_inst's key family."""
        for fix_points, variants in self.cases:
            tc1 = variants[PACVariant.STRIP_ONLY]
            for fp in fix_points:
                inst         = self._at(tc1, fp.slot_id, AUTH_SLOT_POS)
                expected_xpac = _AUTH_TO_XPAC[fp.committed_inst.name.lower()]
                with self.subTest(slot_id=fp.slot_id):
                    self.assertEqual(inst.name.lower(), expected_xpac)

    def test_tc2_auth_mnemonic_matches_committed(self):
        """TC2 AUT* mnemonic at AUTH_SLOT_POS matches committed_inst exactly."""
        for fix_points, variants in self.cases:
            tc2 = variants[PACVariant.AUTH_CORRECT]
            for fp in fix_points:
                inst = self._at(tc2, fp.slot_id, AUTH_SLOT_POS)
                with self.subTest(slot_id=fp.slot_id):
                    self.assertEqual(inst.name.lower(), fp.committed_inst.name.lower())

    def test_arch_tc3_equals_tc2_at_all_positions(self):
        """Arch slots (spec_nesting=0): every position in TC3 matches TC2."""
        for fix_points, variants in self.cases:
            tc2 = variants[PACVariant.AUTH_CORRECT]
            tc3 = variants[PACVariant.AUTH_WRONG]
            for fp in fix_points:
                if fp.spec_nesting != 0:
                    continue
                m2 = _slot_insts(tc2, fp.slot_id)
                m3 = _slot_insts(tc3, fp.slot_id)
                for pos in range(SLOT_SIZE):
                    with self.subTest(slot_id=fp.slot_id, pos=pos):
                        self.assertEqual(m2[pos].name,     m3[pos].name)
                        self.assertEqual(m2[pos].template, m3[pos].template)

    def test_spec_tc3_auth_pos_is_auth(self):
        """Spec slots (spec_nesting!=0): TC3 AUTH_SLOT_POS still holds AUT*."""
        for fix_points, variants in self.cases:
            tc3 = variants[PACVariant.AUTH_WRONG]
            for fp in fix_points:
                if fp.spec_nesting == 0:
                    continue
                inst = self._at(tc3, fp.slot_id, AUTH_SLOT_POS)
                with self.subTest(slot_id=fp.slot_id):
                    self.assertIn(inst.name.lower(), _AUTH_NAMES)

    def test_zero_ctx_variants_have_nop_at_all_ctx_positions(self):
        """Zero-context AUT* (autiza/autizb/autdza/autdzb): ctx positions 0-3 and 9-12 are NOP."""
        for fix_points, variants in self.cases:
            tc2 = variants[PACVariant.AUTH_CORRECT]
            for fp in fix_points:
                if len(fp.committed_inst.operands) > 1:
                    continue  # has a ctx register — skip
                m2 = _slot_insts(tc2, fp.slot_id)
                for pos in list(range(CTX_SLOT_START, CTX_SLOT_START + FIX_COUNT_CTX)) + \
                           list(range(POST_AUTH_CTX_START, POST_AUTH_CTX_START + FIX_COUNT_CTX)):
                    with self.subTest(slot_id=fp.slot_id, pos=pos):
                        self.assertEqual(m2[pos].name.lower(), 'nop')


# ===========================================================================
# 3. No-op optimisation: register already holds the correct value
# ===========================================================================

class TestPacE2ENoOpOptimisation(unittest.TestCase):
    """When pre-slot register value == pool entry value, the load sequence must be NOPs."""

    def _get_stage1_with_fp(self) -> tuple:
        """Return (fp, stage1_tc, pac) for the first seed that produces a fix_point."""
        for seed in range(500):
            result = _gen_stage1(seed)
            if result is not None:
                fix_points, stage1_tc, pac = result
                return fix_points[0], stage1_tc, pac
        self.fail("No seeds produced any fix_points")

    def test_ptr_match_gives_nop_at_ptr_positions(self):
        """pre_slot_ptr == entry.signed_value → positions 4-7 are NOP in TC2."""
        fp, stage1_tc, pac = self._get_stage1_with_fp()
        key   = _AUTH_TO_KEY[fp.committed_inst.name.lower()]
        entry = _arch_entry(key, signed=_SIGNED_VAL, ctx=_CTX_VAL)
        fp.chosen_pool_entry      = entry
        fp.pre_slot_ptr_reg_value = _SIGNED_VAL   # match
        fp.pre_slot_ctx_reg_value = _CTX_VAL + 1 if len(fp.committed_inst.operands) > 1 else None
        fp.spec_nesting           = 0
        variants = pac.instrument_stage2(stage1_tc, [fp], {key: [entry]})
        tc2      = variants[PACVariant.AUTH_CORRECT]
        pos_map  = _slot_insts(tc2, fp.slot_id)
        for pos in range(PTR_SLOT_START, PTR_SLOT_START + FIX_COUNT_PTR):
            with self.subTest(pos=pos):
                self.assertEqual(pos_map[pos].name.lower(), 'nop',
                                 f"pos {pos}: expected NOP (ptr already correct)")

    def test_ctx_match_gives_nop_at_pre_ctx_positions(self):
        """pre_slot_ctx == entry.ctx_value → positions 0-3 are NOP in TC2."""
        # Find a fix_point with a ctx register
        fp = None
        stage1_tc = pac = None
        for seed in range(500):
            result = _gen_stage1(seed)
            if result is None:
                continue
            fps, tc, p = result
            candidate = next((f for f in fps if len(f.committed_inst.operands) > 1), None)
            if candidate is not None:
                fp, stage1_tc, pac = candidate, tc, p
                break
        if fp is None:
            self.skipTest("No fix_points with ctx_reg found in first 500 seeds")
        key   = _AUTH_TO_KEY[fp.committed_inst.name.lower()]
        entry = _arch_entry(key, signed=_SIGNED_VAL, ctx=_CTX_VAL)
        fp.chosen_pool_entry      = entry
        fp.pre_slot_ptr_reg_value = _PRE_PTR    # mismatch → ptr loads needed
        fp.pre_slot_ctx_reg_value = _CTX_VAL    # match
        fp.spec_nesting           = 0
        variants = pac.instrument_stage2(stage1_tc, [fp], {key: [entry]})
        tc2     = variants[PACVariant.AUTH_CORRECT]
        pos_map = _slot_insts(tc2, fp.slot_id)
        for pos in range(CTX_SLOT_START, CTX_SLOT_START + FIX_COUNT_CTX):
            with self.subTest(pos=pos):
                self.assertEqual(pos_map[pos].name.lower(), 'nop',
                                 f"pos {pos}: expected NOP (ctx already correct)")


# ===========================================================================
# 4. Key-family consistency across all three variants
# ===========================================================================

class TestPacE2EPipeline(unittest.TestCase):
    """Stage-1 PAC* sign blocks appear in the TC; stage-2 key families stay consistent."""

    @classmethod
    def setUpClass(cls):
        cls.cases = _collect_cases(50)
        assert len(cls.cases) >= 20, f"Too few PAC TCs: {len(cls.cases)}"

    def test_stage1_tc_contains_pac_sign_instructions(self):
        """Stage-1 TCs collectively contain at least one PAC* signing instruction."""
        found = False
        for _, stage1_tc, _ in self.cases:
            for func in stage1_tc.functions:
                for bb in func:
                    for inst in bb:
                        if inst.name.lower() in _PAC_INFO:
                            found = True
        self.assertTrue(found, "No PAC* sign instructions found across all stage-1 TCs")

    def test_key_family_consistent_tc1_uses_xpac_tc2_tc3_use_auth(self):
        """TC1 XPAC and TC2/TC3 AUTH at AUTH_SLOT_POS must all use the same key family."""
        for fix_points, stage1_tc, pac in self.cases:
            fps   = copy.deepcopy(fix_points)
            pool: Dict[PACKey, List[PoolEntry]] = {}
            for fp in fps:
                entry = _setup_fp_for_stage2(fp, spec_nesting=0, want_loads=True)
                pool.setdefault(entry.key, []).append(entry)
            variants = pac.instrument_stage2(stage1_tc, fps, pool)
            for fp in fps:
                auth_mn   = fp.committed_inst.name.lower()
                exp_xpac  = _AUTH_TO_XPAC[auth_mn]
                for variant, tc in variants.items():
                    inst = _slot_insts(tc, fp.slot_id).get(AUTH_SLOT_POS)
                    with self.subTest(slot_id=fp.slot_id, variant=variant.name):
                        self.assertIsNotNone(inst)
                        if variant == PACVariant.STRIP_ONLY:
                            self.assertEqual(inst.name.lower(), exp_xpac)
                        else:
                            self.assertEqual(inst.name.lower(), auth_mn)


# ===========================================================================
# 5. Layout invariant: all 4 TCs have identical instruction positions
# ===========================================================================

def _flat_insts_e2e(tc):
    result = []
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                result.append(inst)
    return result


class TestPacE2ELayoutInvariant(unittest.TestCase):
    """Stage-1 and all 3 stage-2 TCs must have identical instruction counts and
    identical positions for every instruction that is not inside a PAC slot.

    This guarantees that conditional branches (used for BPU mistraining) sit at
    the same byte offsets in all 4 TCs.  instrument_stage2 deep-copies prep_tc and
    calls _fill_slot which replaces SLOT_SIZE instructions in-place — count is
    preserved by construction.  These tests catch regressions where that invariant
    breaks (e.g. _fill_slot inserts instead of replacing).
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        for seed in range(500):
            result = _gen_stage1(seed)
            if result is None:
                continue
            fix_points, stage1_tc, pac = result
            fps = copy.deepcopy(fix_points)
            pool: Dict[PACKey, List[PoolEntry]] = {}
            for i, fp in enumerate(fps):
                entry = _setup_fp_for_stage2(fp, spec_nesting=i % 2, want_loads=True)
                pool.setdefault(entry.key, []).append(entry)
            variants = pac.instrument_stage2(stage1_tc, fps, pool)
            cls.cases.append((stage1_tc, variants))
            if len(cls.cases) >= 30:
                break
        assert len(cls.cases) >= 10, f"Too few cases: {len(cls.cases)}"

    def test_all_tcs_have_identical_instruction_count(self):
        """stage-1, TC1, TC2, TC3 must all contain the same number of instructions."""
        for stage1_tc, variants in self.cases:
            ref = len(_flat_insts_e2e(stage1_tc))
            for variant, tc in variants.items():
                with self.subTest(variant=variant.name):
                    self.assertEqual(len(_flat_insts_e2e(tc)), ref,
                                     "Instruction count changed — _fill_slot may have inserted/deleted")

    def test_non_slot_instructions_at_identical_positions_in_all_variants(self):
        """Every instruction not in a PAC slot must be at the same flat index in all 4 TCs."""
        for stage1_tc, variants in self.cases:
            ref_list = _flat_insts_e2e(stage1_tc)
            for variant, tc in variants.items():
                tc_list = _flat_insts_e2e(tc)
                for idx, ref_inst in enumerate(ref_list):
                    if hasattr(ref_inst, '_pac_slot_id'):
                        continue  # slot content differs by design
                    with self.subTest(variant=variant.name, idx=idx):
                        self.assertEqual(tc_list[idx].name, ref_inst.name)
                        self.assertEqual(tc_list[idx].template, ref_inst.template)

    def test_auth_slot_pos_at_identical_byte_offset_in_all_variants(self):
        """AUTH_SLOT_POS must map to the same flat index (= byte offset / 4) in all 4 TCs.

        This is the key guarantee that stage-1 CE mistraining applies to TC1/TC2/TC3:
        the branch targets the instruction at AUTH_SLOT_POS, which must be at the same
        address in all variants.
        """
        for stage1_tc, variants in self.cases:
            # Collect AUTH_SLOT_POS index per slot from stage-1
            stage1_list = _flat_insts_e2e(stage1_tc)
            auth_offsets: Dict[int, int] = {}
            for idx, inst in enumerate(stage1_list):
                if getattr(inst, '_pac_slot_pos', None) == AUTH_SLOT_POS:
                    auth_offsets[inst._pac_slot_id] = idx

            for variant, tc in variants.items():
                tc_list = _flat_insts_e2e(tc)
                for idx, inst in enumerate(tc_list):
                    if getattr(inst, '_pac_slot_pos', None) == AUTH_SLOT_POS:
                        sid = inst._pac_slot_id
                        with self.subTest(variant=variant.name, slot_id=sid):
                            self.assertEqual(idx, auth_offsets[sid])


# ===========================================================================
# 6. Contract trace equality: CE traces identical across all 4 TCs (CE required)
# ===========================================================================

_CE_BINARY = "/home/gal_k_1_1998/revizor/sca-fuzzer/src/aarch64/contract_executor/contract_executor"


def _ce_available() -> bool:
    return os.path.exists(_CE_BINARY)


def _tc_to_bytes_ce(tc) -> bytes:
    """Assemble tc → raw bytes for CE (no sandbox pass — production flow uses raw TC bytes)."""
    layout   = Aarch64ASMLayout(tc)
    assembly = Aarch64Printer(Aarch64TargetDesc()).print_layout(layout)
    return ConfigurableGenerator.in_memory_assemble(assembly)


def _slot_byte_offsets_from_tc(tc) -> Set[int]:
    """Return byte offsets of PAC slot instructions, consistent with Aarch64ASMLayout."""
    layout = Aarch64ASMLayout(tc)
    return {addr for inst, addr in layout.instruction_address.items()
            if hasattr(inst, '_pac_slot_id')}


@unittest.skipUnless(_ce_available(), "contract_executor binary not available")
class TestPacE2EContractTraceEquality(unittest.TestCase):
    """Run TC1/TC2/TC3 through CE with identical inputs and verify CPU state, control
    flow, and memory accesses are identical everywhere outside PAC slots.

    Uses Aarch64NonInterferenceExecutor directly — the same flow as the production
    fuzzer (stage-1 CE pool, stage-2 instrumentation, same sandbox base).

    Invariants verified:
      A. TC1 ≡ TC2: full trace identical (XPAC and AUTH_CORRECT have same net effect).
      B. TC3 arch-flow ≡ TC2 arch-flow (arch slots carry correct values).
      C. TC2 ≡ TC3 in spec-flow (CE uses XPAC semantics for AUTH under speculation).
    """

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from src.aarch64.aarch64_executor import (
            Aarch64NonInterferenceExecutor, _input_bytes_with_pstate,
        )
        from src.aarch64.aarch64_contract_executor import ContractExecution, SimArch
        from src.interfaces import Input

        NESTING  = CONF.model_max_nesting
        N_CASES  = 3
        cls.cases = []   # list of (slot_offsets, tc_traces_by_name, has_spec_slots)

        # One executor for all seeds — shares the CE subprocess and kernel connection.
        gen      = _make_gen(0)
        executor = Aarch64NonInterferenceExecutor(_TMPDIR, gen)
        sandbox_base, _ = executor.read_base_addresses()

        for seed in range(N_CASES):
            # Re-create generator and pac instrumentation per seed (same as Revizor)
            gen = _make_gen(seed)
            executor._generator          = gen
            executor._pac_instrumentation = PACInstrumentation(
                gen, CONF.pac_xpac_weight, CONF.pac_auth_weight, CONF.pac_sign_weight)

            asm_path = os.path.join(_TMPDIR, f"pac_{seed}.asm")
            tc = gen.create_test_case(asm_path, disable_assembler=True)
            executor.load_test_case(tc)

            assert executor._stage1_fix_points, \
                f"seed {seed}: stage-1 produced no PAC fix-points"

            # Random input — same structure as production fuzzer inputs
            rng = np.random.default_rng(seed)
            inp = Input()
            inp.linear_view(0)[:] = rng.integers(
                0, np.iinfo(np.uint64).max, size=len(inp.linear_view(0)), dtype=np.uint64)

            # Full production flow: stage-1 CE run → pool → stage-2 instrumentation
            _, _, stage1_traces, tc_variants_per_input = \
                executor.trace_test_case_with_taints([inp], NESTING)

            stage1_cer = stage1_traces[0]
            assert len(stage1_cer) > 0, f"seed {seed}: stage-1 CE trace is empty"

            variants = tc_variants_per_input[0]   # PACVariant → TestCase

            # Run each stage-2 variant through CE with the same input
            data     = _input_bytes_with_pstate(inp)
            tc_mem   = data[:0x2000]
            tc_regs  = data[0x2000:]
            traces   = {}
            for variant, variant_tc in variants.items():
                tc_bytes  = executor._tc_to_bytes(variant_tc)
                execution = ContractExecution(
                    tc_bytes, tc_mem, tc_regs,
                    SimArch.RVZR_ARCH_AARCH64, NESTING, 10,
                    req_mem_base_virt=sandbox_base,
                )
                traces[variant.name] = executor._contract_executor.run(execution)

            slot_offsets  = _slot_byte_offsets_from_tc(executor._stage1_tc)
            has_spec_slots = any(e.metadata.speculation_nesting != 0 for e in stage1_cer)
            cls.cases.append((slot_offsets, traces, has_spec_slots))

        assert len(cls.cases) == N_CASES

    # -----------------------------------------------------------------------
    # Comparison helpers
    # -----------------------------------------------------------------------

    _OBS_GPRS = list(range(6)) + [29]   # x0-x5, x29

    def _compare_entries(self, ite_a, ite_b, base_a: int, base_b: int,
                         slot_offsets: Set[int], label: str, step: int):
        """Compare one pair of trace entries entry by entry.

        Compared fields (outside slot intermediate state):
          - PC offset (control flow)
          - Speculation nesting
          - Registers x0-x5, x29, sp, nzcv (observation footprint)
          - Memory access: sandbox-relative EA, before-value, after-value, size, direction
        Exempted: entire slot (positions 0-12) = "intermediate state" where
        MOVZ/MOVK/NOP/XPAC/AUTH instructions may produce different intermediate
        register values.  After the slot the net effect is identical.
        """
        offset_a = ite_a.cpu.pc - base_a
        offset_b = ite_b.cpu.pc - base_b

        # Control flow
        self.assertEqual(offset_a, offset_b,
                         f"{label} step {step}: PC offset diverged "
                         f"(a=+{offset_a:#x}, b=+{offset_b:#x})")

        # Speculation nesting
        self.assertEqual(ite_a.metadata.speculation_nesting,
                         ite_b.metadata.speculation_nesting,
                         f"{label} step {step} +{offset_a:#x}: speculation_nesting differs")

        if offset_a in slot_offsets:
            # Slot intermediate state: skip register and value comparison.
            # PAC slots must never perform memory accesses.
            self.assertFalse(ite_a.metadata.has_memory_access,
                             f"{label} step {step} slot +{offset_a:#x}: "
                             "unexpected memory access inside PAC slot (a)")
            self.assertFalse(ite_b.metadata.has_memory_access,
                             f"{label} step {step} slot +{offset_a:#x}: "
                             "unexpected memory access inside PAC slot (b)")
            return

        # Observation registers: x0-x5, x29, sp, nzcv
        for r in self._OBS_GPRS:
            self.assertEqual(ite_a.cpu.gpr[r], ite_b.cpu.gpr[r],
                             f"{label} step {step} +{offset_a:#x}: x{r} differs "
                             f"(a={ite_a.cpu.gpr[r]:#x}, b={ite_b.cpu.gpr[r]:#x})")
        self.assertEqual(ite_a.cpu.sp,   ite_b.cpu.sp,
                         f"{label} step {step} +{offset_a:#x}: sp differs")
        self.assertEqual(ite_a.cpu.nzcv, ite_b.cpu.nzcv,
                         f"{label} step {step} +{offset_a:#x}: nzcv differs")

        # Memory access presence
        self.assertEqual(ite_a.metadata.has_memory_access,
                         ite_b.metadata.has_memory_access,
                         f"{label} step {step} +{offset_a:#x}: memory_access presence differs")
        if not ite_a.metadata.has_memory_access:
            return

        ma_a = ite_a.metadata.memory_access
        ma_b = ite_b.metadata.memory_access

        # EA relative to x29 (sandbox base)
        rel_a = ma_a.effective_address - ite_a.cpu.gpr[29]
        rel_b = ma_b.effective_address - ite_b.cpu.gpr[29]
        self.assertEqual(rel_a, rel_b,
                         f"{label} step {step} +{offset_a:#x}: "
                         f"memory EA differs (rel_a={rel_a:#x}, rel_b={rel_b:#x})")

        # Value before the access (what was in memory)
        self.assertEqual(ma_a.before, ma_b.before,
                         f"{label} step {step} +{offset_a:#x}: "
                         f"memory before-value differs ({ma_a.before:#x} vs {ma_b.before:#x})")

        # Value after the access (what was written, or same as before for reads)
        self.assertEqual(ma_a.after, ma_b.after,
                         f"{label} step {step} +{offset_a:#x}: "
                         f"memory after-value differs ({ma_a.after:#x} vs {ma_b.after:#x})")

        self.assertEqual(ma_a.element_size, ma_b.element_size,
                         f"{label} step {step} +{offset_a:#x}: memory element_size differs")
        self.assertEqual(ma_a.is_write, ma_b.is_write,
                         f"{label} step {step} +{offset_a:#x}: memory is_write differs")

    def _compare_full(self, trace_a, trace_b, slot_offsets: Set[int], label: str,
                      arch_only: bool = False):
        """Compare two traces entry by entry, optionally filtering to arch-flow only."""
        if arch_only:
            entries_a = [e for e in trace_a if e.metadata.speculation_nesting == 0]
            entries_b = [e for e in trace_b if e.metadata.speculation_nesting == 0]
        else:
            entries_a = list(trace_a)
            entries_b = list(trace_b)

        if not entries_a:
            return  # empty trace — nothing to compare

        self.assertEqual(len(entries_a), len(entries_b),
                         f"{label}: trace length differs "
                         f"({len(entries_a)} vs {len(entries_b)})")

        base_a = entries_a[0].cpu.pc
        base_b = entries_b[0].cpu.pc
        for step, (ite_a, ite_b) in enumerate(zip(entries_a, entries_b)):
            self._compare_entries(ite_a, ite_b, base_a, base_b, slot_offsets, label, step)

    # -----------------------------------------------------------------------
    # Test A: TC1 ≡ TC2  —  full trace (arch + spec)
    #
    # TC1 uses XPAC and TC2 uses AUTH with the real CE-signed value.
    # Both produce identical net register/memory effects; only the slot
    # intermediate state (MOVZ/MOVK/XPAC vs AUTH) differs.
    # -----------------------------------------------------------------------

    def test_A_tc1_tc2_identical_full_trace(self):
        """TC1 (STRIP_ONLY) ≡ TC2 (AUTH_CORRECT): entire trace identical
        (arch + spec flow) except slot intermediate state (positions 0-12)."""
        for i, (slot_offsets, traces, _) in enumerate(self.cases):
            with self.subTest(case=i):
                self._compare_full(traces['STRIP_ONLY'], traces['AUTH_CORRECT'],
                                   slot_offsets,
                                   label=f"case {i} TC1 vs TC2")

    # -----------------------------------------------------------------------
    # Test B: TC3 arch-flow ≡ TC2 arch-flow
    #
    # TC3 arch slots use correct values → identical to TC2.
    # TC3 spec-flow intentionally diverges (test C); excluded here.
    # -----------------------------------------------------------------------

    def test_B_tc3_arch_flow_identical_to_tc2(self):
        """TC3 (AUTH_WRONG) arch-flow must be identical to TC2 (AUTH_CORRECT).
        Arch slots have correct values; non-slot arch instructions are unchanged."""
        for i, (slot_offsets, traces, _) in enumerate(self.cases):
            with self.subTest(case=i):
                self._compare_full(traces['AUTH_CORRECT'], traces['AUTH_WRONG'],
                                   slot_offsets,
                                   label=f"case {i} TC2 vs TC3 arch",
                                   arch_only=True)

    def _compare_spec_flows(self, trace_a, trace_b, slot_offsets: Set[int], label: str):
        """Compare TC2 vs TC3 spec-flow entry-by-entry.

        CE uses XPAC semantics for AUTH in spec mode, so the net effect of
        the slot is identical between TC2 and TC3.  The only thing that may
        differ is intermediate register values *inside* the slot (MOVZ/MOVK
        load different raw values), which _compare_entries already skips.
        """
        all_a = list(trace_a)
        all_b = list(trace_b)
        if not all_a or not all_b:
            return
        # Use code base from the first entry so offset arithmetic matches slot_offsets.
        base_a = all_a[0].cpu.pc
        base_b = all_b[0].cpu.pc
        spec_a = [e for e in all_a if e.metadata.speculation_nesting != 0]
        spec_b = [e for e in all_b if e.metadata.speculation_nesting != 0]
        if not spec_a:
            return
        self.assertEqual(len(spec_a), len(spec_b),
                         f"{label}: spec trace length differs "
                         f"({len(spec_a)} vs {len(spec_b)})")
        for step, (ite_a, ite_b) in enumerate(zip(spec_a, spec_b)):
            self._compare_entries(ite_a, ite_b, base_a, base_b,
                                  slot_offsets, label, step)

    # -----------------------------------------------------------------------
    # Test C: TC3 spec-flow identical to TC2 spec-flow
    #
    # CE uses XPAC semantics for AUTH in spec mode: both TC2 (AUTIA with real
    # value) and TC3 (AUTIA with wrong value) produce the same stripped pointer.
    # Slot intermediate register state may differ (MOVZ loads different bits),
    # but _compare_entries already skips intra-slot registers.
    # CE must complete TC3 without crashing (trace non-empty).
    # -----------------------------------------------------------------------

    def test_C_tc3_spec_flow_identical_to_tc2(self):
        """TC2 ≡ TC3 in spec-flow: CE XPAC semantics make slot net-effect identical."""
        for i, (slot_offsets, traces, _) in enumerate(self.cases):
            with self.subTest(case=i):
                tc3_trace = list(traces['AUTH_WRONG'])
                self.assertGreater(len(tc3_trace), 0,
                                   f"case {i}: TC3 CE trace is empty (CE crashed?)")
                self._compare_spec_flows(
                    traces['AUTH_CORRECT'], tc3_trace,
                    slot_offsets, label=f"case {i} TC2 vs TC3 spec")


if __name__ == '__main__':
    unittest.main()
