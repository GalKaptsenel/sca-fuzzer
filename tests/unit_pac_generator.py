"""
PAC generator pipeline tests: TC generation → stage1 → simulated executor → stage2.

Each test works over randomly-seeded test cases.  The executor is simulated by
querying the kernel module for real PAC signatures for each FixPoint.

Invariants under test (stage1):
  - Every slot has exactly SLOT_SIZE instructions tagged with _pac_slot_id
  - Position SLOT_SIG_POS  = NOP
  - Position AUTH_SLOT_POS = XPAC (xpaci or xpacd)
  - committed_inst is an AUT* instruction
  - All slot_ids within a TC are unique

Invariants under test (stage2):
  - All 3 variants have the same flat instruction count (layout preserved)
  - Slot flat indices are identical across all 3 variants
  - TC1 (STRIP_ONLY)  : every slot → [NOP, XPAC]
  - TC2 (AUTH_CORRECT): every slot → [MOVK correct_sig, AUTH]
  - TC3 arch (spec_nesting == 0): TC3 slot identical to TC2 slot
  - TC3 spec (spec_nesting  > 0): every slot → [MOVK alt_sig,  AUTH]
  - AUTH mnemonic in TC2/TC3 matches committed_inst
  - XPAC mnemonic in TC1 matches committed_inst key family (xpaci vs xpacd)

CE arch-path state identity (stage2 vs stage1):
  For every non-slotted arch-path (nesting==0) instruction in all 4 TCs
  (stage1, TC1, TC2, TC3), the following fields are identical:
    x0-x5, SP, NZCV
    memory access offset relative to sandbox base (ea - x29), value-before, value-after
  This holds because slot instructions only mutate ptr_reg and produce the
  same stripped address in every variant:
    - arch slots (nesting==0) → correct auth → stripped result = XPAC result
    - spec slots  (nesting>0) → XPAC semantics in CE → stripped result
"""
import copy
import os
import random
import re
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONF
from src.isa_loader import InstructionSet
from src.interfaces import TestCase, Instruction, Input
from src.input_generator import NumpyRandomInputGenerator
from src.aarch64.aarch64_generator import (
    PACInstrumentation, PACVariant, FixPoint,
    Aarch64RandomGenerator, Aarch64Printer, Aarch64ASMLayout,
    SLOT_SIZE, SLOT_SIG_POS, AUTH_SLOT_POS,
    _AUTH_TO_PAC, _AUTH_TO_XPAC, _AUTH_TO_KEY, PACKey,
)
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc
from src.aarch64.aarch64_contract_executor import (
    ContractExecution, ContractExecutionResult, ContractExecutorService, SimArch,
)
from src.aarch64.aarch64_executor import _reconstruct_pstate
from src.generator import ConfigurableGenerator

_isa:          Optional[InstructionSet] = None
_executor      = None    # LocalExecutorImp — pac_sign() + sandbox_base
_ce:           Optional[ContractExecutorService] = None
_target_desc:  Optional[Aarch64TargetDesc] = None
_input_gen:    Optional[NumpyRandomInputGenerator] = None
_tmpdir:       Optional[str] = None
_sandbox_base: Optional[int] = None

_AUTH_NAMES = frozenset(_AUTH_TO_PAC.keys())
_XPAC_NAMES = frozenset({'xpaci', 'xpacd'})
_I_KEY_XPAC = 'xpaci'
_D_KEY_XPAC = 'xpacd'

_CE_BINARY = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "src/aarch64/contract_executor/contract_executor",
))


def setUpModule():
    global _isa, _executor, _ce, _target_desc, _input_gen, _tmpdir, _sandbox_base
    if not os.path.exists('/dev/executor'):
        raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
    try:
        from src.aarch64.aarch64_connection import LocalExecutorImp, PacKeys
        CONF.load("config.yml")
        _isa          = InstructionSet("base.json", CONF.instruction_categories)
        _target_desc  = Aarch64TargetDesc()
        _executor     = LocalExecutorImp('/dev/executor', '/sys/executor', '')
        # Fix PAC keys so _pac_sign (Python) and the CE both use the same key context.
        _ce           = ContractExecutorService(_CE_BINARY)
        keys = PacKeys()
        keys.apia_lo = 0x1234567890abcdef; keys.apia_hi = 0xfedcba9876543210
        keys.apib_lo = 0x0f1e2d3c4b5a6978; keys.apib_hi = 0x8796a5b4c3d2e1f0
        keys.apda_lo = 0x1122334455667788; keys.apda_hi = 0x8877665544332211
        keys.apdb_lo = 0xaabbccddeeff0011; keys.apdb_hi = 0x1100ffeeddccbbaa
        keys.apga_lo = 0x0102030405060708; keys.apga_hi = 0x0807060504030201
        _executor.set_pac_keys(keys)
        _sandbox_base = _executor.sandbox_base
        _input_gen    = NumpyRandomInputGenerator(seed=random.randrange(1 << 32))
        _tmpdir       = tempfile.mkdtemp()
    except Exception as e:
        raise unittest.SkipTest(f"executor setup failed: {e}")


def tearDownModule():
    import shutil
    if _tmpdir:
        shutil.rmtree(_tmpdir, ignore_errors=True)


# ===========================================================================
# Core pipeline helpers
# ===========================================================================

def _pac_sign(ptr: int, ctx: int, auth_mn: str) -> int:
    """Top 16 bits of kernel PAC signature for (ptr, ctx, auth_mn)."""
    return (_executor.pac_sign(ptr, ctx, _AUTH_TO_PAC[auth_mn]) >> 48) & 0xFFFF


def _flat_insts(tc: TestCase) -> List[Instruction]:
    return [i for func in tc.functions for bb in func for i in bb]


def _slot(tc: TestCase, slot_id: int) -> List[Instruction]:
    pos_map: Dict[int, Instruction] = {}
    for inst in _flat_insts(tc):
        if getattr(inst, '_pac_slot_id', None) == slot_id:
            pos_map[inst._pac_slot_pos] = inst
    return [pos_map[p] for p in range(SLOT_SIZE)]


def _sig_from_movk(inst: Instruction) -> int:
    m = re.search(r'#0x([0-9a-fA-F]+)', inst.template)
    assert m, f"no immediate in {inst.template!r}"
    return int(m.group(1), 16) & 0xFFFF


def _read_gpr(cpu, reg: str) -> int:
    return cpu.sp if reg == 'sp' else cpu.gpr[int(reg[1:])]


def _other_gpr(cpu, exclude1: str, exclude2: Optional[str]) -> int:
    """Value of a GPR other than exclude1/exclude2 and the sandbox/LR registers."""
    excl = {exclude1, exclude2, 'x29', 'x30'} if exclude2 else {exclude1, 'x29', 'x30'}
    candidates = [f"x{i}" for i in range(29) if f"x{i}" not in excl]
    return _read_gpr(cpu, random.choice(candidates))


def _fill_fixpoints_from_ce(
    stage1_tc: TestCase,
    fix_points: List[FixPoint],
    _combo_observer: Optional[Any] = None,
) -> Input:
    """Run stage1_tc through the real CE and populate fix_point metadata from the trace.

    For each XPAC slot reached by the CE:
      correct_sig  — pac_sign(actual ptr, actual ctx at slot PC)
      alt_sig      — pac_sign(a different GPR's value for ptr or ctx)
      spec_nesting — actual CE nesting depth (min over all hits)

    For slots the CE never reached:
      correct_sig  stays None
      alt_sig      — pac_sign(random ptr, random ctx)
      spec_nesting stays None  (instrument_stage2 treats None as speculative)

    _combo_observer: optional callable(wrong_ptr: bool, wrong_ctx: bool) invoked once per
                     alt_sig decision.  Intended for tests that count combo coverage.

    Returns the input used to run the CE so callers can reuse it.
    """
    # Map byte offset of XPAC placeholder → FixPoint (same as production executor)
    layout = Aarch64ASMLayout(stage1_tc)
    xpac_off_to_fp: Dict[int, FixPoint] = {
        layout.instruction_address[fp.slot_insts[AUTH_SLOT_POS]]: fp
        for fp in fix_points
    }

    inp = _random_input()
    cer = _run_ce(_tc_to_bytes(stage1_tc), inp)

    if len(cer) > 0:
        code_base = cer[0].cpu.pc
        for ite in cer:
            rel = ite.cpu.pc - code_base
            if rel not in xpac_off_to_fp:
                continue
            fp    = xpac_off_to_fp[rel]
            depth = ite.metadata.speculation_nesting

            # Track shallowest nesting seen (prefer arch path for correct_sig)
            if fp.spec_nesting is None or depth < fp.spec_nesting:
                fp.spec_nesting = depth

            # Populate correct_sig and alt_sig on the first arch hit (or any hit)
            if depth == 0 or fp.correct_sig is None:
                auth_mn = fp.committed_inst.name.lower()
                has_ctx = len(fp.committed_inst.operands) > 1
                ptr_reg = fp.committed_inst.operands[0].value
                ctx_reg = fp.committed_inst.operands[1].value if has_ctx else None

                ptr = _read_gpr(ite.cpu, ptr_reg)
                ctx = _read_gpr(ite.cpu, ctx_reg) if has_ctx else 0
                fp.correct_sig = _pac_sign(ptr, ctx, auth_mn)

                # alt_sig: substitute the ptr or ctx with a value from a different GPR.
                # Force bits[63:48]=0xFFFF so alt_ptr is a canonical kernel VA: PACDA then
                # sets bit 55 in the signed pointer, and XPAC can correctly sign-extend it
                # back to 0xffff... instead of producing a non-canonical 0x0000... address.
                use_wrong_ptr = random.choice([True, False])
                use_wrong_ctx = has_ctx and random.choice([True, False])
                alt_ptr = _other_gpr(ite.cpu, ptr_reg, ctx_reg) if use_wrong_ptr else ptr
                alt_ptr = (alt_ptr & 0x0000FFFFFFFFFFFF) | 0xFFFF000000000000
                alt_ctx = (_other_gpr(ite.cpu, ctx_reg, ptr_reg) if use_wrong_ctx else ctx) if has_ctx else 0
                fp.alt_sig = _pac_sign(alt_ptr, alt_ctx, auth_mn)
                if _combo_observer is not None:
                    _combo_observer(use_wrong_ptr, use_wrong_ctx)

    # Slots CE never reached: alt_sig from random values, correct_sig stays None
    for fp in fix_points:
        if fp.alt_sig is None:
            auth_mn = fp.committed_inst.name.lower()
            has_ctx = len(fp.committed_inst.operands) > 1
            fp.alt_sig = _pac_sign(
                random.randrange(1 << 48) | 0xFFFF000000000000,
                random.randrange(1 << 64) if has_ctx else 0,
                auth_mn,
            )

    return inp


def _run_stage1(max_attempts: int = 500) -> Tuple[PACInstrumentation, TestCase, List[FixPoint]]:
    """Generate TC and run stage1 with random seeds until fix_points are produced.

    Never returns without fix_points — raises RuntimeError after max_attempts.
    """
    for _ in range(max_attempts):
        seed = random.randrange(1 << 32)
        asm_path = os.path.join(_tmpdir, f"pac_{seed}.asm")
        gen = Aarch64RandomGenerator(_isa, seed)
        pac = PACInstrumentation(gen, xpac_weight=1, auth_weight=1)
        try:
            tc = gen.create_test_case(asm_path, disable_assembler=True)
        except Exception:
            continue
        stage1_tc, fix_points = pac.instrument_stage1(copy.deepcopy(tc))
        if fix_points:
            return pac, stage1_tc, fix_points
    raise RuntimeError(f"No fix_points found after {max_attempts} random seeds")


def _run_pipeline() -> Tuple[TestCase, List[FixPoint], Dict, Input]:
    """Full pipeline: stage1 → CE trace → fix_point fill → stage2. Always returns a result.

    Returns (stage1_tc, fix_points, variants, ref_input) where ref_input is the
    same input used to classify spec_nesting in _fill_fixpoints_from_ce.  Reuse
    ref_input when running the CE over stage2 variants so that input-dependent
    execution paths (spec vs arch) are identical across all four test cases.
    """
    pac, stage1_tc, fix_points = _run_stage1()
    ref_input = _fill_fixpoints_from_ce(stage1_tc, fix_points)
    variants = pac.instrument_stage2(stage1_tc, fix_points)
    return stage1_tc, fix_points, variants, ref_input


def _collect_cases(n: int = 20) -> list:
    return [_run_pipeline() for _ in range(n)]


# ===========================================================================
# CE execution helpers
# ===========================================================================

def _tc_to_bytes(tc: TestCase) -> bytes:
    layout   = Aarch64ASMLayout(tc)
    assembly = Aarch64Printer(_target_desc).print_layout(layout)
    return ConfigurableGenerator.in_memory_assemble(assembly)


def _slot_offsets(tc: TestCase) -> Set[int]:
    """Byte offsets (from TC start) of every slotted instruction."""
    layout = Aarch64ASMLayout(tc)
    return {layout.instruction_address[i]
            for i in _flat_insts(tc) if hasattr(i, '_pac_slot_id')}


def _random_input() -> Input:
    return _input_gen.generate(1)[0]


def _run_ce(tc_bytes: bytes, inp: Input) -> ContractExecutionResult:
    data      = inp.tobytes()
    tc_memory = data[:0x2000]
    tc_regs   = bytearray(data[0x2000:])
    _reconstruct_pstate(memoryview(tc_regs).cast('Q'))
    execution = ContractExecution(
        tc_bytes, bytes(tc_memory), bytes(tc_regs),
        SimArch.RVZR_ARCH_AARCH64, 5, 10,
        req_mem_base_virt=_sandbox_base,
    )
    return _ce.run(execution)


def _arch_state_entries(cer: ContractExecutionResult,
                        slot_offs: Set[int]) -> List[Dict[str, Any]]:
    """Extract arch-path (nesting==0) non-slot trace entries with full architectural state.

    Each entry dict contains:
      rel_pc, encoding,
      x0_x5 (tuple of 6 values), sp, nzcv,
      and if has_memory_access: mem_offset (ea - x29), mem_before, mem_after.
    """
    if len(cer) == 0:
        return []
    code_base = cer[0].cpu.pc
    result = []
    for ite in cer:
        if ite.metadata.speculation_nesting != 0:
            continue
        rel = ite.cpu.pc - code_base
        if rel in slot_offs:
            continue
        entry: Dict[str, Any] = {
            'rel_pc':   rel,
            'encoding': ite.cpu.encoding,
            'x0_x5':   tuple(ite.cpu.gpr[i] for i in range(6)),
            'sp':       ite.cpu.sp,
            'nzcv':     ite.cpu.nzcv,
        }
        if ite.metadata.has_memory_access:
            ma = ite.metadata.memory_access
            sandbox_base = ite.cpu.gpr[29]    # x29 is the sandbox base register
            entry['mem_offset'] = ma.effective_address - sandbox_base
            entry['mem_before'] = ma.before
            entry['mem_after']  = ma.after
        result.append(entry)
    return result


# ===========================================================================
# Stage-1 structural invariants
# ===========================================================================

class TestStage1Structure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _collect_cases(20)

    def test_slot_ids_are_unique_per_tc(self):
        for stage1_tc, fix_points, _, _inp in self.cases:
            ids = [fp.slot_id for fp in fix_points]
            self.assertEqual(len(ids), len(set(ids)))

    def test_each_slot_has_exactly_slot_size_instructions(self):
        for stage1_tc, fix_points, _, _inp in self.cases:
            all_insts = _flat_insts(stage1_tc)
            for fp in fix_points:
                count = sum(1 for i in all_insts
                            if getattr(i, '_pac_slot_id', None) == fp.slot_id)
                self.assertEqual(count, SLOT_SIZE,
                                 f"slot {fp.slot_id}: expected {SLOT_SIZE}, got {count}")

    def test_slot_sig_pos_is_nop(self):
        for stage1_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertEqual(_slot(stage1_tc, fp.slot_id)[SLOT_SIG_POS].name, 'nop')

    def test_auth_slot_pos_is_xpac(self):
        for stage1_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertIn(_slot(stage1_tc, fp.slot_id)[AUTH_SLOT_POS].name, _XPAC_NAMES)

    def test_committed_inst_is_auth_mnemonic(self):
        for _, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertIn(fp.committed_inst.name.lower(), _AUTH_NAMES)

    def test_xpac_family_matches_committed_inst_key(self):
        for stage1_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                key = _AUTH_TO_KEY[fp.committed_inst.name.lower()]
                xpac_mn = _slot(stage1_tc, fp.slot_id)[AUTH_SLOT_POS].name
                expected = _I_KEY_XPAC if key in (PACKey.IA, PACKey.IB) else _D_KEY_XPAC
                self.assertEqual(xpac_mn, expected,
                                 f"slot {fp.slot_id}: wrong xpac family")


# ===========================================================================
# Stage-2 layout invariants
# ===========================================================================

class TestStage2Layout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _collect_cases(20)

    def test_all_variants_same_instruction_count(self):
        for stage1_tc, _, variants, _inp in self.cases:
            ref = len(_flat_insts(stage1_tc))
            for variant, tc in variants.items():
                self.assertEqual(len(_flat_insts(tc)), ref,
                                 f"{variant.name}: instruction count changed")

    def test_slot_flat_indices_identical_across_variants(self):
        for stage1_tc, _, variants, _inp in self.cases:
            ref_idx: Dict[Tuple[int, int], int] = {}
            for idx, inst in enumerate(_flat_insts(stage1_tc)):
                if hasattr(inst, '_pac_slot_id'):
                    ref_idx[(inst._pac_slot_id, inst._pac_slot_pos)] = idx
            for variant, tc in variants.items():
                for idx, inst in enumerate(_flat_insts(tc)):
                    if hasattr(inst, '_pac_slot_id'):
                        key = (inst._pac_slot_id, inst._pac_slot_pos)
                        self.assertEqual(idx, ref_idx[key],
                                         f"{variant.name} slot {key}: flat index moved")

    def test_non_slot_instructions_unchanged(self):
        for stage1_tc, _, variants, _inp in self.cases:
            ref_insts = _flat_insts(stage1_tc)
            for variant, tc in variants.items():
                tc_insts = _flat_insts(tc)
                for idx, ref in enumerate(ref_insts):
                    if hasattr(ref, '_pac_slot_id'):
                        continue
                    with self.subTest(variant=variant.name, idx=idx):
                        self.assertEqual(tc_insts[idx].name,     ref.name)
                        self.assertEqual(tc_insts[idx].template, ref.template)


# ===========================================================================
# Stage-2 variant operation contracts
# ===========================================================================

class TestStage2VariantContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _collect_cases(20)

    def test_tc1_no_auth_instructions(self):
        for _, fix_points, variants, _inp in self.cases:
            tc1 = variants[PACVariant.STRIP_ONLY]
            for inst in _flat_insts(tc1):
                self.assertNotIn(inst.name.lower(), _AUTH_NAMES)

    def test_tc2_slots_carry_correct_sig_and_auth(self):
        for _, fix_points, variants, _inp in self.cases:
            tc2 = variants[PACVariant.AUTH_CORRECT]
            for fp in fix_points:
                s = _slot(tc2, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[SLOT_SIG_POS].name, 'movk')
                    self.assertEqual(_sig_from_movk(s[SLOT_SIG_POS]), fp.correct_sig)
                    self.assertIn(s[AUTH_SLOT_POS].name, _AUTH_NAMES)

    def test_tc2_auth_mnemonic_matches_committed_inst(self):
        for _, fix_points, variants, _inp in self.cases:
            tc2 = variants[PACVariant.AUTH_CORRECT]
            for fp in fix_points:
                s = _slot(tc2, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[AUTH_SLOT_POS].name,
                                     fp.committed_inst.name.lower())

    def test_tc3_arch_slot_identical_to_tc2(self):
        for _, fix_points, variants, _inp in self.cases:
            tc2 = variants[PACVariant.AUTH_CORRECT]
            tc3 = variants[PACVariant.AUTH_WRONG]
            for fp in fix_points:
                if fp.spec_nesting != 0:
                    continue
                s2 = _slot(tc2, fp.slot_id)
                s3 = _slot(tc3, fp.slot_id)
                for pos, (i2, i3) in enumerate(zip(s2, s3)):
                    with self.subTest(slot=fp.slot_id, pos=pos):
                        self.assertEqual(i2.name,     i3.name)
                        self.assertEqual(i2.template, i3.template)

    def test_tc3_spec_slot_carries_alt_sig(self):
        for _, fix_points, variants, _inp in self.cases:
            tc3 = variants[PACVariant.AUTH_WRONG]
            for fp in fix_points:
                if fp.spec_nesting == 0:
                    continue
                s = _slot(tc3, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[SLOT_SIG_POS].name, 'movk')
                    self.assertEqual(_sig_from_movk(s[SLOT_SIG_POS]), fp.alt_sig)
                    self.assertIn(s[AUTH_SLOT_POS].name, _AUTH_NAMES)

    def test_tc3_spec_auth_mnemonic_matches_committed_inst(self):
        for _, fix_points, variants, _inp in self.cases:
            tc3 = variants[PACVariant.AUTH_WRONG]
            for fp in fix_points:
                if fp.spec_nesting == 0:
                    continue
                s = _slot(tc3, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[AUTH_SLOT_POS].name,
                                     fp.committed_inst.name.lower())

    def test_xpac_family_preserved_in_tc1(self):
        for _, fix_points, variants, _inp in self.cases:
            tc1 = variants[PACVariant.STRIP_ONLY]
            for fp in fix_points:
                key = _AUTH_TO_KEY[fp.committed_inst.name.lower()]
                expected = _I_KEY_XPAC if key in (PACKey.IA, PACKey.IB) else _D_KEY_XPAC
                s = _slot(tc1, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[AUTH_SLOT_POS].name, expected)


# ===========================================================================
# CE-never-reached path (correct_sig=None)
# ===========================================================================

class TestCENeverReached(unittest.TestCase):
    """When correct_sig=None: AUTH_CORRECT → TC1, AUTH_WRONG → TC3-spec with alt_sig."""

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        while len(cls.cases) < 5:
            pac, stage1_tc, fix_points = _run_stage1()
            _fill_fixpoints_from_ce(stage1_tc, fix_points)   # populates alt_sig + spec_nesting
            for fp in fix_points:
                fp.correct_sig = None    # force never-reached: TC2 falls back to XPAC
            variants = pac.instrument_stage2(stage1_tc, fix_points)
            cls.cases.append((stage1_tc, fix_points, variants))

    def test_auth_correct_falls_back_to_xpac(self):
        for _, fix_points, variants in self.cases:
            tc_correct = variants[PACVariant.AUTH_CORRECT]
            for fp in fix_points:
                s = _slot(tc_correct, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[SLOT_SIG_POS].name, 'nop')
                    self.assertIn(s[AUTH_SLOT_POS].name, _XPAC_NAMES)

    def test_auth_wrong_uses_alt_sig_even_when_unreached(self):
        for _, fix_points, variants in self.cases:
            tc_wrong = variants[PACVariant.AUTH_WRONG]
            for fp in fix_points:
                s = _slot(tc_wrong, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[SLOT_SIG_POS].name, 'movk')
                    self.assertEqual(_sig_from_movk(s[SLOT_SIG_POS]), fp.alt_sig)
                    self.assertIn(s[AUTH_SLOT_POS].name, _AUTH_NAMES)

    def test_strip_only_unaffected(self):
        for _, fix_points, variants in self.cases:
            tc1 = variants[PACVariant.STRIP_ONLY]
            for fp in fix_points:
                s = _slot(tc1, fp.slot_id)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[SLOT_SIG_POS].name, 'nop')
                    self.assertIn(s[AUTH_SLOT_POS].name, _XPAC_NAMES)


# ===========================================================================
# CE arch-path state identity: stage1 vs each stage2 variant
# ===========================================================================

class TestCEArchStateIdentity(unittest.TestCase):
    """For every non-slotted arch-path instruction the architectural state must be
    identical between stage1_tc and each of TC1, TC2, TC3:
      - Registers x0-x5, SP, NZCV
      - Memory offset relative to sandbox base (ea - x29), value before, value after
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = _collect_cases(10)

    def _compare_vs_stage1(self,
                           ref_entries: List[Dict],
                           cmp_entries: List[Dict],
                           label: str) -> None:
        self.assertEqual(len(cmp_entries), len(ref_entries),
                         f"{label}: arch entry count differs "
                         f"(expected {len(ref_entries)}, got {len(cmp_entries)})")
        for i, (ref, cmp) in enumerate(zip(ref_entries, cmp_entries)):
            with self.subTest(variant=label, entry=i):
                self.assertEqual(cmp['rel_pc'],   ref['rel_pc'],
                                 f"entry {i}: rel_pc mismatch")
                self.assertEqual(cmp['encoding'], ref['encoding'],
                                 f"entry {i}: encoding mismatch")
                self.assertEqual(cmp['x0_x5'],   ref['x0_x5'],
                                 f"entry {i}: x0-x5 mismatch")
                self.assertEqual(cmp['sp'],       ref['sp'],
                                 f"entry {i}: SP mismatch")
                self.assertEqual(cmp['nzcv'],     ref['nzcv'],
                                 f"entry {i}: NZCV mismatch")
                if 'mem_offset' in ref:
                    self.assertIn('mem_offset', cmp,
                                  f"entry {i}: memory access missing in {label}")
                    self.assertEqual(cmp['mem_offset'], ref['mem_offset'],
                                     f"entry {i}: memory offset (ea-x29) mismatch")
                    self.assertEqual(cmp['mem_before'], ref['mem_before'],
                                     f"entry {i}: memory value-before mismatch")
                    self.assertEqual(cmp['mem_after'],  ref['mem_after'],
                                     f"entry {i}: memory value-after mismatch")

    def _ref_and_variant(self, stage1_tc, tc, inp):
        slot_offs   = _slot_offsets(stage1_tc)
        ref_entries = _arch_state_entries(_run_ce(_tc_to_bytes(stage1_tc), inp), slot_offs)
        cmp_entries = _arch_state_entries(_run_ce(_tc_to_bytes(tc), inp), slot_offs)
        return ref_entries, cmp_entries

    def test_stage1_vs_tc1(self):
        """TC1 (STRIP_ONLY) is identical to stage1 — arch state must match exactly."""
        for stage1_tc, _, variants, ref_input in self.cases:
            ref, cmp = self._ref_and_variant(
                stage1_tc, variants[PACVariant.STRIP_ONLY], ref_input)
            self._compare_vs_stage1(ref, cmp, "STRIP_ONLY")

    def test_stage1_vs_tc2(self):
        """TC2 (AUTH_CORRECT) uses correct signatures → same stripped ptr → same arch state."""
        for stage1_tc, _, variants, ref_input in self.cases:
            ref, cmp = self._ref_and_variant(
                stage1_tc, variants[PACVariant.AUTH_CORRECT], ref_input)
            self._compare_vs_stage1(ref, cmp, "AUTH_CORRECT")

    def test_stage1_vs_tc3(self):
        """TC3 (AUTH_WRONG) arch slots = TC2, spec slots use XPAC semantics → same arch state.

        Must use the same input as _fill_fixpoints_from_ce so that slots classified
        as speculative (spec_nesting > 0) remain at spec depth during TC3 execution.
        If a different input were used, a previously-speculative slot might execute in
        the arch path with alt_sig → auth failure → poisoned pointer → SIGSEGV.
        """
        for stage1_tc, _, variants, ref_input in self.cases:
            ref, cmp = self._ref_and_variant(
                stage1_tc, variants[PACVariant.AUTH_WRONG], ref_input)
            self._compare_vs_stage1(ref, cmp, "AUTH_WRONG")


# ===========================================================================
# Statistical coverage: all four (use_wrong_ptr, use_wrong_ctx) combos appear
# ===========================================================================

class TestAltSigCombinationCoverage(unittest.TestCase):
    """Statistical test: every (use_wrong_ptr, use_wrong_ctx) combination — (T,T), (T,F),
    (F,T), (F,F) — must appear across repeated runs of _fill_fixpoints_from_ce.

    Strategy:
      - Find a TC with at least one two-operand auth slot so the ctx dimension is covered.
      - Run _fill_fixpoints_from_ce N times; each run passes a _combo_observer that
        directly receives (wrong_ptr, wrong_ctx) as chosen by the production code.
      - Assert all four combos appear.  With N=40 runs and ≥1 two-op slot per run,
        each combo has P≈1/4 per slot-decision, so P(any combo missing) < 4×(3/4)^40 ≈ 4e-5.
    """

    @classmethod
    def setUpClass(cls):
        # Find a TC with at least one two-operand auth slot reached by the CE.
        for _ in range(300):
            pac, stage1_tc, fix_points = _run_stage1()

            # Quick probe: run CE once to see if any two-op slot is reached.
            combos_seen: list = []
            _fill_fixpoints_from_ce(
                stage1_tc, fix_points,
                _combo_observer=lambda wp, wc: combos_seen.append((wp, wc)))

            two_op_reached = any(
                len(fp.committed_inst.operands) > 1 and fp.correct_sig is not None
                for fp in fix_points)
            if two_op_reached:
                cls.pac        = pac
                cls.stage1_tc  = stage1_tc
                cls.fix_points = fix_points
                return

        raise unittest.SkipTest(
            "Could not find a TC with a two-operand auth slot reached by the CE")

    def test_all_four_combos_generated(self):
        """Run _fill_fixpoints_from_ce 40 times and assert all 4 combos appear."""
        combo_counts = {(False, False): 0, (False, True): 0,
                        (True,  False): 0, (True,  True): 0}
        N = 40

        for _ in range(N):
            for fp in self.fix_points:
                fp.reset()
            _fill_fixpoints_from_ce(
                self.stage1_tc, self.fix_points,
                _combo_observer=lambda wp, wc: combo_counts.__setitem__(
                    (wp, wc), combo_counts[(wp, wc)] + 1))

        total = sum(combo_counts.values())
        for combo, count in sorted(combo_counts.items()):
            with self.subTest(combo=combo):
                self.assertGreater(
                    count, 0,
                    f"combo {combo} never appeared in {N} runs "
                    f"({total} slot-decisions total)")


if __name__ == '__main__':
    unittest.main(verbosity=2)
