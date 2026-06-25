"""
PAC engine pipeline tests: TC generation → seal → simulated executor → baseline/decoy variants.

Each test works over randomly-seeded test cases.  The executor is simulated by
querying the kernel module for real PAC signatures for each PACFixPoint.

Invariants under test (sealing):
  - Every slot has exactly SLOT_SIZE instructions (fp.slot_locs)
  - Position SLOT_SIG_POS  = NOP
  - Position AUTH_SLOT_POS = XPAC (xpaci or xpacd)
  - committed_inst is an AUT* instruction
  - All slot_ids within a TC are unique

Invariants under test (variants):
  - Baseline and decoy have the same flat instruction count (layout preserved)
  - Slot flat indices are identical across variants
  - BASELINE: reached slot → [MOVK correct_sig, AUTH]; unreached → [NOP, XPAC]
  - DECOY arch slot (spec_nesting == 0): identical to the baseline slot (genuine)
  - DECOY spec slot (spec_nesting != 0): [MOVK alt_sig, AUTH]
  - AUTH mnemonic matches committed_inst

CE arch-path state identity (variant vs sealed TC):
  For every non-slotted arch-path (nesting==0) instruction in the sealed TC, the baseline, and the
  decoy, these fields are identical: x0-x5, SP, NZCV, and (for memory accesses) the access offset
  relative to the sandbox base (ea - x29), value-before, value-after.
  This holds because slot instructions only mutate value_reg and produce the same stripped address in
  every variant: arch slots authenticate correctly; spec slots take XPAC semantics in the CE.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # repo root: config.yml / base.json

from src.config import CONF
from src.isa_loader import InstructionSet
from src.interfaces import TestCase, Instruction, Input
from src.input_generator import NumpyRandomInputGenerator
from enum import Enum, auto

from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_seal import inst_at
from src.aarch64.aarch64_pac import (
    PACInstrumentation, PACFixPoint,
    SLOT_SIZE, SLOT_SIG_POS, AUTH_SLOT_POS,
    _AUTH_TO_PAC, _AUTH_TO_XPAC, _AUTH_TO_KEY, PACKey,
)


class V(Enum):
    """The two test cases the engine compares: the all-genuine baseline and a decoy instance."""
    BASELINE = auto()
    DECOY    = auto()
from src.aarch64.aarch64_printer import Aarch64Printer, Aarch64ASMLayout
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc
from src.aarch64.aarch64_contract_executor import (
    ContractExecution, ContractExecutionResult, ContractExecutorService, SimArch,
)
from src.aarch64.aarch64_input_layout import _reconstruct_pstate

_isa:          Optional[InstructionSet] = None
_executor      = None    # LocalHWExecutor — pac_sign() + sandbox_base
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
    os.path.dirname(__file__), "..", "..",
    "src/aarch64/contract_executor/contract_executor",
))


def setUpModule():
    global _isa, _executor, _ce, _target_desc, _input_gen, _tmpdir, _sandbox_base
    if not os.path.exists('/dev/executor'):
        raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
    try:
        from src.aarch64.aarch64_kernel import LocalHWExecutor, PacKeys
        CONF.load(os.path.join(_ROOT, "config.yml"))
        _isa          = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        _target_desc  = Aarch64TargetDesc()
        _executor     = LocalHWExecutor('/dev/executor', '/sys/executor')
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


def _slot(tc: TestCase, fp: PACFixPoint) -> List[Instruction]:
    """The slot's instructions in `tc`, located by position (fp.slot_locs)."""
    return [inst_at(tc, loc)[0] for loc in fp.slot_locs]


def _flat_slot_index(tc: TestCase, fix_points) -> Dict[int, Tuple[int, int]]:
    """flat index in _flat_insts(tc) -> (slot_id, pos) for slot instructions."""
    loc_to_slot = {loc: (fp.slot_id, pos)
                   for fp in fix_points for pos, loc in enumerate(fp.slot_locs)}
    result: Dict[int, Tuple[int, int]] = {}
    flat = 0
    for fi, func in enumerate(tc.functions):
        for bi, bb in enumerate(func):
            for ii, inst in enumerate(bb):
                if (fi, bi, ii) in loc_to_slot:
                    result[flat] = loc_to_slot[(fi, bi, ii)]
                flat += 1
    return result


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
    fix_points: List[PACFixPoint],
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
      spec_nesting stays None  (the engine treats None as speculative)

    _combo_observer: optional callable(wrong_ptr: bool, wrong_ctx: bool) invoked once per
                     alt_sig decision.  Intended for tests that count combo coverage.

    Returns the input used to run the CE so callers can reuse it.
    """
    # Map byte offset of XPAC placeholder → PACFixPoint (same as production executor)
    layout = Aarch64ASMLayout(stage1_tc)
    xpac_off_to_fp: Dict[int, PACFixPoint] = {
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
                value_reg = fp.committed_inst.operands[0].value
                ctx_reg = fp.committed_inst.operands[1].value if has_ctx else None

                ptr = _read_gpr(ite.cpu, value_reg)
                ctx = _read_gpr(ite.cpu, ctx_reg) if has_ctx else 0
                fp.correct_sig = _pac_sign(ptr, ctx, auth_mn)

                # alt_sig: substitute the ptr or ctx with a value from a different GPR.
                # Force bits[63:48]=0xFFFF so alt_ptr is a canonical kernel VA: PACDA then
                # sets bit 55 in the signed pointer, and XPAC can correctly sign-extend it
                # back to 0xffff... instead of producing a non-canonical 0x0000... address.
                use_wrong_ptr = random.choice([True, False])
                use_wrong_ctx = has_ctx and random.choice([True, False])
                alt_ptr = _other_gpr(ite.cpu, value_reg, ctx_reg) if use_wrong_ptr else ptr
                alt_ptr = (alt_ptr & 0x0000FFFFFFFFFFFF) | 0xFFFF000000000000
                alt_ctx = (_other_gpr(ite.cpu, ctx_reg, value_reg) if use_wrong_ctx else ctx) if has_ctx else 0
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


def _run_stage1(max_attempts: int = 500) -> Tuple[PACInstrumentation, TestCase, List[PACFixPoint]]:
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
        stage1_tc, fix_points = pac.seal_test_case(copy.deepcopy(tc))
        if fix_points:
            return pac, stage1_tc, fix_points
    raise RuntimeError(f"No fix_points found after {max_attempts} random seeds")


def _run_pipeline() -> Tuple[TestCase, List[PACFixPoint], Dict, Input]:
    """Full pipeline: seal → CE trace → fix_point fill → engine variants. Always returns a result.

    Returns (sealed_tc, fix_points, variants, ref_input) where variants is
    {V.BASELINE: engine.baseline(), V.DECOY: a decoy instance} and ref_input is the input used to
    classify spec_nesting in _fill_fixpoints_from_ce.  Reuse ref_input when running the CE over the
    variants so the spec-vs-arch paths are identical across all test cases.
    """
    pac, stage1_tc, fix_points = _run_stage1()
    ref_input = _fill_fixpoints_from_ce(stage1_tc, fix_points)
    engine = pac.make_engine()
    engine.set_sealed(stage1_tc, fix_points)
    variants = {V.BASELINE: engine.baseline(), V.DECOY: next(engine.decoys(random))}
    return stage1_tc, fix_points, variants, ref_input


def _collect_cases(n: int = 20) -> list:
    return [_run_pipeline() for _ in range(n)]


# ===========================================================================
# CE execution helpers
# ===========================================================================

def _tc_to_bytes(tc: TestCase) -> bytes:
    layout   = Aarch64ASMLayout(tc)
    assembly = Aarch64Printer(_target_desc).print_layout(layout)
    return Aarch64RandomGenerator.in_memory_assemble(assembly)


def _slot_offsets(tc: TestCase, fix_points) -> Set[int]:
    """Byte offsets (from TC start) of every slotted instruction."""
    layout = Aarch64ASMLayout(tc)
    return {layout.instruction_address[inst_at(tc, loc)[0]]
            for fp in fix_points for loc in fp.slot_locs}


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
            for fp in fix_points:
                self.assertEqual(len(fp.slot_locs), SLOT_SIZE,
                                 f"slot {fp.slot_id}: expected {SLOT_SIZE}, got {len(fp.slot_locs)}")

    def test_slot_sig_pos_is_nop(self):
        for stage1_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertEqual(_slot(stage1_tc, fp)[SLOT_SIG_POS].name, 'nop')

    def test_auth_slot_pos_is_xpac(self):
        for stage1_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertIn(_slot(stage1_tc, fp)[AUTH_SLOT_POS].name, _XPAC_NAMES)

    def test_committed_inst_is_auth_mnemonic(self):
        for _, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertIn(fp.committed_inst.name.lower(), _AUTH_NAMES)

    def test_xpac_family_matches_committed_inst_key(self):
        for stage1_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                key = _AUTH_TO_KEY[fp.committed_inst.name.lower()]
                xpac_mn = _slot(stage1_tc, fp)[AUTH_SLOT_POS].name
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
        for stage1_tc, fix_points, variants, _inp in self.cases:
            ref_idx = {flat: key for flat, key in _flat_slot_index(stage1_tc, fix_points).items()}
            for variant, tc in variants.items():
                self.assertEqual(_flat_slot_index(tc, fix_points), ref_idx,
                                 f"{variant.name}: slot flat indices moved")

    def test_non_slot_instructions_unchanged(self):
        for stage1_tc, fix_points, variants, _inp in self.cases:
            ref_insts = _flat_insts(stage1_tc)
            slot_flat = _flat_slot_index(stage1_tc, fix_points)
            for variant, tc in variants.items():
                tc_insts = _flat_insts(tc)
                for idx, ref in enumerate(ref_insts):
                    if idx in slot_flat:
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

    def test_baseline_genuine_slots(self):
        """Baseline: reached slots carry [MOVK correct_sig, AUTH]; unreached fall back to [NOP, XPAC]."""
        for _, fix_points, variants, _inp in self.cases:
            base = variants[V.BASELINE]
            for fp in fix_points:
                auth_mn = fp.committed_inst.name.lower()
                s = _slot(base, fp)
                with self.subTest(slot=fp.slot_id):
                    if fp.correct_sig is None:
                        self.assertEqual(s[SLOT_SIG_POS].name, 'nop')
                        self.assertEqual(s[AUTH_SLOT_POS].name, _AUTH_TO_XPAC[auth_mn])
                    else:
                        self.assertEqual(s[SLOT_SIG_POS].name, 'movk')
                        self.assertEqual(_sig_from_movk(s[SLOT_SIG_POS]), fp.correct_sig)
                        self.assertEqual(s[AUTH_SLOT_POS].name, auth_mn)

    def test_decoy_arch_slot_identical_to_baseline(self):
        """An architectural slot (spec_nesting == 0) is genuine in the decoy too — the NI invariant."""
        for _, fix_points, variants, _inp in self.cases:
            base = variants[V.BASELINE]
            decoy = variants[V.DECOY]
            for fp in fix_points:
                if fp.spec_nesting != 0:
                    continue
                for pos, (ib, idd) in enumerate(zip(_slot(base, fp), _slot(decoy, fp))):
                    with self.subTest(slot=fp.slot_id, pos=pos):
                        self.assertEqual(ib.name,     idd.name)
                        self.assertEqual(ib.template, idd.template)

    def test_decoy_spec_slot_is_wrong_sig_or_strip(self):
        """A speculative decoy slot either authenticates a wrong signature ([MOVK alt_sig, AUT*])
        or strips the auth ([NOP, XPAC])."""
        for _, fix_points, variants, _inp in self.cases:
            decoy = variants[V.DECOY]
            for fp in fix_points:
                if fp.spec_nesting == 0:
                    continue
                s = _slot(decoy, fp)
                auth_mn = fp.committed_inst.name.lower()
                with self.subTest(slot=fp.slot_id):
                    if s[AUTH_SLOT_POS].name == auth_mn:                    # wrong-signature auth
                        self.assertEqual(s[SLOT_SIG_POS].name, 'movk')
                        self.assertEqual(_sig_from_movk(s[SLOT_SIG_POS]), fp.alt_sig)
                    else:                                                   # strip
                        self.assertEqual(s[SLOT_SIG_POS].name, 'nop')
                        self.assertEqual(s[AUTH_SLOT_POS].name, _AUTH_TO_XPAC[auth_mn])


# ===========================================================================
# CE-never-reached path (correct_sig=None)
# ===========================================================================

class TestCENeverReached(unittest.TestCase):
    """A slot the CE never reached (correct_sig=None, spec_nesting=None) is treated as speculative:
    the baseline falls back to the strip placeholder, the decoy still gets alt_sig."""

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        while len(cls.cases) < 5:
            pac, stage1_tc, fix_points = _run_stage1()
            _fill_fixpoints_from_ce(stage1_tc, fix_points)   # populates alt_sig
            for fp in fix_points:
                fp.correct_sig = None      # force never-reached
                fp.spec_nesting = None     # ...consistently: a slot CE never reached has no depth
            engine = pac.make_engine()
            engine.set_sealed(stage1_tc, fix_points)
            variants = {V.BASELINE: engine.baseline(), V.DECOY: next(engine.decoys(random))}
            cls.cases.append((stage1_tc, fix_points, variants))

    def test_baseline_falls_back_to_xpac(self):
        for _, fix_points, variants in self.cases:
            base = variants[V.BASELINE]
            for fp in fix_points:
                s = _slot(base, fp)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[SLOT_SIG_POS].name, 'nop')
                    self.assertIn(s[AUTH_SLOT_POS].name, _XPAC_NAMES)

    def test_decoy_is_decoyed_even_when_unreached(self):
        """An unreached slot is decoyed: a wrong-signature auth (alt_sig) or a strip (XPAC)."""
        for _, fix_points, variants in self.cases:
            decoy = variants[V.DECOY]
            for fp in fix_points:
                s = _slot(decoy, fp)
                auth_mn = fp.committed_inst.name.lower()
                with self.subTest(slot=fp.slot_id):
                    if s[AUTH_SLOT_POS].name in _AUTH_NAMES:                # wrong-signature auth
                        self.assertEqual(s[SLOT_SIG_POS].name, 'movk')
                        self.assertEqual(_sig_from_movk(s[SLOT_SIG_POS]), fp.alt_sig)
                    else:                                                   # strip
                        self.assertEqual(s[SLOT_SIG_POS].name, 'nop')
                        self.assertEqual(s[AUTH_SLOT_POS].name, _AUTH_TO_XPAC[auth_mn])


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

    def _ref_and_variant(self, stage1_tc, fix_points, tc, inp):
        slot_offs   = _slot_offsets(stage1_tc, fix_points)
        ref_entries = _arch_state_entries(_run_ce(_tc_to_bytes(stage1_tc), inp), slot_offs)
        cmp_entries = _arch_state_entries(_run_ce(_tc_to_bytes(tc), inp), slot_offs)
        return ref_entries, cmp_entries

    def test_stage1_vs_baseline(self):
        """Baseline uses correct signatures → same stripped ptr → same arch state as the sealed TC."""
        for stage1_tc, fix_points, variants, ref_input in self.cases:
            ref, cmp = self._ref_and_variant(
                stage1_tc, fix_points, variants[V.BASELINE], ref_input)
            self._compare_vs_stage1(ref, cmp, "BASELINE")

    def test_stage1_vs_decoy(self):
        """Decoy: arch slots are genuine, spec slots carry alt_sig but only on speculative paths →
        the architectural state still matches the sealed TC (the NI invariant).

        Must reuse the input from _fill_fixpoints_from_ce so slots classified speculative
        (spec_nesting > 0) stay at spec depth; otherwise a once-speculative slot could run the arch
        path with alt_sig → auth failure → poisoned pointer → SIGSEGV.
        """
        for stage1_tc, fix_points, variants, ref_input in self.cases:
            ref, cmp = self._ref_and_variant(
                stage1_tc, fix_points, variants[V.DECOY], ref_input)
            self._compare_vs_stage1(ref, cmp, "DECOY")


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
