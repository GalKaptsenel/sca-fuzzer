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
  - DECOY speculative slot (spec_nesting != 0): [MOVK alt_sig, AUTH] — reached slots use a
    PAC-mask-verified pool, unreached slots a random pool
  - AUTH mnemonic matches committed_inst

Safety: a failed AUTH at EL1 (e.g. the contract executor authenticating a forged sig) resets the
box on FEAT_FPAC hardware. A forged TC is run through the CE only after _assert_no_arch_forgery
guarantees every forged AUTH is on a speculative slot (XPAC-stripped by the CE, never run at arch),
and the classification input is reused so arch/spec paths match the sealing run.

CE arch-path state identity (variant vs sealed TC):
  For every non-slotted arch-path (nesting==0) instruction the variant and the sealed TC agree on
  x0-x5, SP, NZCV, and (for memory accesses) the access offset relative to the sandbox base
  (ea - x29), value-before, value-after — a correct signature authenticates to the same canonical
  address the XPAC placeholder strips to, and a decoy only ever rewrites speculative slots.
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


def _pac_field_mask16(auth_mn: str, samples: int) -> int:
    """Top-16 mask of the PAC field (the bits SIGN sets == the bits AUTH checks); mirrors the
    executor's _pac_field_mask16. SIGN only — never AUTH, never XPAC."""
    pac_mn = _AUTH_TO_PAC[auth_mn]
    mask = 0
    for _ in range(samples):
        v = random.randrange(1 << 48)               # clean low-half address: top 16 bits zero
        mask |= _executor.pac_sign(v, random.randrange(1 << 64), pac_mn) ^ v
    mask16 = (mask >> 48) & 0xFFFF
    assert mask16 and not (mask16 & 0x80), f"implausible PAC-field mask 0x{mask16:04x}"
    return mask16


def _build_alt_sigs(cpu, auth_mn: str, correct_sig: int, ctx: int, from_regs: List[str],
                    size: int, tries: int) -> List[int]:
    """Distinct wrong top-16 signatures, each differing from correct_sig within the PAC-field bits
    (so AUTH provably fails); mirrors the executor's _build_alt_sigs. Candidates are signatures of
    live values from from_regs (high prob) or a random value (low prob)."""
    pac_mn = _AUTH_TO_PAC[auth_mn]
    mask16 = _pac_field_mask16(auth_mn, samples=64)
    pool: List[int] = []
    for _ in range(tries):
        if from_regs and random.random() < 0.85:
            value = _read_gpr(cpu, random.choice(from_regs))
        else:
            value = random.randrange(1 << 64)
        cand = (_executor.pac_sign(value, ctx, pac_mn) >> 48) & 0xFFFF
        sig = (correct_sig & ~mask16) | (cand & mask16)   # flip only PAC bits; keep non-PAC bits
        if sig != correct_sig and sig not in pool:
            pool.append(sig)
            if len(pool) >= size:
                break
    assert pool, f"no effective PAC forgery in {tries} tries (pac_field_mask=0x{mask16:04x})"
    return pool


def _random_alt_sigs(auth_mn: str, size: int) -> List[int]:
    """Distinct wrong signatures from random values, for slots the CE never reached (no correct_sig
    to verify against); mirrors the executor's _fill_missing_alt_sigs. SIGN only."""
    pac_mn = _AUTH_TO_PAC[auth_mn]
    sigs: Set[int] = set()
    while len(sigs) < size:
        sigs.add((_executor.pac_sign(random.randrange(1 << 64), random.randrange(1 << 64), pac_mn) >> 48) & 0xFFFF)
    return list(sigs)


def _fill_fixpoints_from_ce(
    sealed_tc: TestCase,
    fix_points: List[PACFixPoint],
) -> Input:
    """Run sealed_tc through the real CE and populate fix_point metadata from the trace.

    For each XPAC slot reached by the CE:
      correct_sig  — pac_sign(actual ptr, actual ctx at slot PC)
      alt_sigs     — verified wrong-sig pool (each differs from correct_sig in the PAC-field bits)
      spec_nesting — actual CE nesting depth (min over all hits)

    For slots the CE never reached:
      correct_sig  stays None
      alt_sigs     random wrong sigs (no committed value to verify against; the slot never executes
                   on the contract path, so they need no verification)
      spec_nesting stays None (the engine treats None as speculative)

    Mirrors the executor: SIGN only, never AUTH. Returns the input used to run the CE.
    """
    # Map byte offset of XPAC placeholder → PACFixPoint (same as production executor)
    layout = Aarch64ASMLayout(sealed_tc)
    xpac_off_to_fp: Dict[int, PACFixPoint] = {
        layout.instruction_address[fp.slot_insts[AUTH_SLOT_POS]]: fp
        for fp in fix_points
    }

    inp = _random_input()
    cer = _run_ce(_tc_to_bytes(sealed_tc), inp)

    if len(cer) > 0:
        code_base = cer[0].cpu.pc
        for ite in cer:
            rel = ite.cpu.pc - code_base
            if rel not in xpac_off_to_fp:
                continue
            fp    = xpac_off_to_fp[rel]
            depth = ite.metadata.speculation_nesting

            # Track shallowest nesting (prefer arch path); the arch hit overrides a prior spec one.
            if fp.spec_nesting is None or depth < fp.spec_nesting:
                fp.spec_nesting = depth
            if fp.correct_sig is not None and depth != 0:
                continue

            auth_mn = fp.committed_inst.name.lower()
            has_ctx = len(fp.committed_inst.operands) > 1
            value_reg = fp.committed_inst.operands[0].value
            ctx_reg = fp.committed_inst.operands[1].value if has_ctx else None

            ptr = _read_gpr(ite.cpu, value_reg)
            ctx = _read_gpr(ite.cpu, ctx_reg) if has_ctx else 0
            fp.correct_sig = _pac_sign(ptr, ctx, auth_mn)

            # from-set: every generatable register except the committed value register.
            from_regs = [r for r in _target_desc.registers[64] if r != value_reg]
            fp.alt_sigs = _build_alt_sigs(ite.cpu, auth_mn, fp.correct_sig, ctx, from_regs,
                                          size=6, tries=64)

    for fp in fix_points:                       # slots the CE never reached: random wrong sigs
        if not fp.alt_sigs:
            fp.alt_sigs = _random_alt_sigs(fp.committed_inst.name.lower(), 6)

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
        sealed_tc, fix_points = pac.seal_test_case(copy.deepcopy(tc))
        if fix_points:
            return pac, sealed_tc, fix_points
    raise RuntimeError(f"No fix_points found after {max_attempts} random seeds")


def _run_pipeline() -> Tuple[TestCase, List[PACFixPoint], Dict, Input]:
    """Full pipeline: seal → CE trace → fix_point fill → engine variants. Always returns a result.

    Returns (sealed_tc, fix_points, variants, ref_input) where variants is
    {V.BASELINE: engine.baseline(), V.DECOY: a decoy instance} and ref_input is the input used to
    classify spec_nesting in _fill_fixpoints_from_ce.  Reuse ref_input when running the CE over the
    variants so the spec-vs-arch paths are identical across all test cases.
    """
    pac, sealed_tc, fix_points = _run_stage1()
    ref_input = _fill_fixpoints_from_ce(sealed_tc, fix_points)
    engine = pac.make_engine()
    engine.set_sealed(sealed_tc, fix_points)
    variants = {V.BASELINE: engine.baseline(), V.DECOY: next(engine.decoys(random))}
    return sealed_tc, fix_points, variants, ref_input


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


def _assert_no_arch_forgery(tc: TestCase, fix_points) -> None:
    """Guarantee a forged TC is safe to run through the CE: any forged slot (MOVK sig != correct_sig)
    must be speculative (spec_nesting != 0), so its AUTH only ever executes under misspeculation
    (XPAC-stripped by the CE), never architecturally (which would fault EL1 on FEAT_FPAC)."""
    for fp in fix_points:
        s = _slot(tc, fp)
        if s[SLOT_SIG_POS].name != 'movk' or _sig_from_movk(s[SLOT_SIG_POS]) == fp.correct_sig:
            continue                                  # strip placeholder or a genuine slot
        assert fp.spec_nesting != 0, \
            f"slot {fp.slot_id}: forged AUTH on an architectural slot (spec_nesting=0)"


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


def _trace_state_entries(cer: ContractExecutionResult,
                         slot_offs: Set[int]) -> List[Dict[str, Any]]:
    """Extract every non-slot trace entry, architectural and speculative, with its state.

    Each entry dict contains:
      rel_pc, encoding, nesting,
      x0_x5 (tuple of 6 values), sp, nzcv,
      and if has_memory_access: mem_offset (ea - x29), mem_before, mem_after.
    The comparison uses the address (mem_offset) + control flow on every path, but the full
    register/memory-value state only on the architectural path — on speculative paths the
    authenticated pointer's top byte legitimately differs between variants (XPAC keeps it) and is
    not contract-observable (the sandbox masks it before any access).
    """
    if len(cer) == 0:
        return []
    code_base = cer[0].cpu.pc
    result = []
    for ite in cer:
        rel = ite.cpu.pc - code_base
        if rel in slot_offs:
            continue
        entry: Dict[str, Any] = {
            'rel_pc':   rel,
            'encoding': ite.cpu.encoding,
            'nesting':  ite.metadata.speculation_nesting,
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
        for sealed_tc, fix_points, _, _inp in self.cases:
            ids = [fp.slot_id for fp in fix_points]
            self.assertEqual(len(ids), len(set(ids)))

    def test_each_slot_has_exactly_slot_size_instructions(self):
        for sealed_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertEqual(len(fp.slot_locs), SLOT_SIZE,
                                 f"slot {fp.slot_id}: expected {SLOT_SIZE}, got {len(fp.slot_locs)}")

    def test_slot_sig_pos_is_nop(self):
        for sealed_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertEqual(_slot(sealed_tc, fp)[SLOT_SIG_POS].name, 'nop')

    def test_auth_slot_pos_is_xpac(self):
        for sealed_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertIn(_slot(sealed_tc, fp)[AUTH_SLOT_POS].name, _XPAC_NAMES)

    def test_committed_inst_is_auth_mnemonic(self):
        for _, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                self.assertIn(fp.committed_inst.name.lower(), _AUTH_NAMES)

    def test_xpac_family_matches_committed_inst_key(self):
        for sealed_tc, fix_points, _, _inp in self.cases:
            for fp in fix_points:
                key = _AUTH_TO_KEY[fp.committed_inst.name.lower()]
                xpac_mn = _slot(sealed_tc, fp)[AUTH_SLOT_POS].name
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
        for sealed_tc, _, variants, _inp in self.cases:
            ref = len(_flat_insts(sealed_tc))
            for variant, tc in variants.items():
                self.assertEqual(len(_flat_insts(tc)), ref,
                                 f"{variant.name}: instruction count changed")

    def test_slot_flat_indices_identical_across_variants(self):
        for sealed_tc, fix_points, variants, _inp in self.cases:
            ref_idx = {flat: key for flat, key in _flat_slot_index(sealed_tc, fix_points).items()}
            for variant, tc in variants.items():
                self.assertEqual(_flat_slot_index(tc, fix_points), ref_idx,
                                 f"{variant.name}: slot flat indices moved")

    def test_non_slot_instructions_unchanged(self):
        for sealed_tc, fix_points, variants, _inp in self.cases:
            ref_insts = _flat_insts(sealed_tc)
            slot_flat = _flat_slot_index(sealed_tc, fix_points)
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

    def test_decoy_speculative_slot_is_wrong_sig(self):
        """Every speculative decoy slot (spec_nesting != 0, incl. unreached) authenticates a wrong
        signature from its pool ([MOVK alt_sig, AUT*]); only arch slots (==0) stay genuine."""
        for _, fix_points, variants, _inp in self.cases:
            decoy = variants[V.DECOY]
            for fp in fix_points:
                if fp.spec_nesting == 0:                       # arch slot -> genuine, not forged
                    continue
                s = _slot(decoy, fp)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[SLOT_SIG_POS].name, 'movk')
                    self.assertEqual(s[AUTH_SLOT_POS].name, fp.committed_inst.name.lower())
                    self.assertIn(_sig_from_movk(s[SLOT_SIG_POS]), fp.alt_sigs)


# ===========================================================================
# CE-never-reached path (correct_sig=None)
# ===========================================================================

class TestCENeverReached(unittest.TestCase):
    """A slot the CE never reached (correct_sig=None, spec_nesting=None): the baseline falls back to
    the strip placeholder (no committed signature), while the decoy still forges with a random pool.
    Such a slot never executes on the contract path, so the forged AUTH is harmless there."""

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        while len(cls.cases) < 5:
            pac, sealed_tc, fix_points = _run_stage1()
            _fill_fixpoints_from_ce(sealed_tc, fix_points)
            for fp in fix_points:
                fp.correct_sig = None      # force never-reached: no committed sig, no depth
                fp.alt_sigs = _random_alt_sigs(fp.committed_inst.name.lower(), 6)
                fp.spec_nesting = None
            engine = pac.make_engine()
            engine.set_sealed(sealed_tc, fix_points)
            variants = {V.BASELINE: engine.baseline(), V.DECOY: next(engine.decoys(random))}
            cls.cases.append((sealed_tc, fix_points, variants))

    def test_baseline_falls_back_to_xpac(self):
        for _, fix_points, variants in self.cases:
            base = variants[V.BASELINE]
            for fp in fix_points:
                s = _slot(base, fp)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[SLOT_SIG_POS].name, 'nop')
                    self.assertIn(s[AUTH_SLOT_POS].name, _XPAC_NAMES)

    def test_decoy_unreached_is_forged_with_random_sig(self):
        """An unreached slot is forged with a random pool signature ([MOVK alt, AUT*])."""
        for _, fix_points, variants in self.cases:
            decoy = variants[V.DECOY]
            for fp in fix_points:
                s = _slot(decoy, fp)
                with self.subTest(slot=fp.slot_id):
                    self.assertEqual(s[SLOT_SIG_POS].name, 'movk')
                    self.assertEqual(s[AUTH_SLOT_POS].name, fp.committed_inst.name.lower())
                    self.assertIn(_sig_from_movk(s[SLOT_SIG_POS]), fp.alt_sigs)


# ===========================================================================
# CE full-trace state identity: sealed TC vs each variant
# ===========================================================================

class TestCEStateIdentity(unittest.TestCase):
    """Across the FULL non-slot trace (architectural and speculative) a variant must match the
    sealed TC on:
      - control flow: rel_pc, encoding, nesting        (every path)
      - data address: mem_offset (ea - x29)            (every path; sandbox-masked, so observable)
      - full state: x0-x5, SP, NZCV, mem before/after  (architectural path only — speculative
        register/value differences in the authenticated pointer's top byte are not observable)

    This is the contract's no-leak prediction: under the COND contract the CE XPAC-strips
    speculative AUTH, so baseline and decoy are indistinguishable; any hardware difference is then a
    true violation. Running a decoy through the CE is safe — _assert_no_arch_forgery guarantees every
    forged AUTH is speculative-only (XPAC-stripped, never run at arch, which would fault EL1 on
    FEAT_FPAC), and the classification input is reused so arch/spec paths match the sealing run.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = _collect_cases(10)

    def _assert_trace_identical(self, sealed: List[Dict], variant: List[Dict], label: str) -> None:
        self.assertEqual(len(variant), len(sealed),
                         f"{label}: trace length differs (sealed {len(sealed)}, got {len(variant)})")
        for i, (s, v) in enumerate(zip(sealed, variant)):
            with self.subTest(variant=label, entry=i):
                self.assertEqual(v['rel_pc'],   s['rel_pc'],   f"entry {i}: rel_pc mismatch")
                self.assertEqual(v['encoding'], s['encoding'], f"entry {i}: encoding mismatch")
                self.assertEqual(v['nesting'],  s['nesting'],  f"entry {i}: nesting mismatch")
                if 'mem_offset' in s:
                    self.assertIn('mem_offset', v, f"entry {i}: memory access missing in {label}")
                    self.assertEqual(v['mem_offset'], s['mem_offset'],
                                     f"entry {i}: memory offset (ea-x29) mismatch")
                if s['nesting'] == 0:                  # full state only on the architectural path
                    self.assertEqual(v['x0_x5'], s['x0_x5'], f"entry {i}: x0-x5 mismatch")
                    self.assertEqual(v['sp'],    s['sp'],    f"entry {i}: SP mismatch")
                    self.assertEqual(v['nzcv'],  s['nzcv'],  f"entry {i}: NZCV mismatch")
                    if 'mem_offset' in s:
                        self.assertEqual(v['mem_before'], s['mem_before'],
                                         f"entry {i}: memory value-before mismatch")
                        self.assertEqual(v['mem_after'], s['mem_after'],
                                         f"entry {i}: memory value-after mismatch")

    def _sealed_and_variant_trace(self, sealed_tc, fix_points, variant_tc, inp):
        slot_offs = _slot_offsets(sealed_tc, fix_points)
        sealed  = _trace_state_entries(_run_ce(_tc_to_bytes(sealed_tc), inp), slot_offs)
        variant = _trace_state_entries(_run_ce(_tc_to_bytes(variant_tc), inp), slot_offs)
        return sealed, variant

    def test_baseline_matches_sealed(self):
        """Baseline uses correct signatures → authenticates to the same canonical pointer the XPAC
        placeholder strips to → identical full trace as the sealed TC."""
        for sealed_tc, fix_points, variants, ref_input in self.cases:
            sealed, variant = self._sealed_and_variant_trace(
                sealed_tc, fix_points, variants[V.BASELINE], ref_input)
            self._assert_trace_identical(sealed, variant, "BASELINE")

    def test_decoy_matches_sealed(self):
        """A decoy is indistinguishable from the sealed TC under the contract (the no-leak
        prediction): forged AUTHs run only speculatively, where the CE XPAC-strips them."""
        for sealed_tc, fix_points, variants, ref_input in self.cases:
            _assert_no_arch_forgery(variants[V.DECOY], fix_points)
            sealed, variant = self._sealed_and_variant_trace(
                sealed_tc, fix_points, variants[V.DECOY], ref_input)
            self._assert_trace_identical(sealed, variant, "DECOY")

# ===========================================================================
# Verified wrong-signature pool
# ===========================================================================

class TestAltSigPool(unittest.TestCase):
    """The decoy's wrong-signature pool: every reached fix point gets a non-empty pool of DISTINCT
    signatures, each differing from correct_sig within the PAC-field bits (so AUTH provably fails —
    never an accidental no-op), and successive decoy instances draw FRESH signatures from it."""

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        while len(cls.cases) < 6:
            pac, sealed_tc, fix_points = _run_stage1()
            _fill_fixpoints_from_ce(sealed_tc, fix_points)
            cls.cases.append((pac, sealed_tc, fix_points))

    def test_pool_sigs_distinct_and_fail_auth_by_mask(self):
        """Every slot's pool is non-empty and distinct. For reached slots each signature differs
        from correct_sig within the PAC-field mask — i.e. it provably fails AUTH on this hardware."""
        for _pac, _tc, fix_points in self.cases:
            for fp in fix_points:
                with self.subTest(slot=fp.slot_id):
                    self.assertTrue(fp.alt_sigs, "every slot must have a non-empty pool")
                    self.assertEqual(len(fp.alt_sigs), len(set(fp.alt_sigs)), "pool must be distinct")
                    if fp.correct_sig is None:               # unreached -> random pool (no mask check)
                        continue
                    mask16 = _pac_field_mask16(fp.committed_inst.name.lower(), samples=64)
                    for s in fp.alt_sigs:
                        self.assertTrue((s ^ fp.correct_sig) & mask16,
                                        f"sig 0x{s:04x} does not differ from correct in PAC bits")

    def test_decoys_draw_fresh_wrong_sigs_from_pool(self):
        """Every speculative decoy authenticates a pool signature, and a slot with a multi-entry
        pool yields more than one distinct forgery over many instances (per-decoy freshness)."""
        for pac, sealed_tc, fix_points in self.cases:
            engine = pac.make_engine(); engine.set_sealed(sealed_tc, fix_points)
            seen = {fp.slot_id: set() for fp in fix_points}
            gen = engine.decoys(random.Random(0))
            for _ in range(40):
                decoy = next(gen)
                for fp in fix_points:
                    if fp.spec_nesting == 0:               # arch slot -> genuine, not forged
                        continue
                    s = _slot(decoy, fp)
                    self.assertEqual(s[SLOT_SIG_POS].name, 'movk')   # always a wrong-sig auth
                    sig = _sig_from_movk(s[SLOT_SIG_POS])
                    self.assertIn(sig, fp.alt_sigs)
                    seen[fp.slot_id].add(sig)
            for fp in fix_points:
                if len(fp.alt_sigs) > 1 and seen[fp.slot_id]:
                    with self.subTest(slot=fp.slot_id):
                        self.assertGreater(len(seen[fp.slot_id]), 1,
                                           "successive decoys should draw different pool signatures")


if __name__ == '__main__':
    unittest.main(verbosity=2)
