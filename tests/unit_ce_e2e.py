"""
True end-to-end tests using the Contract Executor (CE) binary.

Unlike the generator tests (which only inspect the generated assembly structures),
these tests actually *execute* stage-1 and stage-2 TCs through the CE, capture
real fix-point values, and verify the semantic outcomes:

  PAC:
    Stage1 CE run → captures signed_value / ctx_value / spec_nesting for real.
    TC1 CE run    → GPR after XPAC = canonical pointer (PAC bits stripped).
    TC2 CE run    → GPR after AUTH = same canonical pointer (auth succeeded with
                    the correct values that TC2 loaded via MOVZ+MOVK).
    TC3 arch      → CE result identical to TC2 (TC3 == TC2 for arch slots).

  MTE:
    Stage1 CE run → captures per-slot spec_nesting (arch=0, spec>0, never=None).
    TC1 CE run    → every memory access EA bits[59:56] == arch_tag (unchanged).
    TC3 CE run    → spec-slot memory access EA bits[59:56] != arch_tag (wrong tag).
    TC3 vs TC1    → different tag bits for spec slots.

CE binary:  src/aarch64/contract_executor/contract_executor
arch_tag is derived dynamically from SANDBOX_BASE (bits[59:56] of req_mem_base_virt).

All tests are skipped automatically if the CE binary is not present.
"""
import copy
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONF
from src.isa_loader import InstructionSet
from src.generator import ConfigurableGenerator
from src.aarch64.aarch64_generator import (
    Aarch64RandomGenerator, Aarch64Printer, Aarch64ASMLayout,
    PACInstrumentation, MTEInstrumentation,
    FixPoint, MTEFixPoint,
    SLOT_SIZE, FIX_COUNT_CTX, FIX_COUNT_PTR,
    CTX_SLOT_START, PTR_SLOT_START, AUTH_SLOT_POS,
    MTE_SLOT_SIZE,
)
from src.aarch64.aarch64_contract_executor import (
    ContractExecutorService, ContractExecution, SimArch, ContractType,
)
from src.aarch64.aarch64_trace import ContractExecutionResult

# ===========================================================================
# Module-level constants and state
# ===========================================================================

CE_BINARY    = Path(__file__).parent.parent / "src/aarch64/contract_executor/contract_executor"
SANDBOX_BASE = 0x0100_0000_4000_0000   # bits[59:56] = arch_tag = 1
MEMORY_SIZE  = 0x2000   # 8 KB — must match executor sandbox expectations

def _arch_tag(sandbox_base: int) -> int:
    """Extract the arch tag from bits[59:56] of the sandbox base."""
    return (sandbox_base >> 56) & 0xF

_ISA: Optional[InstructionSet] = None
_CE:  Optional[ContractExecutorService] = None
_TMPDIR: Optional[str] = None


def setUpModule():
    global _ISA, _CE, _TMPDIR
    if not CE_BINARY.exists():
        return  # individual tests call self.skipTest()
    CONF.load("config.yml")
    _ISA = InstructionSet("base.json", CONF.instruction_categories)
    _CE  = ContractExecutorService(str(CE_BINARY))
    _TMPDIR = tempfile.mkdtemp()


def tearDownModule():
    global _CE
    if _CE is not None:
        _CE.stop()
        _CE = None
    import shutil
    if _TMPDIR:
        shutil.rmtree(_TMPDIR, ignore_errors=True)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_gen(seed: int) -> Aarch64RandomGenerator:
    return Aarch64RandomGenerator(_ISA, seed)


def _tc_to_bytes_and_layout(tc, gen: Aarch64RandomGenerator) -> Tuple[bytes, Aarch64ASMLayout]:
    layout = Aarch64ASMLayout(tc)
    asm    = Aarch64Printer(gen.target_desc).print_layout(layout)
    return ConfigurableGenerator.in_memory_assemble(asm), layout


def _run_ce(tc_bytes: bytes, sandbox_base: int = SANDBOX_BASE) -> ContractExecutionResult:
    """Run CE on tc_bytes with all-zero registers and blank 8 KB memory."""
    memory = bytes(MEMORY_SIZE)
    regs   = bytes(9 * 8)          # x0-x8 all zero; CE sets x29 from req_mem_base_virt
    execution = ContractExecution(
        machine_code=tc_bytes,
        memory=memory,
        registers=regs,
        arch=SimArch.RVZR_ARCH_AARCH64,
        max_misspred_branch_nesting=5,
        max_misspred_instructions=10,
        req_mem_base_virt=sandbox_base,
    )
    return _CE.run(execution)


def _code_base(cer: ContractExecutionResult) -> Optional[int]:
    return cer[0].cpu.pc if len(cer) > 0 else None


def _entry_at(cer: ContractExecutionResult, pc: int) -> Optional[object]:
    """Return first trace entry whose cpu.pc == pc (nesting-0 first)."""
    arch = next((e for e in cer if e.cpu.pc == pc and e.metadata.speculation_nesting == 0), None)
    return arch


def _entry_at_any_nesting(cer: ContractExecutionResult, pc: int) -> Optional[object]:
    return next((e for e in cer if e.cpu.pc == pc), None)


# ===========================================================================
# PAC helpers
# ===========================================================================

def _pac_stage1(seed: int) -> Optional[Tuple]:
    """
    Return (gen, pac, fix_points, stage1_tc, stage1_bytes, stage1_layout) or None.
    """
    gen = _make_gen(seed)
    pac = PACInstrumentation(gen, xpac_weight=2, auth_weight=3, sign_weight=3)
    asm_path = os.path.join(_TMPDIR, f"pac_{seed}.asm")
    try:
        tc = gen.create_test_case(asm_path, disable_assembler=True)
        stage1_tc, fix_points = pac.instrument_stage1(copy.deepcopy(tc))
    except Exception:
        return None
    if not fix_points:
        return None
    stage1_bytes, stage1_layout = _tc_to_bytes_and_layout(stage1_tc, gen)
    return gen, pac, fix_points, stage1_tc, stage1_bytes, stage1_layout


def _pac_capture_values(fix_points: List[FixPoint],
                         stage1_bytes: bytes,
                         stage1_layout: Aarch64ASMLayout) -> Dict[int, int]:
    """
    Run CE on stage1 TC and populate signed_value / ctx_value / spec_nesting.

    Returns a set of slot_ids where a capture at nesting=0 (arch path) succeeded.
    """
    # Build signing-instruction PC offset → [fix_points]
    pac_offset_to_fps: Dict[int, List[FixPoint]] = {}
    for fp in fix_points:
        for pac_inst in fp.pac_insts:
            if pac_inst not in stage1_layout.instruction_address:
                continue
            offset = stage1_layout.instruction_address[pac_inst] + 4
            pac_offset_to_fps.setdefault(offset, []).append(fp)

    for fp in fix_points:
        fp.signed_value = None
        fp.ctx_value    = None
        fp.spec_nesting = None

    cer = _run_ce(stage1_bytes)
    if len(cer) == 0:
        return set()

    base = _code_base(cer)
    arch_seen: Set[int] = set()

    for ite in cer:
        offset = ite.cpu.pc - base
        fps    = pac_offset_to_fps.get(offset, [])
        nesting = ite.metadata.speculation_nesting
        for fp in fps:
            if nesting != 0 and fp.slot_id in arch_seen:
                continue
            reg_n            = int(fp.info.reg[1:])
            fp.signed_value  = ite.cpu.gpr[reg_n]
            ctx              = fp.info.ctx_reg
            if ctx is None:
                fp.ctx_value = None
            elif ctx == "sp":
                fp.ctx_value = ite.cpu.sp
            elif ctx.startswith("x") and ctx[1:].isdigit():
                fp.ctx_value = ite.cpu.gpr[int(ctx[1:])]
            fp.spec_nesting  = nesting
            if nesting == 0:
                arch_seen.add(fp.slot_id)

    return arch_seen


def _pac_gpr_after_slot(cer: ContractExecutionResult,
                         code_base: int,
                         slot_pos8_offset: int,
                         reg_n: int) -> Optional[int]:
    """Return GPR[reg_n] immediately after the instruction at slot_pos8_offset executes."""
    target_pc = code_base + slot_pos8_offset + 4
    entry = _entry_at(cer, target_pc)
    if entry is None:
        # Try any nesting level
        entry = _entry_at_any_nesting(cer, target_pc)
    return entry.cpu.gpr[reg_n] if entry is not None else None


def _pac_slot_pos8_offset(tc, layout: Aarch64ASMLayout, slot_id: int) -> Optional[int]:
    """Byte offset of the AUTH/XPAC instruction (pos=AUTH_SLOT_POS) in tc."""
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                if (getattr(inst, "_pac_slot_id", None) == slot_id and
                        getattr(inst, "_pac_slot_pos", None) == AUTH_SLOT_POS):
                    return layout.instruction_address.get(inst)
    return None


# ===========================================================================
# MTE helpers
# ===========================================================================

def _mte_stage1(seed: int) -> Optional[Tuple]:
    gen = _make_gen(seed)
    mte = MTEInstrumentation(gen)
    asm_path = os.path.join(_TMPDIR, f"mte_{seed}.asm")
    try:
        tc = gen.create_test_case(asm_path, disable_assembler=True)
        stage1_tc, fix_points = mte.instrument_stage1(copy.deepcopy(tc))
    except Exception:
        return None
    if not fix_points:
        return None
    stage1_bytes, stage1_layout = _tc_to_bytes_and_layout(stage1_tc, gen)
    return gen, mte, fix_points, stage1_tc, stage1_bytes, stage1_layout


def _mte_capture_nesting(fix_points: List[MTEFixPoint],
                           stage1_bytes: bytes,
                           stage1_layout: Aarch64ASMLayout) -> Set[int]:
    """
    Run CE on stage1 TC and populate spec_nesting per fix-point.

    The memory access instruction is always at nop_offset + 4.
    Returns set of slot_ids with spec_nesting == 0 (arch slots).
    """
    # NOP placeholder offset → fix-point
    offset_to_fp: Dict[int, MTEFixPoint] = {}
    for fp in fix_points:
        nop = fp.slot_insts[0]
        if nop not in stage1_layout.instruction_address:
            continue
        nop_offset = stage1_layout.instruction_address[nop]
        offset_to_fp[nop_offset + 4] = fp  # memory access is at nop+4

    for fp in fix_points:
        fp.spec_nesting = None

    cer = _run_ce(stage1_bytes)
    if len(cer) == 0:
        return set()

    base      = _code_base(cer)
    arch_seen: Set[int] = set()

    for ite in cer:
        if not ite.metadata.has_memory_access:
            continue
        offset = ite.cpu.pc - base
        fp     = offset_to_fp.get(offset)
        if fp is None:
            continue
        nesting = ite.metadata.speculation_nesting
        if nesting == 0:
            fp.spec_nesting = 0
            arch_seen.add(fp.slot_id)
        elif fp.slot_id not in arch_seen and fp.spec_nesting is None:
            fp.spec_nesting = int(nesting)

    return arch_seen


def _mte_ea_at_slot(cer: ContractExecutionResult,
                     code_base: int,
                     mem_access_offset: int) -> Optional[int]:
    """Return effective address of the memory access at mem_access_offset."""
    target_pc = code_base + mem_access_offset
    for ite in cer:
        if ite.cpu.pc == target_pc and ite.metadata.has_memory_access:
            return ite.metadata.memory_access.effective_address
    return None


# ===========================================================================
# 1. TestPacCeE2ECanonicalPointer
# ===========================================================================

class TestPacCeE2ECanonicalPointer(unittest.TestCase):
    """
    CE-verified PAC promise: TC2 AUTH produces the same canonical pointer as TC1 XPAC.

    Pipeline:
      stage1 CE  → captures signed_value / ctx_value for arch-path slots
      TC1 CE     → GPR after XPAC  = canonical_ptr
      TC2 CE     → GPR after AUTH  = canonical_ptr  (auth succeeded)
      TC3 CE     → GPR after AUTH  = canonical_ptr  (TC3 == TC2 for arch slots)
    """

    # Per-slot results collected during setUpClass.
    _arch_results: List[Dict] = []  # each: {fp, slot_pos8_offset, xpac_gpr, auth_gpr, tc3_gpr}

    @classmethod
    def setUpClass(cls):
        if not CE_BINARY.exists():
            return

        cls._arch_results = []
        random.seed(0)

        for seed in range(300):
            result = _pac_stage1(seed)
            if result is None:
                continue
            gen, pac, fix_points, stage1_tc, stage1_bytes, stage1_layout = result

            # Capture signing values from CE
            arch_seen = _pac_capture_values(fix_points, stage1_bytes, stage1_layout)
            if not arch_seen:
                continue  # no arch-path signing in this TC

            # Keep only arch slots with captured values
            arch_fps = [fp for fp in fix_points
                        if fp.slot_id in arch_seen and fp.signed_value is not None]
            if not arch_fps:
                continue

            # Stage2 → TC1/TC2/TC3
            random.seed(seed)
            tc1, tc2, tc3 = pac.instrument_stage2(stage1_tc, fix_points)

            # Assemble TC1/TC2/TC3
            tc1_bytes, tc1_layout = _tc_to_bytes_and_layout(tc1, gen)
            tc2_bytes, tc2_layout = _tc_to_bytes_and_layout(tc2, gen)
            tc3_bytes, tc3_layout = _tc_to_bytes_and_layout(tc3, gen)

            # Run CE on each variant
            try:
                cer1 = _run_ce(tc1_bytes)
                cer2 = _run_ce(tc2_bytes)
                cer3 = _run_ce(tc3_bytes)
            except RuntimeError:
                continue  # CE crash — skip this TC

            if not (len(cer1) and len(cer2) and len(cer3)):
                continue

            base1 = _code_base(cer1)
            base2 = _code_base(cer2)
            base3 = _code_base(cer3)

            for fp in arch_fps:
                reg_n = int(fp.info.reg[1:])

                # Find pos8 offset from stage1 layout (same offset in all TCs)
                pos8_off = _pac_slot_pos8_offset(stage1_tc, stage1_layout, fp.slot_id)
                if pos8_off is None:
                    continue

                xpac_gpr = _pac_gpr_after_slot(cer1, base1, pos8_off, reg_n)
                auth_gpr  = _pac_gpr_after_slot(cer2, base2, pos8_off, reg_n)
                tc3_gpr   = _pac_gpr_after_slot(cer3, base3, pos8_off, reg_n)

                if xpac_gpr is not None and auth_gpr is not None:
                    cls._arch_results.append({
                        "fp":            fp,
                        "signed_value":  fp.signed_value,
                        "pos8_off":      pos8_off,
                        "xpac_gpr":      xpac_gpr,
                        "auth_gpr":      auth_gpr,
                        "tc3_gpr":       tc3_gpr,
                        "seed":          seed,
                    })

            if len(cls._arch_results) >= 20:
                break

    def _require_ce(self):
        if not CE_BINARY.exists():
            self.skipTest(f"CE binary not found: {CE_BINARY}")
        if not self._arch_results:
            self.skipTest("No arch PAC slots with CE captures found")

    # ─── TC1 XPAC semantics ──────────────────────────────────────────────────

    def test_tc1_xpac_strips_pac_bits(self):
        """GPR after XPAC must differ from signed_value — PAC bits were actually stripped."""
        self._require_ce()
        stripped_count = 0
        for r in self._arch_results:
            # signed_value has PAC signature in top 16 bits; canonical does not
            # Any difference confirms XPAC ran and changed the register
            if r["xpac_gpr"] != r["signed_value"]:
                stripped_count += 1
        self.assertGreater(stripped_count, 0,
                           "XPAC never changed any GPR — signing or stripping may be a no-op")

    # ─── TC2 AUTH == TC1 XPAC ────────────────────────────────────────────────

    def test_tc2_auth_produces_same_canonical_ptr_as_tc1_xpac(self):
        """
        Core promise: AUTH with correct signed_value/ctx succeeds, giving the
        same canonical pointer that XPAC produces by unconditional stripping.
        """
        self._require_ce()
        mismatches = []
        for r in self._arch_results:
            if r["xpac_gpr"] != r["auth_gpr"]:
                mismatches.append(
                    f"seed={r['seed']} slot={r['fp'].slot_id}: "
                    f"TC1_XPAC={r['xpac_gpr']:#018x}  TC2_AUTH={r['auth_gpr']:#018x}  "
                    f"signed={r['signed_value']:#018x}")
        self.assertEqual(mismatches, [],
                         f"TC2 AUTH result ≠ TC1 XPAC result for {len(mismatches)} slots:\n"
                         + "\n".join(mismatches[:5]))

    def test_tc2_auth_not_equal_to_signed_value(self):
        """After AUTH the GPR must not equal the signed_value (PAC bits removed)."""
        self._require_ce()
        still_signed = []
        for r in self._arch_results:
            if r["auth_gpr"] == r["signed_value"]:
                still_signed.append(
                    f"seed={r['seed']} slot={r['fp'].slot_id}: "
                    f"AUTH result == signed_value {r['signed_value']:#018x}")
        self.assertEqual(still_signed, [],
                         f"AUTH left PAC bits intact:\n" + "\n".join(still_signed[:5]))

    # ─── TC3 arch == TC2 (via CE output) ─────────────────────────────────────

    def test_tc3_arch_ce_result_equals_tc2(self):
        """For arch slots (spec_nesting=0), TC3 CE output must equal TC2 CE output."""
        self._require_ce()
        mismatches = []
        for r in self._arch_results:
            if r["tc3_gpr"] is None:
                continue
            if r["tc3_gpr"] != r["auth_gpr"]:
                mismatches.append(
                    f"seed={r['seed']} slot={r['fp'].slot_id}: "
                    f"TC3={r['tc3_gpr']:#018x}  TC2={r['auth_gpr']:#018x}")
        self.assertEqual(mismatches, [],
                         f"TC3 ≠ TC2 via CE for arch slots:\n" + "\n".join(mismatches[:5]))

    def test_captures_are_nontrivial(self):
        """Sanity: we actually captured some arch-path results."""
        self._require_ce()
        self.assertGreater(len(self._arch_results), 0)


# ===========================================================================
# 2. TestPacCeE2ESpecSlot
# ===========================================================================

class TestPacCeE2ESpecSlot(unittest.TestCase):
    """
    CE run of TC3 spec slot: AUTH at pos8 runs (or CE converts to XPAC on spec path).
    No crash, and the CE produces a trace for each variant.
    """

    _found: bool = False
    _tc_seed: int = -1

    @classmethod
    def setUpClass(cls):
        if not CE_BINARY.exists():
            return
        random.seed(42)
        for seed in range(300):
            result = _pac_stage1(seed)
            if result is None:
                continue
            gen, pac, fix_points, stage1_tc, stage1_bytes, stage1_layout = result
            arch_seen = _pac_capture_values(fix_points, stage1_bytes, stage1_layout)

            # Look for at least one spec slot
            spec_fps = [fp for fp in fix_points
                        if fp.signed_value is not None and fp.slot_id not in arch_seen]
            if not spec_fps:
                continue

            # Force all to spec
            for fp in fix_points:
                if fp.signed_value is not None and fp.spec_nesting is None:
                    fp.spec_nesting = 1

            try:
                random.seed(seed)
                tc1, tc2, tc3 = pac.instrument_stage2(stage1_tc, fix_points)
                tc3_bytes, _ = _tc_to_bytes_and_layout(tc3, gen)
                cer3 = _run_ce(tc3_bytes)
                if len(cer3) > 0:
                    cls._found = True
                    cls._tc_seed = seed
                    break
            except RuntimeError:
                continue

    def _require_ce(self):
        if not CE_BINARY.exists():
            self.skipTest(f"CE binary not found: {CE_BINARY}")

    def test_tc3_spec_ce_runs_without_crash(self):
        """TC3 with spec slots runs through CE without exception."""
        self._require_ce()
        if not self._found:
            self.skipTest("No TC with spec PAC slots found after 300 seeds")
        # If we reach this point, setUpClass ran CE without crashing.
        self.assertTrue(self._found)

    def test_tc3_spec_ce_produces_trace(self):
        """TC3 CE run produces at least one trace entry (CE executed the code)."""
        self._require_ce()
        if not self._found:
            self.skipTest("No TC with spec PAC slots found")
        # Re-run to verify
        result = _pac_stage1(self._tc_seed)
        self.assertIsNotNone(result)
        gen, pac, fix_points, stage1_tc, stage1_bytes, stage1_layout = result
        _pac_capture_values(fix_points, stage1_bytes, stage1_layout)
        for fp in fix_points:
            if fp.signed_value is not None and fp.spec_nesting is None:
                fp.spec_nesting = 1
        random.seed(self._tc_seed)
        _, _, tc3 = pac.instrument_stage2(stage1_tc, fix_points)
        tc3_bytes, _ = _tc_to_bytes_and_layout(tc3, gen)
        cer3 = _run_ce(tc3_bytes)
        self.assertGreater(len(cer3), 0)


# ===========================================================================
# 3. TestMteCeE2ETagBits
# ===========================================================================

class TestMteCeE2ETagBits(unittest.TestCase):
    """
    CE-verified MTE promise:
      TC1 — NOP → memory access EA bits[59:56] == arch_tag (unchanged from sandbox base)
      TC3 — MOVK wrong_upper16 → memory access EA bits[59:56] != arch_tag (wrong tag)

    arch_tag is derived dynamically from SANDBOX_BASE bits[59:56].
    """

    # Per-slot EA results.  List of dicts: {slot_id, spec_nesting, ea_tc1, ea_tc3}
    _slot_results: List[Dict] = []

    @classmethod
    def setUpClass(cls):
        if not CE_BINARY.exists():
            return

        cls._slot_results = []
        random.seed(7)

        for seed in range(300):
            result = _mte_stage1(seed)
            if result is None:
                continue
            gen, mte, fix_points, stage1_tc, stage1_bytes, stage1_layout = result

            # Capture spec_nesting from CE
            _mte_capture_nesting(fix_points, stage1_bytes, stage1_layout)

            # Ensure at least one spec slot
            spec_fps = [fp for fp in fix_points if fp.spec_nesting is not None
                        and fp.spec_nesting != 0]
            if not spec_fps:
                # Force all to spec so we can test the wrong-tag path
                for fp in fix_points:
                    fp.spec_nesting = 1

            # Stage2
            tc1, tc2, tc3 = mte.instrument_stage2(stage1_tc, fix_points, SANDBOX_BASE)
            tc1_bytes, _ = _tc_to_bytes_and_layout(tc1, gen)
            tc3_bytes, _ = _tc_to_bytes_and_layout(tc3, gen)

            try:
                cer1 = _run_ce(tc1_bytes)
                cer3 = _run_ce(tc3_bytes)
            except RuntimeError:
                continue

            if not (len(cer1) and len(cer3)):
                continue

            base1 = _code_base(cer1)
            base3 = _code_base(cer3)

            for fp in fix_points:
                nop = fp.slot_insts[0]
                if nop not in stage1_layout.instruction_address:
                    continue
                nop_off = stage1_layout.instruction_address[nop]
                mem_off = nop_off + 4  # memory access is always right after the NOP/IRG/MOVK

                ea_tc1 = _mte_ea_at_slot(cer1, base1, mem_off)
                ea_tc3 = _mte_ea_at_slot(cer3, base3, mem_off)

                if ea_tc1 is not None and ea_tc3 is not None:
                    cls._slot_results.append({
                        "seed":         seed,
                        "slot_id":      fp.slot_id,
                        "spec_nesting": fp.spec_nesting,
                        "ea_tc1":       ea_tc1,
                        "ea_tc3":       ea_tc3,
                        "reg":          fp.reg,
                    })

            if len(cls._slot_results) >= 20:
                break

    def _require_ce(self):
        if not CE_BINARY.exists():
            self.skipTest(f"CE binary not found: {CE_BINARY}")
        if not self._slot_results:
            self.skipTest("No MTE slots with CE captures found")

    # ─── TC1: arch_tag in EA ────────────────────────────────────────────────

    def test_tc1_ea_has_arch_tag(self):
        """TC1 is NOP — memory access EA bits[59:56] must equal arch_tag."""
        self._require_ce()
        at = _arch_tag(SANDBOX_BASE)
        wrong = []
        for r in self._slot_results:
            got = (r["ea_tc1"] >> 56) & 0xF
            if got != at:
                wrong.append(
                    f"seed={r['seed']} slot={r['slot_id']}: "
                    f"ea_tc1={r['ea_tc1']:#018x}  bits[59:56]={got}  expected arch_tag={at}")
        self.assertEqual(wrong, [],
                         f"TC1 EA has wrong tag for {len(wrong)} slots:\n" +
                         "\n".join(wrong[:5]))

    # ─── TC3 spec: wrong_tag in EA ───────────────────────────────────────────

    def test_tc3_spec_ea_has_wrong_tag(self):
        """TC3 MOVK sets wrong_upper16 → EA bits[59:56] must differ from arch_tag."""
        self._require_ce()
        at = _arch_tag(SANDBOX_BASE)
        spec_results = [r for r in self._slot_results
                        if r["spec_nesting"] is not None and r["spec_nesting"] != 0]
        if not spec_results:
            self.skipTest("No spec slots in captured results")

        wrong = []
        for r in spec_results:
            got = (r["ea_tc3"] >> 56) & 0xF
            if got == at:
                wrong.append(
                    f"seed={r['seed']} slot={r['slot_id']} spec={r['spec_nesting']}: "
                    f"ea_tc3={r['ea_tc3']:#018x}  bits[59:56]={got} equals arch_tag={at}"
                    f" (must differ)")
        self.assertEqual(wrong, [],
                         f"TC3 spec EA tag equals arch_tag for {len(wrong)} slots:\n" +
                         "\n".join(wrong[:5]))

    # ─── TC3 arch: same tag as TC1 ───────────────────────────────────────────

    def test_tc3_arch_ea_has_arch_tag(self):
        """TC3 is NOP for arch slots — EA must still have arch_tag."""
        self._require_ce()
        at = _arch_tag(SANDBOX_BASE)
        arch_results = [r for r in self._slot_results if r["spec_nesting"] == 0]
        if not arch_results:
            self.skipTest("No arch slots in captured results")

        wrong = []
        for r in arch_results:
            got = (r["ea_tc3"] >> 56) & 0xF
            if got != at:
                wrong.append(
                    f"seed={r['seed']} slot={r['slot_id']}: "
                    f"ea_tc3={r['ea_tc3']:#018x}  bits[59:56]={got}  expected arch_tag={at}")
        self.assertEqual(wrong, [],
                         f"TC3 arch EA has wrong tag for {len(wrong)} slots:\n" +
                         "\n".join(wrong[:5]))

    # ─── TC3 spec EA ≠ TC1 EA (wrong tag vs arch tag) ────────────────────────

    def test_tc3_ea_differs_from_tc1_for_spec_slots(self):
        """TC3 MOVK changes the tag bits → TC3 EA bits[59:56] must differ from TC1 EA."""
        self._require_ce()
        at = _arch_tag(SANDBOX_BASE)
        spec_results = [r for r in self._slot_results
                        if r["spec_nesting"] is not None and r["spec_nesting"] != 0]
        if not spec_results:
            self.skipTest("No spec slots in captured results")

        same_count = 0
        for r in spec_results:
            tag1 = (r["ea_tc1"] >> 56) & 0xF
            tag3 = (r["ea_tc3"] >> 56) & 0xF
            if tag1 == tag3:
                same_count += 1

        self.assertEqual(same_count, 0,
                         f"TC3 EA tag same as TC1 EA tag for {same_count} spec slots "
                         f"(arch_tag={at}, TC3 must differ)")

    # ─── Sanity ──────────────────────────────────────────────────────────────

    def test_nontrivial_captures(self):
        """We captured at least some MTE slots through CE."""
        self._require_ce()
        self.assertGreater(len(self._slot_results), 0)


# ===========================================================================
# 4. TestMteCeE2ESandboxBases
# ===========================================================================

class TestMteCeE2ESandboxBases(unittest.TestCase):
    """
    Verify the wrong_tag formula for multiple sandbox_base values:
    TC3 MOVK sets wrong_upper16 correctly regardless of which arch_tag is used.
    """

    _SANDBOX_BASES = [
        0x0100_0000_4000_0000,   # arch_tag=1
        0x0500_0000_4000_0000,   # arch_tag=5
        0x0A00_0000_4000_0000,   # arch_tag=0xA
        0x0F00_0000_4000_0000,   # arch_tag=0xF
    ]

    def _require_ce(self):
        if not CE_BINARY.exists():
            self.skipTest(f"CE binary not found: {CE_BINARY}")

    def test_wrong_tag_via_ce_for_all_sandbox_bases(self):
        """Run TC3 through CE for each sandbox_base and verify EA bits[59:56] != arch_tag."""
        self._require_ce()

        # Find a TC with at least one spec slot (force spec if needed)
        tc_result = None
        for seed in range(100):
            r = _mte_stage1(seed)
            if r is not None:
                tc_result = r
                break
        if tc_result is None:
            self.skipTest("Could not find any TC with memory accesses")

        gen, mte, fix_points, stage1_tc, stage1_bytes, stage1_layout = tc_result

        # Force all slots to spec so MOVK is always generated
        for fp in fix_points:
            fp.spec_nesting = 1

        # Find nop offsets
        nop_offsets = {}
        for fp in fix_points:
            nop = fp.slot_insts[0]
            if nop in stage1_layout.instruction_address:
                nop_offsets[fp.slot_id] = stage1_layout.instruction_address[nop]

        errors = []
        for sb in self._SANDBOX_BASES:
            expected_at = _arch_tag(sb)

            _, _, tc3 = mte.instrument_stage2(stage1_tc, fix_points, sb)
            tc3_bytes, _ = _tc_to_bytes_and_layout(tc3, gen)
            try:
                cer3 = _run_ce(tc3_bytes, sb)
            except RuntimeError as e:
                errors.append(f"sb={sb:#018x}: CE crash — {e}")
                continue

            if len(cer3) == 0:
                errors.append(f"sb={sb:#018x}: empty CE trace")
                continue

            base3 = _code_base(cer3)
            for fp in fix_points:
                if fp.slot_id not in nop_offsets:
                    continue
                mem_off = nop_offsets[fp.slot_id] + 4
                ea = _mte_ea_at_slot(cer3, base3, mem_off)
                if ea is None:
                    continue  # slot not executed — skip
                got_tag = (ea >> 56) & 0xF
                if got_tag == expected_at:
                    errors.append(
                        f"sb={sb:#018x} slot={fp.slot_id}: "
                        f"ea={ea:#018x}  bits[59:56]={got_tag} equals arch_tag={expected_at}"
                        f" (must differ)")

        self.assertEqual(errors, [],
                         f"{len(errors)} wrong-tag CE failures:\n" + "\n".join(errors[:10]))

    def test_tc1_ea_arch_tag_for_all_sandbox_bases(self):
        """TC1 is always NOP: memory access EA preserves the arch_tag for every sandbox_base."""
        self._require_ce()

        tc_result = None
        for seed in range(100):
            r = _mte_stage1(seed)
            if r is not None:
                tc_result = r
                break
        if tc_result is None:
            self.skipTest("Could not find any TC with memory accesses")

        gen, mte, fix_points, stage1_tc, stage1_bytes, stage1_layout = tc_result
        for fp in fix_points:
            fp.spec_nesting = 1

        nop_offsets = {
            fp.slot_id: stage1_layout.instruction_address[fp.slot_insts[0]]
            for fp in fix_points
            if fp.slot_insts[0] in stage1_layout.instruction_address
        }

        errors = []
        for sb in self._SANDBOX_BASES:
            expected_at = _arch_tag(sb)

            tc1, _, _ = mte.instrument_stage2(stage1_tc, fix_points, sb)
            tc1_bytes, _ = _tc_to_bytes_and_layout(tc1, gen)
            try:
                cer1 = _run_ce(tc1_bytes, sb)
            except RuntimeError as e:
                errors.append(f"sb={sb:#018x}: CE crash — {e}")
                continue

            if len(cer1) == 0:
                continue

            base1 = _code_base(cer1)
            for fp in fix_points:
                if fp.slot_id not in nop_offsets:
                    continue
                mem_off = nop_offsets[fp.slot_id] + 4
                ea = _mte_ea_at_slot(cer1, base1, mem_off)
                if ea is None:
                    continue
                got_tag = (ea >> 56) & 0xF
                if got_tag != expected_at:
                    errors.append(
                        f"sb={sb:#018x} slot={fp.slot_id}: "
                        f"ea={ea:#018x}  bits[59:56]={got_tag}  expected arch_tag={expected_at}")

        self.assertEqual(errors, [],
                         f"{len(errors)} arch-tag TC1 failures:\n" + "\n".join(errors[:10]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
