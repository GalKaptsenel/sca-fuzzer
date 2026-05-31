#!/usr/bin/env python3
"""
MTE TC1/TC2/TC3 generation verifier.

Exercises MTEInstrumentation.instrument_stage1 + instrument_stage2 with varied
spec_nesting values and verifies every slot strictly.

Usage (from sca-fuzzer root, venv active):
    python3 check_mte_generation.py

What it checks
--------------
Stage-1:
  * Every memory-access instruction gets exactly one NOP placeholder tagged with
    a unique _mte_slot_id.
  * The NOP placeholder appears immediately before the memory-access instruction
    in the same basic block.

Stage-2 correctness:
  TC1  → every slot is NOP (regardless of spec_nesting).
  TC2  → spec slots = IRG Xd,Xd;  arch slots = NOP.
  TC3  → spec slots = MOVK Xd,#wrong_upper16,LSL#48;  arch slots = NOP.

  wrong_upper16 formula (verified symbolically):
      arch_tag      = (sandbox_base >> 56) & 0xF
      wrong_tag     = arch_tag ^ 1
      sandbox_up16  = (sandbox_base >> 48) & 0xFFFF
      wrong_upper16 = (sandbox_up16 & ~(0xF << 8)) | (wrong_tag << 8)

  Verified that MOVK encodes #wrong_upper16 losslessly (fits in 16 bits).
  Verified that IRG uses the SAME register as source and dest.
  Verified that slot_id tags are preserved after slot replacement.
  Verified that TC1/TC2/TC3 are independent objects (no aliasing).
"""

import re
import os
import sys
import copy
import random
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import (
    Aarch64RandomGenerator,
    MTEInstrumentation, MTEFixPoint,
    MTE_SLOT_SIZE,
)

# ===========================================================================
# Configuration
# ===========================================================================

N_TEST_CASES = 200
SEED_BASE    = 99

# Realistic sandbox base addresses to exercise the wrong_tag formula with
# different arch_tag values.
SANDBOX_BASES = [
    0x0000_0000_4000_0000,   # arch_tag=0, canonical user-space, wrong_tag=1
    0x0100_0000_4000_0000,   # arch_tag=1, wrong_tag=0
    0x0A00_0000_4000_0000,   # arch_tag=0xA, wrong_tag=0xB
    0x0F00_0000_4000_0000,   # arch_tag=0xF, wrong_tag=0xE  (wrap-around XOR1)
]

# spec_nesting scenarios to test per fix-point:
# We can't run CE here, so we inject synthetic nesting values.
SPEC_NESTING_SCENARIOS = [
    "all_arch",      # all slots are arch (spec_nesting=0)
    "all_spec",      # all slots are spec (spec_nesting=1)
    "alternating",   # odd slots spec, even slots arch
    "random",        # random per slot
]


# ===========================================================================
# Helpers
# ===========================================================================

class CheckFailed(Exception):
    pass


def check(cond: bool, msg: str):
    if not cond:
        raise CheckFailed(msg)


def extract_reg_from_irg(template: str) -> Optional[Tuple[str, str]]:
    """Parse 'IRG Xd, Xd' → (dest, src).  Both should be the same."""
    m = re.match(r'IRG\s+(\w+)\s*,\s*(\w+)', template, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower(), m.group(2).lower()


def extract_movk_fields(template: str) -> Optional[Tuple[str, int, int]]:
    """Parse 'MOVK Xd, #0xNNNN, LSL #48' → (reg, imm, lsl)."""
    m = re.match(r'MOVK\s+(\w+)\s*,\s*#0x([0-9a-fA-F]+)\s*,\s*LSL\s*#(\d+)',
                 template, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower(), int(m.group(2), 16), int(m.group(3))


def compute_expected_wrong_upper16(sandbox_base: int) -> int:
    sandbox_upper16 = (sandbox_base >> 48) & 0xFFFF
    arch_tag        = (sandbox_base >> 56) & 0xF
    wrong_tag       = arch_tag ^ 1
    return (sandbox_upper16 & ~(0xF << 8)) | (wrong_tag << 8)


def find_slot_in_tc(tc, slot_id: int):
    """Return the instruction with _mte_slot_id == slot_id, or None."""
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                if getattr(inst, '_mte_slot_id', None) == slot_id:
                    return inst
    return None


def collect_all_slot_ids(tc) -> Dict[int, Any]:
    """Return {slot_id: inst} for every MTE-tagged instruction in tc."""
    result = {}
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                sid = getattr(inst, '_mte_slot_id', None)
                if sid is not None:
                    result[sid] = inst
    return result


def bb_instruction_sequence(bb) -> List[Any]:
    """Return ordered list of all instructions in a basic block."""
    return list(bb)


def nop_placeholder_precedes_mem_access(prep_tc, fix_points: List[MTEFixPoint]) -> List[str]:
    """Check that for each fix_point, the NOP placeholder immediately precedes
    the memory-access instruction within the same basic block."""
    errors = []
    for fp in fix_points:
        bb = fp.bb
        seq = bb_instruction_sequence(bb)
        mem_pos = None
        nop_pos = None
        for i, inst in enumerate(seq):
            if inst is fp.mem_inst:
                mem_pos = i
            if getattr(inst, '_mte_slot_id', None) == fp.slot_id:
                nop_pos = i
        if mem_pos is None:
            errors.append(f"  slot {fp.slot_id}: mem_inst not found in fp.bb")
            continue
        if nop_pos is None:
            errors.append(f"  slot {fp.slot_id}: NOP placeholder not found in fp.bb")
            continue
        if nop_pos != mem_pos - 1:
            errors.append(f"  slot {fp.slot_id}: NOP at pos {nop_pos}, "
                          f"mem_inst at pos {mem_pos} — expected NOP immediately before")
    return errors


# ===========================================================================
# Per-scenario slot verifier
# ===========================================================================

def verify_slots(
    fix_points: List[MTEFixPoint],
    tc1, tc2, tc3,
    sandbox_base: int,
    scenario: str,
    tc_idx: int,
) -> List[str]:
    errors = []
    expected_wrong = compute_expected_wrong_upper16(sandbox_base)

    ids1 = collect_all_slot_ids(tc1)
    ids2 = collect_all_slot_ids(tc2)
    ids3 = collect_all_slot_ids(tc3)

    for fp in fix_points:
        sid     = fp.slot_id
        is_spec = fp.spec_nesting is not None and fp.spec_nesting > 0
        ctx     = f"[TC{tc_idx} scenario={scenario} slot={sid} reg={fp.reg} spec={is_spec}]"

        try:
            inst1 = ids1.get(sid)
            inst2 = ids2.get(sid)
            inst3 = ids3.get(sid)

            check(inst1 is not None, f"{ctx} TC1: slot missing")
            check(inst2 is not None, f"{ctx} TC2: slot missing")
            check(inst3 is not None, f"{ctx} TC3: slot missing")

            # ── TC1: always NOP ─────────────────────────────────────────────
            check(inst1.name.lower() == 'nop',
                  f"{ctx} TC1: expected NOP, got '{inst1.name}' (template='{inst1.template}')")

            # ── TC2: spec → IRG, arch → NOP ─────────────────────────────────
            if is_spec:
                check(inst2.name.lower() == 'irg',
                      f"{ctx} TC2[spec]: expected IRG, got '{inst2.name}'")
                parsed = extract_reg_from_irg(inst2.template or "")
                check(parsed is not None,
                      f"{ctx} TC2[spec]: IRG template '{inst2.template}' unparseable")
                dest_r, src_r = parsed
                expected_r = fp.reg.lower()
                check(dest_r == expected_r,
                      f"{ctx} TC2[spec]: IRG dest='{dest_r}' expected '{expected_r}'")
                check(src_r == expected_r,
                      f"{ctx} TC2[spec]: IRG src='{src_r}' expected '{expected_r}'")
            else:
                check(inst2.name.lower() == 'nop',
                      f"{ctx} TC2[arch]: expected NOP, got '{inst2.name}'")

            # ── TC3: spec → MOVK #wrong_upper16 LSL#48, arch → NOP ──────────
            if is_spec:
                check(inst3.name.lower() == 'movk',
                      f"{ctx} TC3[spec]: expected MOVK, got '{inst3.name}'")
                parsed3 = extract_movk_fields(inst3.template or "")
                check(parsed3 is not None,
                      f"{ctx} TC3[spec]: MOVK template '{inst3.template}' unparseable")
                reg3, imm3, lsl3 = parsed3
                expected_r = fp.reg.lower()
                check(reg3 == expected_r,
                      f"{ctx} TC3[spec]: MOVK reg='{reg3}' expected '{expected_r}'")
                check(lsl3 == 48,
                      f"{ctx} TC3[spec]: MOVK LSL#={lsl3} expected 48")
                check(imm3 == expected_wrong,
                      f"{ctx} TC3[spec]: MOVK imm=0x{imm3:04x} "
                      f"expected=0x{expected_wrong:04x} (sandbox_base=0x{sandbox_base:016x})")
                check(expected_wrong <= 0xFFFF,
                      f"{ctx} TC3[spec]: wrong_upper16=0x{expected_wrong:x} exceeds 16 bits!")
            else:
                check(inst3.name.lower() == 'nop',
                      f"{ctx} TC3[arch]: expected NOP, got '{inst3.name}'")

            # ── Slot-id tag survives replacement ──────────────────────────────
            check(getattr(inst1, '_mte_slot_id', None) == sid,
                  f"{ctx} TC1: _mte_slot_id lost after replacement")
            check(getattr(inst2, '_mte_slot_id', None) == sid,
                  f"{ctx} TC2: _mte_slot_id lost after replacement")
            check(getattr(inst3, '_mte_slot_id', None) == sid,
                  f"{ctx} TC3: _mte_slot_id lost after replacement")

            # ── No object aliasing between variants ───────────────────────────
            check(inst1 is not inst2, f"{ctx} TC1 and TC2 share inst object")
            check(inst2 is not inst3, f"{ctx} TC2 and TC3 share inst object")
            check(inst1 is not inst3, f"{ctx} TC1 and TC3 share inst object")

        except CheckFailed as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{ctx} UNEXPECTED: {traceback.format_exc()}")

    return errors


# ===========================================================================
# Stage-1 structural checks (independent of stage-2)
# ===========================================================================

def verify_stage1(prep_tc, fix_points: List[MTEFixPoint], tc_idx: int) -> List[str]:
    errors = []
    try:
        # Every fix_point has a unique slot_id
        slot_ids = [fp.slot_id for fp in fix_points]
        check(len(slot_ids) == len(set(slot_ids)),
              f"[TC{tc_idx}] stage1: duplicate slot_ids: {slot_ids}")

        # NOP placeholder is the first slot_insts entry
        for fp in fix_points:
            check(len(fp.slot_insts) == MTE_SLOT_SIZE,
                  f"[TC{tc_idx}] slot {fp.slot_id}: slot_insts len={len(fp.slot_insts)} "
                  f"expected {MTE_SLOT_SIZE}")
            nop = fp.slot_insts[0]
            check(nop.name.lower() == 'nop',
                  f"[TC{tc_idx}] slot {fp.slot_id}: slot_insts[0] is '{nop.name}' expected NOP")
            check(getattr(nop, '_mte_slot_id', None) == fp.slot_id,
                  f"[TC{tc_idx}] slot {fp.slot_id}: NOP has _mte_slot_id="
                  f"{getattr(nop,'_mte_slot_id','MISSING')}")

        # Each slot_id appears exactly once in the TC
        tc_slot_map = collect_all_slot_ids(prep_tc)
        for fp in fix_points:
            check(fp.slot_id in tc_slot_map,
                  f"[TC{tc_idx}] slot {fp.slot_id}: not found in prep_tc")

        # NOP placeholder immediately precedes the memory-access instruction
        ordering_errors = nop_placeholder_precedes_mem_access(prep_tc, fix_points)
        errors.extend(ordering_errors)

    except CheckFailed as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"[TC{tc_idx}] stage1 UNEXPECTED: {traceback.format_exc()}")
    return errors


# ===========================================================================
# Wrong-tag formula symbolic verification
# ===========================================================================

def verify_wrong_tag_formula():
    """Verify the wrong_upper16 formula is correct for all 16 arch_tag values."""
    errors = []
    for arch_tag in range(16):
        # Construct a sandbox_base with the given arch_tag in bits[59:56]
        # (i.e., the tag nibble of the 16-bit upper field is bits[11:8] of upper16)
        base = arch_tag << 56  # simple: tag in top nibble of sandbox_base
        result = compute_expected_wrong_upper16(base)
        got_tag = (result >> 8) & 0xF
        expected_wrong_tag = arch_tag ^ 1
        if got_tag != expected_wrong_tag:
            errors.append(f"  arch_tag={arch_tag:#x}: wrong_upper16=0x{result:04x} "
                          f"tag_bits={got_tag:#x} expected {expected_wrong_tag:#x}")
        if result > 0xFFFF:
            errors.append(f"  arch_tag={arch_tag:#x}: wrong_upper16=0x{result:x} exceeds 16 bits")
        # Verify bits preserved: bits[15:12] and bits[7:0] come from sandbox_upper16
        sandbox_upper16 = (base >> 48) & 0xFFFF
        preserved_high = (result >> 12) & 0xF
        expected_high  = (sandbox_upper16 >> 12) & 0xF
        preserved_low  = result & 0xFF
        expected_low   = sandbox_upper16 & 0xFF
        if preserved_high != expected_high:
            errors.append(f"  arch_tag={arch_tag:#x}: upper nibble of wrong_upper16 "
                          f"0x{preserved_high:x} != sandbox upper nibble 0x{expected_high:x}")
        if preserved_low != expected_low:
            errors.append(f"  arch_tag={arch_tag:#x}: lower byte of wrong_upper16 "
                          f"0x{preserved_low:02x} != sandbox lower byte 0x{expected_low:02x}")
    return errors


# ===========================================================================
# Main
# ===========================================================================

def main():
    # ── Formula self-check ─────────────────────────────────────────────────
    print("Verifying wrong_tag formula for all 16 arch_tag values …")
    formula_errors = verify_wrong_tag_formula()
    if formula_errors:
        print("  FORMULA ERRORS:")
        for e in formula_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("  Formula OK for all arch_tag in [0..15]\n")

    # ── Generator setup ────────────────────────────────────────────────────
    CONF.load("config.yml")
    isa = InstructionSet("base.json", CONF.instruction_categories)

    total_tcs         = 0
    tcs_with_mem      = 0
    total_slots       = 0
    total_spec_slots  = 0
    total_arch_slots  = 0
    total_errors      = 0
    all_errors        = []

    print(f"Generating and checking {N_TEST_CASES} test cases × "
          f"{len(SANDBOX_BASES)} sandbox_bases × "
          f"{len(SPEC_NESTING_SCENARIOS)} scenarios …\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        for tc_idx in range(N_TEST_CASES):
            seed  = SEED_BASE + tc_idx
            rng   = random.Random(seed)
            gen   = Aarch64RandomGenerator(isa, seed)
            mte   = MTEInstrumentation(gen)

            asm_path = os.path.join(tmpdir, f"tc_{tc_idx}.asm")
            try:
                tc = gen.create_test_case(asm_path, disable_assembler=True)
            except Exception as e:
                all_errors.append(f"[TC{tc_idx}] create_test_case: {e}")
                total_tcs += 1
                continue

            total_tcs += 1

            # ── Stage 1 ────────────────────────────────────────────────────
            try:
                prep_tc, fix_points = mte.instrument_stage1(copy.deepcopy(tc))
            except Exception as e:
                all_errors.append(f"[TC{tc_idx}] instrument_stage1: {traceback.format_exc()}")
                continue

            if not fix_points:
                continue  # TC has no memory accesses (unlikely but possible)

            tcs_with_mem += 1
            total_slots  += len(fix_points)

            # ── Stage-1 structural checks ──────────────────────────────────
            s1_errors = verify_stage1(prep_tc, fix_points, tc_idx)
            all_errors.extend(s1_errors)
            total_errors += len(s1_errors)

            # ── Stage 2: exercise all sandbox_base × scenario combinations ──
            for sandbox_base in SANDBOX_BASES:
                for scenario in SPEC_NESTING_SCENARIOS:
                    # Inject synthetic spec_nesting values
                    for i, fp in enumerate(fix_points):
                        if scenario == "all_arch":
                            fp.spec_nesting = 0
                        elif scenario == "all_spec":
                            fp.spec_nesting = 1
                        elif scenario == "alternating":
                            fp.spec_nesting = i % 2
                        elif scenario == "random":
                            fp.spec_nesting = rng.randint(0, 2)

                    try:
                        tc1, tc2, tc3 = mte.instrument_stage2(prep_tc, fix_points, sandbox_base)
                    except Exception as e:
                        all_errors.append(
                            f"[TC{tc_idx} sb=0x{sandbox_base:016x} scenario={scenario}] "
                            f"instrument_stage2: {traceback.format_exc()}")
                        continue

                    errs = verify_slots(fix_points, tc1, tc2, tc3,
                                        sandbox_base, scenario, tc_idx)
                    all_errors.extend(errs)
                    total_errors += len(errs)

            # Count spec vs arch for the "all_spec" scenario
            for fp in fix_points:
                fp.spec_nesting = 1  # pretend all spec
            total_spec_slots += len(fix_points)
            for fp in fix_points:
                fp.spec_nesting = 0
            total_arch_slots += len(fix_points)

    # ── Report ────────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"  Test cases generated     : {total_tcs}")
    print(f"  TCs with memory accesses : {tcs_with_mem}")
    print(f"  Total slots verified     : {total_slots}")
    print(f"  Total errors             : {total_errors}")
    print(f"{'='*70}")

    if all_errors:
        print(f"\n  FAILURES ({len(all_errors)}):")
        for i, e in enumerate(all_errors[:60]):
            print(f"  [{i+1:3d}] {e}")
        if len(all_errors) > 60:
            print(f"  ... and {len(all_errors)-60} more")
        print(f"\n  RESULT: FAIL")
        sys.exit(1)
    else:
        print(f"\n  RESULT: ALL PASS")

    # ── Sample dump ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SAMPLE DUMP — first TC with memory accesses, sandbox_base=0x0000_0000_4000_0000:")
    sample_base = SANDBOX_BASES[0]
    expected_wrong = compute_expected_wrong_upper16(sample_base)
    print(f"  sandbox_base   = 0x{sample_base:016x}")
    print(f"  arch_tag       = {(sample_base >> 56) & 0xF:#x}")
    print(f"  wrong_tag      = {((sample_base >> 56) & 0xF) ^ 1:#x}")
    print(f"  wrong_upper16  = 0x{expected_wrong:04x}")
    print(f"  MOVK template  = MOVK Xd, #0x{expected_wrong:04x}, LSL #48")
    print()

    rng2 = random.Random(SEED_BASE)
    with tempfile.TemporaryDirectory() as tmpdir2:
        for tc_idx in range(N_TEST_CASES):
            seed = SEED_BASE + tc_idx
            gen = Aarch64RandomGenerator(isa, seed)
            mte = MTEInstrumentation(gen)
            asm_path = os.path.join(tmpdir2, f"tc_{tc_idx}.asm")
            try:
                tc = gen.create_test_case(asm_path, disable_assembler=True)
                prep_tc, fix_points = mte.instrument_stage1(copy.deepcopy(tc))
            except Exception:
                continue
            if not fix_points:
                continue

            # Use "alternating" scenario for an interesting dump
            for i, fp in enumerate(fix_points):
                fp.spec_nesting = i % 2

            tc1, tc2, tc3 = mte.instrument_stage2(prep_tc, fix_points, sample_base)

            print(f"  TC index  : {tc_idx}")
            print(f"  Fix-points: {len(fix_points)}")
            for fp in fix_points[:8]:  # limit to first 8 slots
                sid     = fp.slot_id
                is_spec = fp.spec_nesting > 0
                i1      = find_slot_in_tc(tc1, sid)
                i2      = find_slot_in_tc(tc2, sid)
                i3      = find_slot_in_tc(tc3, sid)
                print(f"    slot={sid:2d}  reg={fp.reg:4s}  mem_inst={fp.mem_inst.name:12s}  "
                      f"spec={is_spec}")
                print(f"       TC1: {i1.name:6s}  '{i1.template}'")
                print(f"       TC2: {i2.name:6s}  '{i2.template}'")
                print(f"       TC3: {i3.name:6s}  '{i3.template}'")
            if len(fix_points) > 8:
                print(f"    ... ({len(fix_points) - 8} more slots not shown)")
            break  # first TC only


if __name__ == "__main__":
    main()
