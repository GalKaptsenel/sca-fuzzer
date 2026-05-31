#!/usr/bin/env python3
"""
Run PAC TC1/TC2/TC3 generation at scale and verify every slot structurally.

Usage (from sca-fuzzer root, venv active):
    python3 check_pac_generation.py

What it does
------------
1. Load real InstructionSet + Aarch64RandomGenerator (same path as the fuzzer).
2. Generate N random test cases with PAC signing instructions.
3. Run instrument_stage1 → fix_points.
4. Inject realistic fake CE-captured values (signed_value, ctx_value, spec_nesting=0).
5. Run instrument_stage2 → tc1, tc2, tc3.
6. For every fix_point in every test case, verify:
     - Each of TC1/TC2/TC3 has exactly SLOT_SIZE=9 tagged instructions for that slot.
     - TC1 pos 0-3 : MOVZ+MOVK×3 loading ctx_value  (NOPs when ctx_reg is None)
     - TC1 pos 4-7 : MOVZ+MOVK×3 loading signed_value
     - TC1 pos 8   : XPAC instruction matching signing variant
     - TC2 pos 0-3 : same ctx restore as TC1
     - TC2 pos 4-7 : same ptr restore as TC1
     - TC2 pos 8   : AUTH instruction matching signing variant
     - TC3          : identical to TC2 (name + template) for all positions
     - _pac_slot_id / _pac_slot_pos tags correct on every instruction
     - MOVZ+MOVK×3 value encoding losslessly round-trips signed_value and ctx_value
     - AUTH and XPAC mnemonics are consistent with the PAC signing mnemonic
7. Print pass/fail summary with per-TC statistics.
"""

import re
import os
import sys
import copy
import random
import tempfile
import traceback
from typing import Dict, List, Optional, Tuple

# ── project root must be in sys.path ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import (
    Aarch64RandomGenerator, PACInstrumentation, FixPoint, SignedRegInfo,
    SLOT_SIZE, FIX_COUNT_CTX, FIX_COUNT_PTR, CTX_SLOT_START, PTR_SLOT_START, AUTH_SLOT_POS,
    _PAC_TO_AUTH,
)

# ===========================================================================
# Configuration
# ===========================================================================

N_TEST_CASES = 200           # number of random TCs to generate and check
SEED_BASE    = 42             # deterministic base seed

# PAC signing → expected auth mnemonic
_SIGN_TO_AUTH = {
    "pacia":  "autia",  "pacib":  "autib",
    "pacda":  "autda",  "pacdb":  "autdb",
    "paciza": "autiza", "pacizb": "autizb",
    "pacdza": "autdza", "pacdzb": "autdzb",
}
# PAC signing → expected xpac mnemonic
_SIGN_TO_XPAC = {
    "pacia":  "xpaci", "pacib":  "xpaci",
    "pacda":  "xpacd", "pacdb":  "xpacd",
    "paciza": "xpaci", "pacizb": "xpaci",
    "pacdza": "xpacd", "pacdzb": "xpacd",
}

# ===========================================================================
# Helpers
# ===========================================================================

def decode_value(insts) -> int:
    """Reconstruct a 64-bit integer from a MOVZ+MOVK× sequence."""
    val = 0
    for inst in insts:
        m_imm = re.search(r'#0x([0-9a-fA-F]+)', inst.template or "")
        m_lsl = re.search(r'LSL #(\d+)', inst.template or "")
        if not m_imm:
            return None  # not a load-imm instruction
        imm = int(m_imm.group(1), 16)
        lsl = int(m_lsl.group(1)) if m_lsl else 0
        val |= (imm << lsl)
    return val


def get_slot_insts(tc, slot_id: int) -> List:
    """Return the SLOT_SIZE instructions for a slot in position order."""
    pos_map: Dict[int, object] = {}
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                if getattr(inst, "_pac_slot_id", None) == slot_id:
                    pos_map[inst._pac_slot_pos] = inst
    if len(pos_map) != SLOT_SIZE:
        return None  # missing/extra positions
    return [pos_map[p] for p in range(SLOT_SIZE)]


class CheckFailed(Exception):
    pass


def check(cond: bool, msg: str):
    if not cond:
        raise CheckFailed(msg)


# ===========================================================================
# Per-slot structural verifier
# ===========================================================================

def verify_slot(fp: FixPoint, tc1, tc2, tc3, tc_idx: int) -> List[str]:
    """Verify a single slot across TC1/TC2/TC3. Returns list of error strings."""
    errors = []
    sid = fp.slot_id
    pac_mn = fp.info.pac_mnemonic.lower()

    try:
        # ── Extract all three slot instruction lists ──────────────────────────
        s1 = get_slot_insts(tc1, sid)
        s2 = get_slot_insts(tc2, sid)
        s3 = get_slot_insts(tc3, sid)

        check(s1 is not None, f"TC1 slot {sid}: wrong number of tagged instructions")
        check(s2 is not None, f"TC2 slot {sid}: wrong number of tagged instructions")
        check(s3 is not None, f"TC3 slot {sid}: wrong number of tagged instructions")

        check(len(s1) == SLOT_SIZE, f"TC1 slot {sid}: len={len(s1)} expected {SLOT_SIZE}")
        check(len(s2) == SLOT_SIZE, f"TC2 slot {sid}: len={len(s2)} expected {SLOT_SIZE}")
        check(len(s3) == SLOT_SIZE, f"TC3 slot {sid}: len={len(s3)} expected {SLOT_SIZE}")

        # ── TAG INTEGRITY: every instruction carries correct slot id and pos ──
        for label, s in (("TC1", s1), ("TC2", s2), ("TC3", s3)):
            for pos, inst in enumerate(s):
                check(getattr(inst, "_pac_slot_id", None) == sid,
                      f"{label} slot {sid} pos {pos}: _pac_slot_id={getattr(inst,'_pac_slot_id','MISSING')} expected {sid}")
                check(getattr(inst, "_pac_slot_pos", None) == pos,
                      f"{label} slot {sid} pos {pos}: _pac_slot_pos={getattr(inst,'_pac_slot_pos','MISSING')} expected {pos}")

        # ── TC1 POSITION 8: XPAC, never AUTH ─────────────────────────────────
        expected_xpac = _SIGN_TO_XPAC.get(pac_mn)
        check(expected_xpac is not None,
              f"TC1 slot {sid}: unknown pac mnemonic '{pac_mn}'")
        check(s1[AUTH_SLOT_POS].name == expected_xpac,
              f"TC1 slot {sid} pos 8: got '{s1[AUTH_SLOT_POS].name}' expected '{expected_xpac}'")
        auth_names = set(_SIGN_TO_AUTH.values())
        for pos, inst in enumerate(s1):
            check(inst.name not in auth_names,
                  f"TC1 slot {sid} pos {pos}: unexpected AUTH '{inst.name}'")

        # ── TC2 POSITION 8: AUTH matching the signing variant ─────────────────
        expected_auth = _SIGN_TO_AUTH.get(pac_mn)
        check(expected_auth is not None,
              f"TC2 slot {sid}: unknown pac mnemonic '{pac_mn}'")
        check(s2[AUTH_SLOT_POS].name == expected_auth,
              f"TC2 slot {sid} pos 8: got '{s2[AUTH_SLOT_POS].name}' expected '{expected_auth}'")

        # ── CTX RESTORE POSITIONS 0–3 ─────────────────────────────────────────
        has_ctx = fp.info.ctx_reg is not None  # False for PACIZA/PACIZB zero-context

        if has_ctx:
            # Positions 0-3: MOVZ at 0, MOVK at 1/2/3
            check(s1[0].name == "movz",
                  f"TC1 slot {sid} pos 0: expected MOVZ, got '{s1[0].name}'")
            check(s2[0].name == "movz",
                  f"TC2 slot {sid} pos 0: expected MOVZ, got '{s2[0].name}'")
            for i in range(1, FIX_COUNT_CTX):
                check(s1[i].name == "movk",
                      f"TC1 slot {sid} pos {i}: expected MOVK, got '{s1[i].name}'")
                check(s2[i].name == "movk",
                      f"TC2 slot {sid} pos {i}: expected MOVK, got '{s2[i].name}'")

            # Value round-trip for ctx
            ctx_from_tc1 = decode_value(s1[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX])
            ctx_from_tc2 = decode_value(s2[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX])
            check(ctx_from_tc1 is not None,
                  f"TC1 slot {sid}: ctx positions not decodable as MOVZ+MOVK")
            check(ctx_from_tc1 == fp.ctx_value,
                  f"TC1 slot {sid}: ctx_value encoded={ctx_from_tc1:#018x} expected={fp.ctx_value:#018x}")
            check(ctx_from_tc2 == fp.ctx_value,
                  f"TC2 slot {sid}: ctx_value encoded={ctx_from_tc2:#018x} expected={fp.ctx_value:#018x}")

            # ctx register name in templates
            for pos in range(FIX_COUNT_CTX):
                check(fp.info.ctx_reg in s1[pos].template,
                      f"TC1 slot {sid} pos {pos}: ctx_reg '{fp.info.ctx_reg}' not in template '{s1[pos].template}'")
        else:
            # Zero-context: positions 0-3 must be NOPs
            for tc_label, s in (("TC1", s1), ("TC2", s2), ("TC3", s3)):
                for i in range(FIX_COUNT_CTX):
                    check(s[i].name == "nop",
                          f"{tc_label} slot {sid} pos {i}: expected NOP (zero-ctx), got '{s[i].name}'")

        # ── PTR RESTORE POSITIONS 4–7 ─────────────────────────────────────────
        check(s1[PTR_SLOT_START].name == "movz",
              f"TC1 slot {sid} pos {PTR_SLOT_START}: expected MOVZ, got '{s1[PTR_SLOT_START].name}'")
        check(s2[PTR_SLOT_START].name == "movz",
              f"TC2 slot {sid} pos {PTR_SLOT_START}: expected MOVZ, got '{s2[PTR_SLOT_START].name}'")
        for i in range(1, FIX_COUNT_PTR):
            check(s1[PTR_SLOT_START + i].name == "movk",
                  f"TC1 slot {sid} pos {PTR_SLOT_START+i}: expected MOVK, got '{s1[PTR_SLOT_START+i].name}'")
            check(s2[PTR_SLOT_START + i].name == "movk",
                  f"TC2 slot {sid} pos {PTR_SLOT_START+i}: expected MOVK, got '{s2[PTR_SLOT_START+i].name}'")

        # Value round-trip for signed_value
        sv_from_tc1 = decode_value(s1[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
        sv_from_tc2 = decode_value(s2[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
        check(sv_from_tc1 is not None,
              f"TC1 slot {sid}: ptr positions not decodable as MOVZ+MOVK")
        check(sv_from_tc1 == fp.signed_value,
              f"TC1 slot {sid}: signed_value encoded={sv_from_tc1:#018x} expected={fp.signed_value:#018x}")
        check(sv_from_tc2 == fp.signed_value,
              f"TC2 slot {sid}: signed_value encoded={sv_from_tc2:#018x} expected={fp.signed_value:#018x}")

        # ptr register in templates
        for pos in range(FIX_COUNT_PTR):
            check(fp.info.reg in s1[PTR_SLOT_START + pos].template,
                  f"TC1 slot {sid} ptr pos {pos}: reg '{fp.info.reg}' not in template '{s1[PTR_SLOT_START+pos].template}'")

        # ── TC3 == TC2 invariant (spec_nesting=0 always in current CE) ────────
        for pos in range(SLOT_SIZE):
            check(s3[pos].name == s2[pos].name,
                  f"TC3 slot {sid} pos {pos}: name '{s3[pos].name}' ≠ TC2 '{s2[pos].name}'")
            check(s3[pos].template == s2[pos].template,
                  f"TC3 slot {sid} pos {pos}: template '{s3[pos].template}' ≠ TC2 '{s2[pos].template}'")

        # ── TC1/TC2/TC3 are independent objects ───────────────────────────────
        for pos in range(SLOT_SIZE):
            check(s1[pos] is not s2[pos],
                  f"TC1 and TC2 share instruction object at slot {sid} pos {pos}")
            check(s2[pos] is not s3[pos],
                  f"TC2 and TC3 share instruction object at slot {sid} pos {pos}")

    except CheckFailed as e:
        errors.append(f"[TC{tc_idx} slot {sid} ({pac_mn})] {e}")
    except Exception as e:
        errors.append(f"[TC{tc_idx} slot {sid} ({pac_mn})] UNEXPECTED: {traceback.format_exc()}")

    return errors


# ===========================================================================
# Main
# ===========================================================================

def main():
    # ── Setup ─────────────────────────────────────────────────────────────────
    CONF.load("config.yml")
    isa = InstructionSet("base.json", CONF.instruction_categories)
    rng = random.Random(SEED_BASE)

    # Stats
    total_tcs      = 0
    tcs_with_pac   = 0
    total_slots    = 0
    total_errors   = 0
    all_errors     = []
    pac_mn_counts  = {}   # how many slots per signing mnemonic

    print(f"Generating and checking {N_TEST_CASES} test cases …")

    with tempfile.TemporaryDirectory() as tmpdir:
        for tc_idx in range(N_TEST_CASES):
            seed = SEED_BASE + tc_idx
            gen = Aarch64RandomGenerator(isa, seed)

            # Generate a test case (no assembler needed — we work on the AST)
            asm_path = os.path.join(tmpdir, f"tc_{tc_idx}.asm")
            try:
                tc = gen.create_test_case(asm_path, disable_assembler=True)
            except Exception as e:
                all_errors.append(f"[TC{tc_idx}] create_test_case failed: {e}")
                total_tcs += 1
                continue

            total_tcs += 1

            # ── Stage 1 ────────────────────────────────────────────────────────
            pac = PACInstrumentation(gen, CONF.pac_xpac_weight,
                                     CONF.pac_auth_weight, CONF.pac_sign_weight)
            try:
                prep_tc, fix_points = pac.instrument_stage1(copy.deepcopy(tc))
            except Exception as e:
                all_errors.append(f"[TC{tc_idx}] instrument_stage1 failed: {e}")
                continue

            if not fix_points:
                continue  # no PAC instructions in this TC, skip

            tcs_with_pac += 1

            # ── Inject fake CE captures ─────────────────────────────────────────
            # Use deterministic but varied values to exercise the full 64-bit range.
            for fp in fix_points:
                # signed_value: realistic PAC-signed pointer
                # bits[47:0] = canonical address, bits[63:48] = PAC signature
                canonical = (rng.randint(0, 0xFFFF_FFFF_FFFF) & ~0xF) | 0x4000_0000_0000
                pac_sig   = rng.randint(0, 0xFFFF)
                fp.signed_value = canonical | (pac_sig << 48)

                # ctx_value: only for non-zero-context variants
                if fp.info.ctx_reg is not None:
                    fp.ctx_value = rng.randint(0, 0xFFFF_FFFF_FFFF_FFFF)

                fp.spec_nesting = 0  # CE never speculates

                mn = fp.info.pac_mnemonic.lower()
                pac_mn_counts[mn] = pac_mn_counts.get(mn, 0) + 1
                total_slots += 1

            # ── Stage 2 ────────────────────────────────────────────────────────
            try:
                tc1, tc2, tc3 = pac.instrument_stage2(prep_tc, fix_points)
            except Exception as e:
                all_errors.append(f"[TC{tc_idx}] instrument_stage2 failed: {e}")
                continue

            # ── Structural verification ────────────────────────────────────────
            for fp in fix_points:
                errs = verify_slot(fp, tc1, tc2, tc3, tc_idx)
                all_errors.extend(errs)
                total_errors += len(errs)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Test cases generated  : {total_tcs}")
    print(f"  TCs with PAC slots    : {tcs_with_pac}")
    print(f"  Total slots verified  : {total_slots}")
    print(f"  Total errors          : {total_errors}")

    if pac_mn_counts:
        print(f"\n  Signing mnemonics seen:")
        for mn, cnt in sorted(pac_mn_counts.items()):
            auth = _SIGN_TO_AUTH.get(mn, "?")
            xpac = _SIGN_TO_XPAC.get(mn, "?")
            print(f"    {mn:10s}  count={cnt:4d}  → auth={auth}  xpac={xpac}")
    else:
        print("\n  WARNING: no PAC fix-points encountered — check instruction_categories in config.yml")

    if all_errors:
        print(f"\n{'='*70}")
        print(f"  FAILURES ({len(all_errors)}):")
        for i, e in enumerate(all_errors[:50]):  # cap at 50 to avoid wall of text
            print(f"  [{i+1:3d}] {e}")
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors)-50} more errors")
        print(f"\n{'='*70}")
        print("  RESULT: FAIL")
        sys.exit(1)
    else:
        print(f"\n  RESULT: ALL PASS")

    # ── Detailed slot dump for first TC with PAC (visual confirmation) ────────
    print(f"\n{'='*70}")
    print("  SAMPLE SLOT DUMP (first TC that has PAC fix-points):")
    rng2 = random.Random(SEED_BASE)
    with tempfile.TemporaryDirectory() as tmpdir2:
        for tc_idx in range(N_TEST_CASES):
            seed = SEED_BASE + tc_idx
            gen = Aarch64RandomGenerator(isa, seed)
            asm_path = os.path.join(tmpdir2, f"tc_{tc_idx}.asm")
            try:
                tc = gen.create_test_case(asm_path, disable_assembler=True)
            except Exception:
                continue
            pac = PACInstrumentation(gen, CONF.pac_xpac_weight,
                                     CONF.pac_auth_weight, CONF.pac_sign_weight)
            try:
                prep_tc, fix_points = pac.instrument_stage1(copy.deepcopy(tc))
            except Exception:
                continue
            if not fix_points:
                continue

            for fp in fix_points:
                canonical = (rng2.randint(0, 0xFFFF_FFFF_FFFF) & ~0xF) | 0x4000_0000_0000
                pac_sig   = rng2.randint(0, 0xFFFF)
                fp.signed_value = canonical | (pac_sig << 48)
                if fp.info.ctx_reg is not None:
                    fp.ctx_value = rng2.randint(0, 0xFFFF_FFFF_FFFF_FFFF)
                fp.spec_nesting = 0

            tc1, tc2, tc3 = pac.instrument_stage2(prep_tc, fix_points)

            print(f"\n  TC index : {tc_idx}")
            for fp in fix_points:
                sid = fp.slot_id
                print(f"\n  --- Slot {sid}  pac={fp.info.pac_mnemonic}  reg={fp.info.reg}  "
                      f"ctx_reg={fp.info.ctx_reg}")
                print(f"       signed_value = {fp.signed_value:#018x}")
                if fp.ctx_value is not None:
                    print(f"       ctx_value    = {fp.ctx_value:#018x}")

                for label, tvar in (("TC1", tc1), ("TC2", tc2), ("TC3", tc3)):
                    s = get_slot_insts(tvar, sid)
                    if s is None:
                        print(f"       {label}: MISSING SLOT")
                        continue
                    names = [i.name for i in s]
                    print(f"       {label}: {names}")
                    # Show value decoded from ptr positions
                    sv_enc = decode_value(s[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
                    match = "✓" if sv_enc == fp.signed_value else "✗"
                    print(f"            ptr_encoded={sv_enc:#018x}  {match}")
                    if fp.info.ctx_reg is not None and fp.ctx_value is not None:
                        cv_enc = decode_value(s[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX])
                        match2 = "✓" if cv_enc == fp.ctx_value else "✗"
                        print(f"            ctx_encoded={cv_enc:#018x}  {match2}")
            break  # only dump the first TC with PAC


if __name__ == "__main__":
    main()
