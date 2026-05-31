#!/usr/bin/env python3
"""
Show original TC → stage-1 TC → stage-2 {STRIP_ONLY, AUTH_CORRECT, AUTH_WRONG}
disassemblies for one interesting PAC test case.

Run from sca-fuzzer root with venv active:
    python3 show_pac_stages.py [seed]
"""

import copy
import os
import sys
import random
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import (
    Aarch64RandomGenerator, Aarch64Printer, Aarch64ASMLayout,
    PACInstrumentation, FixPoint, PACVariant,
    SLOT_SIZE, AUTH_SLOT_POS, ConfigurableGenerator,
)
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc

# ─── config ───────────────────────────────────────────────────────────────────
CONF.load("config_pac.yml")
ISA_PATH   = "base.json"
SEARCH_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# minimum fix-points to accept; keeps searching until found
MIN_FPS = 3

# ─── helpers ──────────────────────────────────────────────────────────────────

cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
cs.detail = False


_target_desc = Aarch64TargetDesc()


def assemble(tc) -> bytes:
    layout   = Aarch64ASMLayout(tc)
    assembly = Aarch64Printer(_target_desc).print_layout(layout)
    return ConfigurableGenerator.in_memory_assemble(assembly)


def disasm(data: bytes, label: str, slot_pcs: dict = None, base: int = 0):
    """
    Print Capstone disassembly.  slot_pcs: {pc → annotation_string}
    """
    print(f"\n{'─'*70}")
    print(f"  {label}  ({len(data)} bytes)")
    print(f"{'─'*70}")
    for insn in cs.disasm(data, base):
        ann = ""
        if slot_pcs:
            ann = slot_pcs.get(insn.address, "")
        flag = f"  ◄ {ann}" if ann else ""
        print(f"  +{insn.address - base:04x}: {insn.mnemonic:<8} {insn.op_str}{flag}")


def build_slot_annotations(tc, fix_points, layout, base=0) -> dict:
    """
    Return {pc: annotation} for every slot instruction and every signing macro instruction.
    """
    ann = {}
    slot_label = {
        AUTH_SLOT_POS: "AUTH_SLOT",
    }
    for fp in fix_points:
        nesting_str = f"slot_nesting={fp.slot_nesting}"
        for inst in fp.slot_insts:
            sid  = getattr(inst, "_pac_slot_id",  None)
            spos = getattr(inst, "_pac_slot_pos", None)
            if sid is None:
                continue
            pc = base + layout.instruction_address[inst]
            pos_label = slot_label.get(spos, f"pos{spos}")
            ann[pc] = f"slot{sid} {pos_label}  [{nesting_str}]"
        for inst in fp.pac_insts:
            pc = base + layout.instruction_address[inst]
            ann[pc] = f"slot{fp.slot_id} SIGN ({fp.info.pac_mnemonic} {fp.info.reg})"
    return ann


# ─── find an interesting test case ────────────────────────────────────────────

isa = InstructionSet(ISA_PATH, CONF.instruction_categories)

print(f"Searching for a TC with ≥{MIN_FPS} fix-points …")

chosen_seed    = None
chosen_tc_orig = None
chosen_s1_tc   = None
chosen_fps     = None

with tempfile.TemporaryDirectory() as tmpdir:
    for offset in range(2000):
        seed = SEARCH_SEED + offset
        gen  = Aarch64RandomGenerator(isa, seed)
        try:
            tc_orig = gen.create_test_case(
                os.path.join(tmpdir, f"tc_{seed}.asm"),
                disable_assembler=True,
            )
        except Exception:
            continue

        pac = PACInstrumentation(gen, CONF.pac_xpac_weight,
                                 CONF.pac_auth_weight, CONF.pac_sign_weight)
        try:
            s1_tc, fps = pac.instrument_stage1(copy.deepcopy(tc_orig))
        except Exception:
            continue

        if len(fps) >= MIN_FPS:
            chosen_seed    = seed
            chosen_tc_orig = tc_orig
            chosen_s1_tc   = s1_tc
            chosen_fps     = fps
            chosen_pac     = pac
            break

if chosen_fps is None:
    print("No suitable TC found — try a different seed.")
    sys.exit(1)

print(f"Found TC at seed={chosen_seed}  fix-points={len(chosen_fps)}\n")

# ─── inject realistic CE-captured values + set slot_nesting ───────────────────
# Simulate what the CE would report:
#   • even-indexed fix-points: arch (slot_nesting=0)
#   • odd-indexed fix-points:  spec (slot_nesting=1)
# signed_value uses realistic PAC-encoded upper bits (not all-0xff / 0).

rng = random.Random(chosen_seed ^ 0xDEADBEEF)
SANDBOX_BASE = 0x1000_0000_0000     # fake but canonical

for i, fp in enumerate(chosen_fps):
    # Canonical address within sandbox (lower 13 bits are the offset from base)
    canonical   = SANDBOX_BASE + rng.randint(0, 0x1FFF) * 8
    pac_sig_hi  = rng.randint(0x10, 0xFE)              # realistic 8-bit PAC tag
    fp.signed_value = (pac_sig_hi << 56) | canonical   # PAC bits in [63:56]

    if fp.info.ctx_reg is not None:
        fp.ctx_value = SANDBOX_BASE + rng.randint(0, 0x1FFF) * 8

    # Alternate arch / spec so we see both cases
    fp.slot_nesting  = 0 if (i % 2 == 0) else 1
    fp.spec_nesting  = fp.slot_nesting

# ─── stage 2 ──────────────────────────────────────────────────────────────────
variants = chosen_pac.instrument_stage2(chosen_s1_tc, chosen_fps)

# ─── assemble everything ──────────────────────────────────────────────────────
orig_bytes  = assemble(chosen_tc_orig)
s1_bytes    = assemble(chosen_s1_tc)
s1_layout   = Aarch64ASMLayout(chosen_s1_tc)

strip_bytes   = assemble(variants[PACVariant.STRIP_ONLY])
correct_bytes = assemble(variants[PACVariant.AUTH_CORRECT])
wrong_bytes   = assemble(variants[PACVariant.AUTH_WRONG])

# Build slot annotations for stage-1 (slots still have layout positions)
s1_ann = build_slot_annotations(chosen_s1_tc, chosen_fps, s1_layout, base=0)

# For stage-2 variants: only annotate the AUTH_SLOT_POS of each fix-point
# (The rest of each slot is ctx/ptr restore which is the same across variants).
def variant_ann(variant_tc, pac_obj, fps) -> dict:
    vl       = Aarch64ASMLayout(variant_tc)
    slot_map = pac_obj._find_slot_insts(variant_tc)
    a = {}
    for fp in fps:
        pos_map = slot_map.get(fp.slot_id, {})
        entry   = pos_map.get(AUTH_SLOT_POS)
        if entry is None:
            continue
        inst = entry[0]
        pc   = vl.instruction_address[inst]
        nstr = "arch" if fp.slot_nesting == 0 else f"spec(depth={fp.slot_nesting})"
        a[pc] = f"slot{fp.slot_id}  AUTH_SLOT  [{nstr}]"
    return a

strip_ann   = variant_ann(variants[PACVariant.STRIP_ONLY],   chosen_pac, chosen_fps)
correct_ann = variant_ann(variants[PACVariant.AUTH_CORRECT], chosen_pac, chosen_fps)
wrong_ann   = variant_ann(variants[PACVariant.AUTH_WRONG],   chosen_pac, chosen_fps)

# ─── print fix-point metadata ─────────────────────────────────────────────────
print("Fix-points:")
for fp in chosen_fps:
    ctx  = fp.info.ctx_reg if fp.info.ctx_reg else "none"
    nest = "arch" if fp.slot_nesting == 0 else f"spec(depth={fp.slot_nesting})"
    print(f"  slot={fp.slot_id}  reg={fp.info.reg}  ctx={ctx}"
          f"  sign={fp.info.pac_mnemonic}  auth={fp.info.auth_mnemonic}"
          f"  xpac={fp.info.xpac_mnemonic}  nesting={nest}"
          f"  signed_value=0x{fp.signed_value:016x}")

# ─── disassemblies ────────────────────────────────────────────────────────────
disasm(orig_bytes,    "ORIGINAL TC (before any PAC instrumentation)")
disasm(s1_bytes,      "STAGE-1  TC (signing macros + XPAC slots inserted)", s1_ann)
disasm(strip_bytes,   "STAGE-2  STRIP_ONLY  (slot pos 8 = XPAC, ptr loaded)",  strip_ann)
disasm(correct_bytes, "STAGE-2  AUTH_CORRECT (slot pos 8 = AUTH with correct signed_value)", correct_ann)
disasm(wrong_bytes,   "STAGE-2  AUTH_WRONG   (arch slots=correct, spec slots=wrong upper16)", wrong_ann)

print()
print("Legend:")
print("  SIGN     = PAC signing instruction (AND+ADD sandbox macro + PAC*)")
print("  AUTH_SLOT pos 0-3 = ctx_value restore (MOVZ+MOVK×3 or NOPs)")
print("  AUTH_SLOT pos 4-7 = signed_value restore (MOVZ+MOVK×3)")
print("  AUTH_SLOT pos 8   = XPAC / AUTH-correct / AUTH-wrong (variant-specific)")
print("  AUTH_SLOT pos 9-12= post-auth ctx restore (AUTH_WRONG spec only, else NOPs)")
print()
print("  AUTH_WRONG spec slots: pos 4-7 loads only lower 48 bits of signed_value")
print("  (upper 16 = stale → auth always fails → corrupted pointer → speculative fault)")
