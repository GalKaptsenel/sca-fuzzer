#!/usr/bin/env python3
"""Local MTE-speculative-exception triage (Steps 1+2 of verify-mte-tag-leak) — no device.
Usage: accounting.py <config.yml> <violation-dir> <suspect_input_idx>

Prints: counterexample same-input check, tag-only code-diff, per-set HW divergence (MSB-first),
CE per-set accounting (arch/spec demands + prefetch streams), and — for each diverging set — whether it
is a prefetch stream of a SPECULATIVE tag-sealed access and which base register gates it. The printed
REG + LEAK sets feed ctrl_sweep.py."""
import sys, glob, os, re, itertools
REPO = "/home/gal_k_1_1998/revizor/sca-fuzzer"
sys.path.insert(0, REPO)
from src.config import CONF
CFG, VD, SUS = sys.argv[1], sys.argv[2], int(sys.argv[3])
CONF.load(CFG)
from src.fuzzer import SpeculativeExceptionFuzzer
from src.aarch64.aarch64_executor_input_encoder import deserialize
from src.aarch64.aarch64_trace import _cache_sets
from src.aarch64.aarch64_contract_executor import ExecutionClause

def dec_addg(w):
    if w == 0xd503201f: return "NOP"
    if (w >> 22) & 0x3ff == 0b1001000110: return f"ADDG x{w&0x1f} tag={(w>>10)&0xf}"
    return f"OTHER:0x{w:08x}"

# --- Step 1a: counterexamples (same-input?) ---
lines = open(f"{VD}/report.txt").read().splitlines()
ce = []
insec = False
for ln in lines:
    if ln.strip() == "## Counterexample Inputs": insec = True; continue
    if ln.startswith("## Branch"): break
    m = re.match(r"Input #(\d+)$", ln.strip())
    if insec and m: ce.append(int(m.group(1)))
print(f"counterexamples: {ce}  (same input iff they pair as N and N+K)")

# --- Step 1b: tag-only is checked on the GST image after resolve (see below) ---
# Under the GlobalSealTable the test-case code is byte-identical across every variant; a variant is
# selected purely by its GST image (the SIMD section), so "tag-only" means genuine vs decoy images
# differ only in MTE applied-tag lanes. Computed after `res` is resolved, below.

# --- Step 1c: per-set HW divergence (report htraces are MSB-first: set = 63 - char) ---
def freq(n):
    cur = None; blk = []
    for ln in lines:
        mm = re.match(r"Input #(\d+)$", ln.strip())
        if mm: cur = int(mm.group(1)); continue
        if cur == n:
            m2 = re.match(r"^([\.\^]{8,})\s*\[(\d+)\]", ln.strip())
            if m2: blk.append((int(m2.group(2)), m2.group(1)))
            elif ln.startswith("* Contract"): break
    tot = sum(c for c, _ in blk) or 1; f = {}
    for c, pat in blk:
        for i, ch in enumerate(pat):
            if ch == "^": f[63-i] = f.get(63-i, 0) + c
    return {k: v/tot for k, v in f.items()}
if len(ce) >= 2:
    fa, fb = freq(ce[0]), freq(ce[1])
    diverge = sorted(s for s in range(64) if abs(fa.get(s, 0) - fb.get(s, 0)) >= 0.08)
    print(f"HW diverging sets (|Δ|>=0.08) between #{ce[0]} and #{ce[1]}: {diverge}")
else:
    diverge = []

# --- Step 2: CE per-set accounting (local) ---
fz = SpeculativeExceptionFuzzer(os.path.join(REPO, "base.json"), REPO); fz.initialize_modules()
ex = fz.executor; ex._sandbox_base = 0xffff8873821000
sealed = ex._sealer.seal(fz.asm_parser.parse_file(f"{VD}/generated.asm"))
_inp_files = glob.glob(f"{VD}/input_{SUS:04d}.reif") or glob.glob(f"{VD}/input_{SUS:04d}.bin")
if not _inp_files:
    raise FileNotFoundError(f"no input_{SUS:04d}.reif or .bin in {VD}")
inp = deserialize(open(_inp_files[0], "rb").read()).input_
res = sealed.resolve(inp)
code = res.object_code   # GST: one byte-identical test case; variants differ only in the GST image

# --- Step 1b (GST): tag-only = genuine vs decoy images differ ONLY in MteSealing applied lanes ---
import random as _rnd
_gen, _dec = res.genuine(), res.decoy(_rnd.Random(0xA5A5))
_diffs = [i for i in range(min(len(_gen), len(_dec))) if _gen[i] != _dec[i]]
_mte_lanes = [(e.sealing.location.byte_range().start, e.sealing.location.byte_range().stop)
              for e in res._entries if e.sealing.__class__.__name__ == "MteSealing"]
_tag_only = bool(_diffs) and all(any(lo <= d < hi for lo, hi in _mte_lanes) for d in _diffs)
print(f"TAG-ONLY: {_tag_only}  (GST image diff bytes {_diffs} vs MTE applied lanes {_mte_lanes})")

elig = {r.sealing.value_reg for r in res._entries
        if r.sealing.__class__.__name__ == "MteSealing" and r.speculative and r.alts}
cer = ex._contract_executor.run(ex._make_ce_execution(
    code, inp, ex._sandbox_base, CONF.model_max_nesting, CONF.model_max_spec_window,
    ExecutionClause.COND | ExecutionClause.BPAS, mte_tags=ex._mte_tags_for(inp),
    pac_keys=ex._input_pac_keys(inp), pac_profile=ex._pac_profile_value))
cb = cer[0].cpu.pc
arch, spec = {}, {}
for ite in cer:
    nest = int(ite.metadata.speculation_nesting)
    for ma in ite.metadata.accesses():
        off = ite.cpu.pc - cb
        if not (0 <= off < len(code)): continue
        base = (int.from_bytes(code[off:off+4], "little") >> 5) & 0x1f
        for s in _cache_sets(ma, ite.cpu.gpr[29]):
            (spec if nest else arch).setdefault(s, (off, base))
arch_s, spec_s = set(arch), set(spec) - set(arch)
print(f"\nCE arch demand sets: {sorted(arch_s)}")
print(f"CE spec-only sets   : {sorted(spec_s)}  (leak-relevant)")

def nearest(s):
    best = None
    for D in sorted(arch_s | spec_s):
        k = (s - D) % 64
        if 1 <= k <= 16 and (best is None or k < best[1]): best = (D, k)
    return best

print("\ndiverging set -> trainer (leak is prefetch of a SPEC tag-sealed access):")
leak_regs, leak_sets = set(), []
for s in diverge:
    if s in spec_s: src, reg = "IS spec demand", f"x{spec[s][1]}"
    elif s in arch_s: src, reg = "arch demand (tag-independent - suspicious)", f"x{arch[s][1]}"
    else:
        nd = nearest(s)
        if nd and nd[0] in spec_s:
            src, reg = f"prefetch of SPEC demand {nd[0]} (+{nd[1]})", f"x{spec[nd[0]][1]}"
            leak_sets.append(s)
        elif nd and nd[0] in arch_s: src, reg = f"prefetch of arch demand {nd[0]} (+{nd[1]})", f"x{arch[nd[0]][1]}"
        else: src, reg = "UNACCOUNTED", "?"
    if "SPEC" in src or "IS spec" in src:
        if reg.lstrip("x").isdigit(): leak_regs.add(reg)
    print(f"  set {s:2d} -> {src}   (base {reg})")
print(f"\neligible tag-sealed regs: {sorted(elig)}")
print(f"=> feed the sweep:  REG in {sorted(leak_regs) or sorted(elig)}   LEAK_SETS = {sorted(set(leak_sets)) or diverge}")
