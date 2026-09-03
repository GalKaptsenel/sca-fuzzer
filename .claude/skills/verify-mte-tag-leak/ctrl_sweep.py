#!/usr/bin/env python3
"""Controlled MTE tag sweep (Step 3+4 of verify-mte-tag-leak).
Usage: ctrl_sweep.py <config.yml> <violation-dir> <REG> <suspect_input_idx> <leak_sets_csv> [reps] [repeats]

Fixed prefix of 14 baseline variants (of OTHER inputs), swap ONLY the suspect at the last slot with the
sealed register REG retagged to each of 16 tag deltas; one measurement per (delta, repeat). A genuine
tag-check (TikTag-v1) leak makes the MATCH delta stand out (highest leak-sum, all mismatches lower);
noise fires uniformly. Device — pause any live campaign first. Use a superbatch_size:1 config."""
import sys, random, glob, os
REPO = "/home/gal_k_1_1998/sca-fuzzer"
sys.path.insert(0, REPO)
from src.config import CONF
CFG, VD, REG, SUS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
LEAK = [int(x) for x in sys.argv[5].split(",")]
REPS = int(sys.argv[6]) if len(sys.argv) > 6 else 500
REPEATS = int(sys.argv[7]) if len(sys.argv) > 7 else 6
CONF.load(CFG)
from src.fuzzer import NoninterferenceFuzzer
from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
from src.aarch64.seal.sealer import Relocation, _encode
from src.aarch64.aarch64_executor import NIVariant

fz = NoninterferenceFuzzer("base.json", REPO); fz.initialize_modules()
ex = fz.executor
tc = fz.asm_parser.parse_file(f"{VD}/generated.asm"); ex.load_test_case(tc)
avail = sorted(int(os.path.basename(p)[6:10]) for p in glob.glob(f"{VD}/input_0*.bin"))
pref_idx = [k for k in avail if k != SUS and k < 100][:14]
prefix = [ex.variants_for_input(deserialize(open(f"{VD}/input_{k:04d}.bin", "rb").read()).input_)[NIVariant.BASELINE]
          for k in pref_idx]
inp = deserialize(open(f"{VD}/input_{SUS:04d}.bin", "rb").read()).input_
res = ex._resolve(inp)
genuine = next(r.value for r in res._entries
               if getattr(r.sealing, "value_reg", None) == REG and r.speculative and r.alts)
print(f"{os.path.basename(VD)} suspect inp{SUS}, sweep {REG} (MATCH delta={genuine}) leak={LEAK} reps={REPS}x{REPEATS}")

def ei_for(delta):
    rng = random.Random(1); out = []
    for r in res._entries:
        offs = res._offsets.get(id(r.sealing))
        if offs is None: continue
        val = delta if (getattr(r.sealing, "value_reg", None) == REG and r.speculative and r.alts) else r.value
        out += [Relocation(off, _encode(i)) for off, i in zip(offs, r.sealing.seal(val, rng))]
    return ExecutorInput(inp, code_reloc=tuple(out), mte_tags=ex._mte_tags_for(inp),
                         pac_keys=ex._pac_keys_words(), bpu_training=ex._bpu_entries(inp))

variants = {d: ei_for(d) for d in range(16)}
agg = {d: {s: [] for s in LEAK} for d in range(16)}
for rep in range(REPEATS):
    for d in range(16):
        raw = ex.trace_test_case(prefix + [variants[d]], REPS)[0][-1].raw; n = len(raw) or 1
        for s in LEAK:
            agg[d][s].append(sum((r >> s) & 1 for r in raw) / n)
    print(f"  repeat {rep+1}/{REPEATS}", flush=True)
print(f"\ndelta match?  " + "  ".join(f"s{s}" for s in LEAK) + "   sum")
for d in range(16):
    m = {s: sum(agg[d][s]) / len(agg[d][s]) for s in LEAK}
    print(f" {d:2d}  {'MATCH ' if d == genuine else 'mismis'}  " +
          "  ".join(f"{m[s]:.2f}" for s in LEAK) + f"   {sum(m.values()):.2f}")
gm = sum(sum(agg[genuine][s]) / len(agg[genuine][s]) for s in LEAK)
mm = [sum(sum(agg[d][s]) / len(agg[d][s]) for s in LEAK) for d in range(16) if d != genuine]
print(f"\nMATCH sum={gm:.2f}   MISMATCH mean={sum(mm)/len(mm):.2f} min={min(mm):.2f} max={max(mm):.2f}")
print("VERDICT: GENUINE tag leak iff MATCH sum is clearly ABOVE every mismatch (beyond repeat jitter); "
      "else prefetcher/measurement NOISE.")
