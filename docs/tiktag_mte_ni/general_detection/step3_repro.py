#!/usr/bin/env python3
"""Step 3 of triage: reproduce the fuzzer's EXACT baseline-vs-decoy htrace divergence under a controlled
fixed prefix, and test whether it SURVIVES a changed prefix (genuine, context-independent) or FLIPS/
VANISHES (batch/prefetcher contamination). Uses the real baseline + decoy variants from
variants_for_input (so the seal/re-merge is exactly what the fuzzer measured). Reads the suspect at the
LAST slot. No reloc surgery.

Usage: step3_repro.py <cfg> <violation-dir> <suspect_idx> <sets_csv> [--reps N]
"""
import glob
import os
import re as _re
import sys


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd, sus = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3])
    sets = [int(x) for x in sys.argv[4].split(",")]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1000
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize
    from src.aarch64.aarch64_executor import SpecExcVariant
    wd = "/tmp/step3_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm"); ex = fz.executor; ex.load_test_case(tc)

    def load(i):
        return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0], "rb").read()).input_

    sinp = load(sus)
    variants = ex.variants_for_input(sinp)
    baseline = variants[SpecExcVariant.BASELINE]
    decoys = [(k, v) for k, v in variants.items() if k != SpecExcVariant.BASELINE]

    others = sorted({int(m.group(1)) for p in glob.glob(f"{vd}/input_0*.*")
                     for m in [_re.search(r'input_(\d+)\.', os.path.basename(p))] if m and int(m.group(1)) != sus})
    base_prefix = [ex.genuine_variant(load(k)) for k in others]
    # prefix A = as-is; prefix B = reversed order (same inputs, different prefetcher/BPU context)
    prefixes = {"A(as-is)": base_prefix, "B(reversed)": list(reversed(base_prefix))}

    print(f"{os.path.basename(vd)} suspect#{sus} sets={sets} reps={reps} decoys={[k for k,_ in decoys]} "
          f"prefix_len={len(base_prefix)}")
    for pl, pref in prefixes.items():
        braw = ex.trace_test_case(pref + [baseline], reps)[0][-1]
        b = {s: res(braw, s) for s in sets}
        for dk, dv in decoys:
            draw = ex.trace_test_case(pref + [dv], reps)[0][-1]
            d = {s: res(draw, s) for s in sets}
            line = "  ".join(f"s{s}:{b[s]:.2f}->{d[s]:.2f}(Δ{d[s]-b[s]:+.2f})" for s in sets)
            print(f"  prefix {pl} | baseline vs {dk}: {line}")


if __name__ == "__main__":
    main()
