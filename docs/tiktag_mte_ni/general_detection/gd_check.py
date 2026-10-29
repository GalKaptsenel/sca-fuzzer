#!/usr/bin/env python3
"""Ground-truth speculative-exception check: measure the REAL genuine variant vs its decoy variant(s) vs noncanon of a saved
violation input, directly on the device, and report per-L1D-set residency. The decoy differs from genuine
ONLY in the speculative (decoyed) tag seals, so a robust genuine-vs-decoy divergence is the real
speculative-exception signal, independent of which gate a sweep picks. view_rotation must be ON (it is, via
enable_pre_run_flush in the config). Prints the sets where genuine and decoy diverge the most.

Usage: gd_check.py <cfg> <violation-dir> <suspect_idx> [--reps N] [--set S]
"""
import glob
import os
import sys


def residency(ht, nsets=64):
    raw = ht.raw
    return [0.0] * nsets if not raw else [sum((int(r) >> s) & 1 for r in raw) / len(raw) for s in range(nsets)]


def main():
    cfg, vd, sus = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 4000
    want = int(sys.argv[sys.argv.index("--set") + 1]) if "--set" in sys.argv else None
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize
    from src.aarch64.aarch64_executor import SpecExcVariant
    wd = "/tmp/gd_check_wd"
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm")
    ex = fz.executor
    ex.load_test_case(tc)
    inp = deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()).input_
    if not ex.has_decoy(inp):
        print("no decoy for this input; abort")
        return
    variants = dict(ex.variants_for_input(inp))
    variants["noncanon"] = ex.noncanon_variant(inp)
    names = list(variants)
    htraces, _ = ex.trace_test_case([variants[n] for n in names], reps)
    res = {n: residency(h) for n, h in zip(names, htraces)}
    base = res[SpecExcVariant.BASELINE]
    print(f"{os.path.basename(vd)} suspect#{sus} reps={reps}  variants={names}")
    print(f"view_rotation expected ON (enable_pre_run_flush={CONF.enable_pre_run_flush})")
    # resident sets per variant
    for n in names:
        hot = [(s, round(res[n][s], 2)) for s in range(64) if res[n][s] > 0.15]
        print(f"  {str(n):10s} resident(>0.15): {hot}")
    # genuine vs each decoy divergence
    print("  --- genuine(baseline) vs each other variant, |Δ|>0.15 ---")
    for n in names:
        if n == SpecExcVariant.BASELINE:
            continue
        diffs = [(s, round(base[s], 2), round(res[n][s], 2)) for s in range(64)
                 if abs(base[s] - res[n][s]) > 0.15]
        print(f"    {str(n):10s}: {diffs}")
    if want is not None:
        print(f"  --- set {want} residency per variant ---")
        for n in names:
            print(f"    {str(n):10s} set{want} = {res[n][want]:.2f}")


if __name__ == "__main__":
    main()
