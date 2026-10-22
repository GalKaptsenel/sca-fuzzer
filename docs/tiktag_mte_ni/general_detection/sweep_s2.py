#!/usr/bin/env python3
"""Sweep the gate length (and optionally SDIV depth) to find the misprediction-window EDGE and test
whether the forced-all-bad (noncanon) variant falls off the edge earlier than baseline — i.e. whether
a speculative MTE tag mismatch shrinks the window. For each (sdiv,gap) it generates the template, builds
the NI pipeline, and measures baseline vs noncanon on `n` inputs, reporting the hottest L1D set's
residency for each variant.

Usage: sweep_s2.py -s base.json -c <hw.yml> --gaps 160,300,440,560 --sdiv 24 --n 2 --reps 1500
"""
import argparse
import importlib.util
import os
import sys

sys.setrecursionlimit(200000)


def residency(htrace, nsets=64):
    raw = htrace.raw
    if not raw:
        return [0.0] * nsets
    return [sum((int(r) >> s) & 1 for r in raw) / len(raw) for s in range(nsets)]


def load_gen(gd):
    spec = importlib.util.spec_from_file_location("gen_s2", os.path.join(gd, "gen_s2.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--spec", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--gaps", default="160,300,440,560")
    ap.add_argument("--sdiv", type=int, default=24)
    ap.add_argument("--ldnum", type=int, default=16)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--reps", type=int, default=1500)
    ap.add_argument("--repo", default="/home/gal_k_1_1998/revizor/sca-fuzzer")
    a = ap.parse_args()

    repo = os.path.abspath(a.repo)
    sys.path.insert(0, repo)
    gd = os.path.join(repo, "docs/tiktag_mte_ni/general_detection")
    spec, config = os.path.abspath(a.spec), os.path.abspath(a.config)
    gen = load_gen(gd)

    from src.config import CONF
    CONF.load(config)
    from src import factory
    from src.aarch64.aarch64_executor import NIVariant

    wd = "/tmp/sweep_s2_wd"
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)

    gaps = [int(x) for x in a.gaps.split(",")]
    print(f"sdiv={a.sdiv} ldnum={a.ldnum} reps={a.reps}  (hottest-set residency: baseline -> noncanon)")
    for gap in gaps:
        tpath = os.path.join(gd, f"s2_sweep_{a.ldnum}_{gap}_{a.sdiv}.asm")
        with open(tpath, "w") as f:
            f.write(gen.gen(a.ldnum, gap, a.sdiv, True))
        fuzzer = factory.get_fuzzer(spec, wd, tpath, "")
        fuzzer.initialize_modules()
        tc = fuzzer.generator.create_test_case_from_template(tpath)
        ex = fuzzer.executor
        ex.load_test_case(tc)
        inputs = fuzzer.input_gen.generate(a.n)
        rows = []
        for i, inp in enumerate(inputs):
            if not ex.has_decoy(inp):
                continue
            allv = ex.variants_for_input(inp)   # baseline + random decoy0
            variants = dict(allv)
            variants["noncanon"] = ex.noncanon_variant(inp)
            names = list(variants.keys())
            htraces, _ = ex.trace_test_case([variants[n] for n in names], a.reps)
            res = {n: residency(h) for n, h in zip(names, htraces)}
            b = res[NIVariant.BASELINE]
            probe = 32
            dec = res.get(NIVariant.decoy_n(0), b)
            nc = res["noncanon"]
            rows.append(f"in{i}: probe32 base{b[probe]:.2f} decoy{dec[probe]:.2f} nc{nc[probe]:.2f}")
        print(f"  gap={gap:4d}: " + "   ".join(rows))


if __name__ == "__main__":
    main()
