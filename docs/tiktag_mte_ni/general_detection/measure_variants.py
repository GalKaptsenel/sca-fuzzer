#!/usr/bin/env python3
"""Device diagnostic: measure the baseline (genuine) vs decoy (mistagged) sealed variants of a template
directly on /dev/executor and print per-L1D-set residency (fraction of reps the set is L1-resident),
highlighting sets that diverge between the variants. Answers: does the speculative body reach the probe
set (window opens), and does the mistagged decoy shrink it (TikTag divergence)?

Usage: measure_variants.py -s base.json -c <hw.yml> -t <template.asm> [--n 3] [--reps 2000]
"""
import argparse
import os
import sys


def residency(htrace, nsets=64):
    raw = htrace.raw
    if not raw:
        return [0.0] * nsets
    return [sum((int(r) >> s) & 1 for r in raw) / len(raw) for s in range(nsets)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--spec", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-t", "--template", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--repo", default="/home/gal_k_1_1998/revizor/sca-fuzzer")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    spec, config, template = map(os.path.abspath, (args.spec, args.config, args.template))

    from src.config import CONF
    CONF.load(config)
    from src import factory
    from src.aarch64.aarch64_executor import NIVariant

    wd = "/tmp/measure_variants_wd"
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)

    fuzzer = factory.get_fuzzer(spec, wd, template, "")
    fuzzer.initialize_modules()
    if template.endswith(".py"):
        from src.aarch64.template.runner import load_template, build_test_case
        tc = build_test_case(fuzzer.generator, load_template(template))
    else:
        tc = fuzzer.generator.create_test_case_from_template(template)
    ex = fuzzer.executor
    ex.load_test_case(tc)
    inputs = fuzzer.input_gen.generate(args.n)

    for i, inp in enumerate(inputs):
        if not ex.has_decoy(inp):
            print(f"input {i}: no decoy, skip")
            continue
        variants = ex.variants_for_input(inp)
        # Add the forced-all-bad lane (every eligible speculative slot mistagged) for a clean
        # mechanistic test of whether mistagging shrinks the probe at all.
        variants = dict(variants)
        variants["noncanon"] = ex.noncanon_variant(inp)
        names = list(variants.keys())
        exec_inputs = [variants[n] for n in names]
        htraces, _ = ex.trace_test_case(exec_inputs, args.reps)
        res = {n: residency(h) for n, h in zip(names, htraces)}
        base = res[NIVariant.BASELINE]
        print(f"\n=== input {i} seed={getattr(inp,'seed','?')} reps={args.reps} ===")
        # resident sets per variant
        for n in names:
            hot = [(s, round(res[n][s], 2)) for s in range(64) if res[n][s] > 0.1]
            print(f"  {n:10s} resident sets (>0.1): {hot}")
        # divergence baseline vs each decoy
        for n in names:
            if n == NIVariant.BASELINE:
                continue
            diffs = [(s, round(base[s], 2), round(res[n][s], 2))
                     for s in range(64) if abs(base[s] - res[n][s]) > 0.15]
            print(f"  DIVERGE baseline vs {n} (|d|>0.15): {diffs}")


if __name__ == "__main__":
    main()
