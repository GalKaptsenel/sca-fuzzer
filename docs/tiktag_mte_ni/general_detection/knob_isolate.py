#!/usr/bin/env python3
"""Pinpoint which sub-knob of the legacy enable_pre_run_flush (phr_flush vs view_rotation) is the source
of the MTE/TikTag signal. enable_pre_run_flush is a legacy combined sysfs knob that sets BOTH
executor.config.phr_flush and executor.config.view_rotation (src/aarch64/executor/sysfs.c
enable_pre_run_flush_store). measurement.c reads the two independently:
    view_rotation -> serve the next code view (invalidate_bpu_entries) -> tagged-BTB miss, per run
    phr_flush     -> zero the branch-history register once at the start of the trace
This measures a KNOWN-GOOD MTE gadget's match (genuine) vs mismatch (noncanon) probe-set residency under
all four (phr_flush, view_rotation) combinations; the combo(s) that preserve the match>mismatch gap are
the source of the cold-misprediction window the detector needs.

Usage: knob_isolate.py -c <hw.yml> -t <known_good.asm> [--set 32] [--reps 2000] [--n 2]
"""
import argparse
import os
import sys


def residency(htrace, s):
    raw = htrace.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-t", "--template", required=True)
    ap.add_argument("-s", "--spec", default="base.json")
    ap.add_argument("--set", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--repo", default="/home/gal_k_1_1998/revizor/sca-fuzzer")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    sys.path.insert(0, repo)
    spec, config, template = (os.path.join(repo, a.spec), os.path.abspath(a.config),
                              os.path.abspath(a.template))

    from src.config import CONF
    CONF.load(config)
    from src import factory
    from src.aarch64.aarch64_executor import SpecExcVariant
    from src.aarch64.aarch64_fuzzer import executor_regime

    wd = "/tmp/knob_isolate_wd"
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)
    fz = factory.get_fuzzer(spec, wd, template, "")
    fz.initialize_modules()
    tc = fz.generator.create_test_case_from_template(template)
    ex = fz.executor
    ex.load_test_case(tc)
    inputs = [inp for inp in fz.input_gen.generate(a.n) if ex.has_decoy(inp)]
    if not inputs:
        print("no input with a decoy; abort")
        return

    combos = [("1", "1"), ("1", "0"), ("0", "1"), ("0", "0")]
    print(f"template={os.path.basename(template)} set={a.set} reps={a.reps} "
          f"inputs_with_decoy={len(inputs)}")
    print("  (phr_flush, view_rotation) -> match(genuine) / mismatch(noncanon) / gap  [per input]")
    for phr, rot in combos:
        with executor_regime(ex, [("enable_phr_flush", phr), ("enable_view_rotation", rot)]):
            rows = []
            for inp in inputs:
                g = ex.genuine_variant(inp)
                nc = ex.noncanon_variant(inp)
                ht, _ = ex.trace_test_case([g, nc], a.reps)
                m, mm = residency(ht[0], a.set), residency(ht[1], a.set)
                rows.append((m, mm))
        desc = {("1", "1"): "both (==enable_pre_run_flush:1)", ("0", "0"): "neither (==flush:0)",
                ("1", "0"): "phr_flush ONLY", ("0", "1"): "view_rotation ONLY"}[(phr, rot)]
        cells = "  ".join(f"{m:.2f}/{mm:.2f}/{m-mm:+.2f}" for m, mm in rows)
        print(f"  phr={phr} rot={rot}  {desc:32s} {cells}")


if __name__ == "__main__":
    main()
