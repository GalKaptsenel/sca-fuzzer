#!/usr/bin/env python3
"""Hardware verification of a Revizor AArch64 violation via the controlled batch-context protocol.

PITFALL this defends against: an input's htrace depends on its predecessors in the SAME test case's
input batch (prefetcher/BPU context carries WITHIN a test case; the executor resets between test cases,
so DO NOT reiterate other test cases). Measuring counterexample inputs in isolation gives them different
contexts and spurious differences. Here we build ONE identical batch context and swap only the slot
under test between the V counterexample inputs, at a chosen position, then measure the swapped input's
htrace DISTRIBUTION at high reps and report whether the difference persists (genuine) or washes out
(noise / batch-contamination false positive).

Runs local OR remote exactly as reproduce.yaml's executor_* selects (no separate code path).

Usage:
  verify.py <violation-dir> [--repo DIR] [--inputs A,B,...] [--position lower|higher|last|all]
            [--reps 2000] [--sets 14,55]      # predicted leaking SETS from triage.py (identity/raw space)

Position (the prefix is slots 0..P-1, identical for every measurement; only slot P is swapped):
  lower  -> P = min(counterexample idxs)   (prime up to the first leaking slot)
  higher -> P = max(counterexample idxs)   (prime up to the last leaking slot)
  last   -> P = N-1                        (all iids as context, swap the final slot; most symmetric)
  all    -> run lower, higher and last and cross-check they agree
"""
import argparse, os, re, sys, glob, collections

def find_repo(start):
    d = os.path.abspath(start)
    while d != "/":
        if os.path.isdir(os.path.join(d, "src", "aarch64")):
            return d
        d = os.path.dirname(d)
    sys.exit("could not locate repo root; pass --repo")

def parse_inputs(report):
    return [int(m) for m in re.findall(r'Input #(\d+)\n\* Hardware trace:', report)]

def freq(ht):
    """per-SET '^' frequency over ALL reps (raw htrace int: bit b == cache set b, identity map)."""
    acc = collections.defaultdict(int); n = len(ht.raw)
    for v in ht.raw:
        for b in range(64):
            if (v >> b) & 1:
                acc[b] += 1
    return {b: acc[b] / n for b in acc}, n

def run_position(f, allex, ce_inputs, P, reps, predicted, mode, core):
    assert 1 <= P < len(allex), f"position slot {P} out of range for {len(allex)} inputs"
    prefix = allex[:P]                                   # identical context for every measurement
    print(f"\n# mode={mode} core={core} reps={reps} position slot={P} (prefix 0..{P-1}) inputs={ce_inputs}")
    fr = {}
    n = 0
    for idx in ce_inputs:
        hts, _ = f.executor.trace_test_case(prefix + [allex[idx]], reps)   # one batch trace call
        fr[idx], n = freq(hts[-1])                       # htrace of the swapped last slot only
    bins = set().union(*[set(fr[i]) for i in ce_inputs])
    rows = []
    for b in sorted(bins):
        vals = [fr[i].get(b, 0) for i in ce_inputs]
        if max(vals) - min(vals) >= 0.03:                # distribution spread, not just dominant
            rows.append((b, max(vals) - min(vals), vals))
    if not rows:
        print(f"  reps={n}: NO set spreads >=3% across inputs -> INDISTINGUISHABLE in identical context")
        print("  VERDICT: violation was NOISE / batch-contamination (washed out)")
        return set()
    print(f"  reps={n}: persistent per-set spread (>=3%):")
    for b, sp, vals in rows:
        mark = " [PREDICTED]" if predicted and b in predicted else ""
        print(f"    set {b:>2}: spread={sp*100:4.1f}%  ({' '.join(f'{v*100:4.1f}%' for v in vals)}){mark}")
    persist = {b for b, _, _ in rows}
    if predicted is not None:
        hit = sorted(persist & predicted)
        print(f"  VERDICT: persistence at predicted sets {sorted(predicted)}: "
              + (f"{hit} -> GENUINE" if hit else "NONE -> washed out at predicted sets -> NOISE"))
    return persist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vd"); ap.add_argument("--repo")
    ap.add_argument("--inputs"); ap.add_argument("--position", default="last",
                                                 choices=["lower", "higher", "last", "all"])
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--sets")
    a = ap.parse_args()
    vd = os.path.abspath(a.vd); repo = a.repo or find_repo(vd); sys.path.insert(0, repo)
    from src.config import CONF; CONF.load(f"{vd}/reproduce.yaml")
    from src.factory import get_fuzzer
    report = open(f"{vd}/report.txt").read()
    ce_inputs = [int(x) for x in a.inputs.split(",")] if a.inputs else parse_inputs(report)
    assert len(ce_inputs) >= 2, "need >=2 counterexample inputs"
    predicted = {int(x) for x in a.sets.split(",")} if a.sets else None

    f = get_fuzzer("base.json", ".", None, ""); f.initialize_modules()
    tc = f.asm_parser.parse_file(f"{vd}/generated.asm")
    paths = sorted(glob.glob(f"{vd}/input_*.reif")) or sorted(glob.glob(f"{vd}/input_*.bin"))
    assert paths, "no input_*.reif/.bin in violation dir"
    allinp = f.input_gen.load(paths)
    f.executor.load_test_case(tc)
    allex = [f.executor.as_executor_input(x) for x in allinp]
    N = len(allex)
    core = getattr(CONF, "executor_pinned_core", 0)
    positions = {"lower": min(ce_inputs), "higher": max(ce_inputs), "last": N - 1}
    todo = list(positions) if a.position == "all" else [a.position]
    results = {p: run_position(f, allex, ce_inputs, positions[p], a.reps, predicted,
                               CONF.executor_mode, core) for p in todo}
    if len(todo) > 1:
        print("\n# cross-position check (a real leak persists at every position):")
        for p in todo:
            print(f"  {p:>6} (slot {positions[p]}): persistent sets = {sorted(results[p]) or 'NONE'}")

if __name__ == "__main__":
    main()
