#!/usr/bin/env python3
"""Triage a Revizor AArch64 CROSS-INPUT (generalized-priming / cross_input_priming) violation on
hardware.

A cross-input leak is not a single-slot v1/v4 leak: an earlier input (the LEAKING pair) leaves a
microarchitectural residue (e.g. a BTB entry it trains under a canonical vs non-canonical
branch-target seal) that only surfaces as a LATER input (the DETECTING pair)'s cache divergence. The
standard CE-driven triage.py cannot model that channel (and crashes on templates with unmasked
addresses), so this verifier uses the real executor as the arbiter.

It reproduces, at high reps, the control that defines a genuine cross-input leak, following the
`report.txt` "Cross-input priming (leaker/detector localization)" section (leaking pair lo,
detecting pair hi, Sequence A = all-genuine, Sequence B = the toggled lane):

  1. Own-slot indistinguishability (the "noncache" premise, def:xio(a)): the leaker's OWN htrace at
     its own position lo must be the SAME for its genuine and decoy variants; the leak is NOT in the
     leaker's own slot, only downstream. (Plus the report's identical contract-trace hashes.)
  2. Leaker/detector matrix at the detector (position hi): fix the prefix, then cross the leaker
     {genuine, decoy} with the detector seal {genuine, decoy} and read the detector's htrace. A
     GENUINE cross-input leak shows one or more sets that track the LEAKER's seal, INDEPENDENT
     of the detector's own seal:
         genuine leaker -> leak set present for BOTH detector variants
         decoy   leaker -> leak set absent  for BOTH detector variants
     Sets that are the same across all four cells are the detector's own footprint (architectural),
     not the leak.

Because every measurement is one real `trace_test_case` on /dev/executor, this is simultaneously the
manual reproduction (step 3/5). Run from the repo root with the venv python.

Usage:
  cross_input_triage.py <violation-dir> [--template templates/foo.py] [--reps 500] [--threshold 0.5]
  # --template: rebuild the TestCase from a .py template when generated.asm can't be re-parsed
  #             (dispatch tables etc.); omit to parse generated.asm directly.
"""
import argparse
import os
import re
import sys
import collections


def parse_finding(report: str):
    """Pull (lo, hi, seqA, seqB) from the cross-input priming section of report.txt."""
    lo = int(re.search(r"Leaking pair:\s*position\s*(\d+)", report).group(1))
    hi = int(re.search(r"Detecting pair:\s*position\s*(\d+)", report).group(1))
    seqA = [int(x) for x in re.search(r"Sequence A:\s*\[([^\]]*)\]", report).group(1).split(",")]
    seqB = [int(x) for x in re.search(r"Sequence B:\s*\[([^\]]*)\]", report).group(1).split(",")]
    return lo, hi, seqA, seqB


def set_freq(ht):
    """Per-cache-set '^' frequency over all reps. htrace.raw ints are identity (bit b == set b)."""
    acc = collections.defaultdict(int)
    n = len(ht.raw)
    for v in ht.raw:
        for b in range(64):
            if (v >> b) & 1:
                acc[b] += 1
    return {b: acc[b] / n for b in acc}, n


def load_test_case(fuzzer, vd, template):
    if template:
        from src.aarch64.template.runner import load_template, build_test_case
        return build_test_case(fuzzer.generator, load_template(template))
    return fuzzer.asm_parser.parse_file(f"{vd}/generated.asm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vd")
    ap.add_argument("--template", default=None)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="a set counts as PRESENT above this frequency (default 0.5)")
    a = ap.parse_args()
    vd = os.path.abspath(a.vd)
    sys.path.insert(0, os.getcwd())

    from src.config import CONF
    CONF.load(f"{vd}/reproduce.yaml")
    from src.factory import get_fuzzer
    from src.aarch64.aarch64_executor_input_encoder import deserialize

    report = open(f"{vd}/report.txt").read()
    lo, hi, seqA, seqB = parse_finding(report)

    f = get_fuzzer("base.json", ".", None, "")
    f.initialize_modules()
    tc = load_test_case(f, vd, a.template)
    f.executor.load_test_case(tc)

    def inp(i):
        return deserialize(open(f"{vd}/input_{i:04d}.reif", "rb").read())

    pin = getattr(CONF, 'executor_pinned_core', 0)

    if lo == hi:
        # Self-dependent (own-target confound): the pair leaks about its OWN seal. From a fixed
        # prefix, its own-slot htrace differs between its genuine and decoy variants; own-slot
        # DISTINGUISHABLE, so standard priming already catches it (it is NOT a noncache/cross-input
        # leak). Still a genuine leak: the input's own canonical branch evicts a set the
        # non-canonical one does not.
        prefix = [inp(seqA[p]) for p in range(lo)]
        variants = {"genuine": inp(seqA[lo]), "decoy": inp(seqB[lo])}
        print(f"# {vd}")
        print(f"# SELF-DEPENDENT finding @pos {lo} (genuine {seqA[lo]} / decoy {seqB[lo]})")
        print(f"# mode={CONF.executor_mode} pin={pin} reps={a.reps}\n")
        fr = {}
        for name, v in variants.items():
            hts, _ = f.executor.trace_test_case(prefix + [v], a.reps)
            fr[name], _ = set_freq(hts[lo])
        allsets = set(fr["genuine"]) | set(fr["decoy"])
        print(f"## Own-slot htrace at position {lo}, genuine vs decoy")
        print(f"   {'set':>4} {'genuine':>9} {'decoy':>9}")
        leak = []
        for b in sorted(allsets):
            gv, dv = fr["genuine"].get(b, 0), fr["decoy"].get(b, 0)
            mark = "  <== differs" if abs(gv - dv) > a.threshold else ""
            if abs(gv - dv) > a.threshold:
                leak.append(b)
            print(f"   {b:>4} {gv * 100:8.1f}% {dv * 100:8.1f}%{mark}")
        print("\n## Verdict")
        if leak:
            print(f"   LEAK sets (own seal changes own readout): {leak}")
            print("   -> GENUINE self-dependent (own-target) leak: the input's canonical branch")
            print(f"      evicts {leak}, the non-canonical one does not. Own-slot DISTINGUISHABLE")
            print("      standard priming already catches this (not a cross-input leak).")
        else:
            print(f"   no own-slot set difference > {a.threshold} -> washed out / noise")
        return

    # Sequence A is all-genuine; B differs at lo (leaker) and hi (detector). Genuine = A, decoy = B.
    prefix = [inp(seqA[p]) for p in range(lo)]              # identical context per measurement
    gap = [inp(seqA[p]) for p in range(lo + 1, hi)]         # positions between leaker and detector
    leaker = {"genuine": inp(seqA[lo]), "decoy": inp(seqB[lo])}
    detector = {"genuine": inp(seqA[hi]), "decoy": inp(seqB[hi])}

    print(f"# {vd}")
    print(f"# leaking pair @pos {lo} (genuine {seqA[lo]} / decoy {seqB[lo]}); "
          f"detecting pair @pos {hi} (genuine {seqA[hi]} / decoy {seqB[hi]})")
    print(f"# mode={CONF.executor_mode} pin={pin} reps={a.reps}\n")

    # --- Leaker self-leak note (informational): does the leaker also leak in its OWN slot? ---
    # This is NOT the noncache criterion for the flagged (detecting) violation; it just tells us
    # whether the same canonical branch also evicts in the leaker's own run (a co-located
    # self-dependent violation) or whether the leaker is a pure BTB trainer.
    own = {}
    for name, lk in leaker.items():
        hts, _ = f.executor.trace_test_case(prefix + [lk], a.reps)
        own[name], _ = set_freq(hts[lo])
    own_sets = set().union(*[set(own[k]) for k in own])
    own_spread = sorted(b for b in own_sets
                        if abs(own["genuine"].get(b, 0) - own["decoy"].get(b, 0)) >= 0.05)
    print(f"## Leaker self-leak (informational): own-slot htrace @pos {lo}, genuine vs decoy")
    print("   genuine sets:", {b: round(own['genuine'][b], 2) for b in sorted(own['genuine'])})
    print("   decoy   sets:", {b: round(own['decoy'][b], 2) for b in sorted(own['decoy'])})
    print("   -> " + (f"the leaker ALSO self-leaks at sets {own_spread} (co-located self-dependent"
                      " leak)" if own_spread
                      else "the leaker's own slot identical: PURE cross-input trainer") + "\n")

    # --- Steps 1-4: leaker x detector matrix at the detector (position hi) ---
    # The noncache premise (why standard priming misses this) is read straight off this matrix: the
    # leak set is INDEPENDENT of the detector's OWN seal (the two detector cells agree for each
    # leaker), i.e. the DETECTING pair is own-slot-indistinguishable; priming, which swaps only the
    # detector, sees nothing. The leak instead tracks the LEAKER's seal.
    print(f"## Steps 2-4: detector htrace (position {hi}) over leaker x detector seals")
    cells = {}
    allsets = set()
    for lk_name, lk in leaker.items():
        for dt_name, dt in detector.items():
            hts, _ = f.executor.trace_test_case(prefix + [lk] + gap + [dt], a.reps)
            fr, _ = set_freq(hts[hi])
            cells[(lk_name, dt_name)] = fr
            allsets |= set(fr)
    hdr = "".join(f"set{b:<5}" for b in sorted(allsets))
    print(f"   {'leaker':<9}{'detector':<10}{hdr}")
    for (lk_name, dt_name), fr in cells.items():
        row = "".join(f"{fr.get(b, 0) * 100:5.0f}% " for b in sorted(allsets))
        print(f"   {lk_name:<9}{dt_name:<10}{row}")

    # A leak set: present (>threshold) with genuine leaker for BOTH detectors, absent with decoy.
    def present(lk_name, b):
        return all(cells[(lk_name, dt)].get(b, 0) > a.threshold for dt in ("genuine", "decoy"))

    def absent(lk_name, b):
        return all(cells[(lk_name, dt)].get(b, 0) <= a.threshold for dt in ("genuine", "decoy"))

    leak_sets = sorted(b for b in allsets if present("genuine", b) and absent("decoy", b))
    own_footprint = sorted(b for b in allsets if present("genuine", b) and present("decoy", b))
    print("\n## Verdict")
    print(f"   detector own footprint (independent of leaker): sets {own_footprint}")
    if leak_sets:
        print(f"   LEAK sets (track the LEAKER's seal, independent of the detector's): {leak_sets}")
        print("   noncache premise: these sets are the SAME across the detector's own seal, so the")
        print("      detecting pair is own-slot-indistinguishable -> standard priming (which swaps")
        print("      only the detector) misses it; cross-input priming localizes it to the leaker.")
        print(f"   -> GENUINE cross-input leak: leaking pair {seqA[lo]}/{seqB[lo]} @pos {lo}")
        print(f"      trains an entry that changes detecting pair @pos {hi}'s readout at")
        print(f"      {leak_sets}, regardless of the detector's own seal. Not in the contract.")
    else:
        print("   NO set tracks the leaker independently of the detector -> NOT a cross-input leak")
        print("      (washed out / detector-seal-dependent / noise).")


if __name__ == "__main__":
    main()
