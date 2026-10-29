#!/usr/bin/env python3
"""Separate BATCH POSITION from the TAG for a saved speculative-exception violation. gd_check puts baseline at slot 0 and
decoys at slots 1..N, so only the baseline gets the warm-up rounds + PHR flush -> a per-set difference
there is confounded with position. This uses a FIXED PREFIX of genuine variants of OTHER inputs and swaps
ONLY the suspect at a fixed slot (the skill's controlled protocol), and also measures the suspect genuine
at head vs tail to expose any pure position effect.

Prints, for the queried sets, residency of: genuine@tail, decoy@tail (same slot -> tag only),
genuine@head, decoy@head. If gen@tail ~= dec@tail the gd_check divergence was POSITION; if dec@tail
exceeds gen@tail beyond jitter it is the TAG.

Usage: ctrl_pos_tag.py <cfg> <vd> <suspect_idx> <set,set,...> [--reps N] [--prefix P]
"""
import glob
import os
import sys


def res1(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd, sus = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3])
    sets = [int(x) for x in sys.argv[4].split(",")]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3000
    P = int(sys.argv[sys.argv.index("--prefix") + 1]) if "--prefix" in sys.argv else 14
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize
    from src.aarch64.aarch64_executor import SpecExcVariant
    wd = "/tmp/ctrl_pos_tag_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    ex = fz.executor
    ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))

    def load(i):
        return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0], "rb").read()).input_

    dsel = sys.argv[sys.argv.index("--decoy") + 1] if "--decoy" in sys.argv else None
    sinp = load(sus)
    sv = dict(ex.variants_for_input(sinp))
    G = sv[SpecExcVariant.BASELINE]
    if dsel == "noncanon":
        dkey, Dc = "noncanon", ex.noncanon_variant(sinp)
    elif dsel is not None:
        dkey = next(k for k in sv if str(k) == dsel)
        Dc = sv[dkey]
    else:
        dkey = next(k for k in sv if k != SpecExcVariant.BASELINE)  # first decoy
        Dc = sv[dkey]
    # fixed prefix = genuine variants of other inputs (stable context, never the suspect)
    others = [i for i in range(P + 6) if i != sus][:P]
    prefix = [ex.genuine_variant(load(i)) for i in others]

    def tail(variant):
        hts, _ = ex.trace_test_case(prefix + [variant], reps)
        return hts[-1]

    def head(variant):
        hts, _ = ex.trace_test_case([variant] + prefix, reps)
        return hts[0]

    gt, dt, gh, dh = tail(G), tail(Dc), head(G), head(Dc)
    print(f"{os.path.basename(vd)} suspect#{sus} decoy={dkey} reps={reps} prefix={len(prefix)}")
    print(f"{'set':>4} | gen@tail dec@tail (tag, same slot) | gen@head dec@head | pos(gen tail-head)")
    for s in sets:
        a, b, c, d = res1(gt, s), res1(dt, s), res1(gh, s), res1(dh, s)
        tag_fixed = b - a
        pos_gen = a - c
        # a real tag effect at a FIXED slot is |Δtag|>0.15 (gain if +, shrink if -); otherwise, a swing
        # that only appears with position (large Δpos) is a batch artifact.
        if abs(tag_fixed) > 0.15:
            note = "TAG-gain" if tag_fixed > 0 else "TAG-shrink"
        elif abs(pos_gen) > 0.15 or abs(c - d) > 0.15:
            note = "position-only"
        else:
            note = "flat"
        print(f"{s:>4} |   {a:.2f}    {b:.2f}  (Δtag={tag_fixed:+.2f})   |   {c:.2f}    {d:.2f}   | Δpos={pos_gen:+.2f}  -> {note}")


if __name__ == "__main__":
    main()
