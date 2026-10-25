#!/usr/bin/env python3
"""Match-vs-mismatch PMU differential for an MTE NI suspect, using the executor's authoritative
baseline/decoy variants (variants_for_input) under the fixed-prefix swap-only protocol. Reports, per
variant, the probe-set residency and the mean per-victim-window PMU counts.

The three perf_counter columns depend on the loaded module's event selectors:
  pmutest build -> [INST_RETIRED(0x08), EXC_TAKEN(0x09), MEM_ACCESS(0x13)]
  canonical build -> [L2D_CACHE_REFILL(0x17), MEM_ACCESS(0x13), BR_MIS_PRED(0x22)]

Usage: pmu_variant.py <cfg> <vd> <sus> <probe_sets csv> [--reps N] [--prefix P] [--repeat R] [--thr T]
"""
import glob
import os
import sys


def residency(ht, nsets=64):
    raw = ht.raw
    return [0.0] * nsets if not raw else [sum((int(r) >> s) & 1 for r in raw) / len(raw) for s in range(nsets)]


def main():
    cfg, vd, sus = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3])
    probes = [int(x) for x in sys.argv[4].split(",")]

    def opt(n, d, c=int):
        return c(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d

    reps = opt("--reps", 2000); P = opt("--prefix", 14); R = opt("--repeat", 3); thr = opt("--thr", 0.5, float)
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0, repo)
    from src.config import CONF; CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize
    from src.aarch64.aarch64_executor import NIVariant
    wd = "/tmp/pmu_variant_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", ""); fz.initialize_modules()
    ex = fz.executor; ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))

    def load(i):
        return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0], "rb").read()).input_

    sinp = load(sus)
    sv = dict(ex.variants_for_input(sinp))
    G = sv[NIVariant.BASELINE]
    variants = [("baseline", G)] + [(str(k), v) for k, v in sv.items() if k != NIVariant.BASELINE]
    try:
        variants.append(("noncanon", ex.noncanon_variant(sinp)))
    except Exception:
        pass
    others = [i for i in range(P + 6) if i != sus][:P]
    prefix = [ex.genuine_variant(load(i)) for i in others]

    def one(v):
        hts, _ = ex.trace_test_case(prefix + [v], reps); ht = hts[-1]
        res = residency(ht); pc = ht.perf_counters
        means = [float(pc[:, c].mean()) for c in range(pc.shape[1])]
        lit = [s for s in range(64) if res[s] > thr]
        return res, means, lit

    print(f"{os.path.basename(vd)} sus#{sus} reps={reps} prefix={len(prefix)} repeat={R} thr={thr}")
    print("PMU cols: pmutest=[INST_RETIRED, EXC_TAKEN, MEM_ACCESS]  canonical=[L2D_REFILL, MEM_ACCESS, BR_MIS]")
    print(f"{'variant':>10} {'run':>3} | " + " ".join(f"s{p}" for p in probes) +
          " | width | " + f"{'c0':>10} {'c1':>10} {'c2':>10}")
    for label, v in variants:
        for r in range(R):
            res, means, lit = one(v)
            pv = " ".join(f"{res[p]:.2f}" for p in probes)
            width = f"n={len(lit)}[{min(lit) if lit else '-'}..{max(lit) if lit else '-'}]"
            print(f"{label:>10} {r:>3} | {pv} | {width} | {means[0]:10.2f} {means[1]:10.2f} {means[2]:10.2f}")


if __name__ == "__main__":
    main()
