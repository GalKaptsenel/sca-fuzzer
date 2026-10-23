#!/usr/bin/env python3
"""Match-vs-mismatch PMU + residency differential for one decoyed MTE seal (mechanism proof for 072421).

Fixed-prefix / swap-only-suspect protocol (same as tag_sweep/ctrl_pos_tag): a stable prefix of genuine
variants of OTHER inputs, the suspect at the tail slot, and only the suspect seal's pointer-tag delta
(ADDG uimm4 at `gate`) is changed. For each requested uimm4 it runs R independent trace batches of `reps`
each and reports, at the tail:
  - residency at the probe sets and the lit-set block [min..max] (width) at threshold `thr`
  - mean per-victim-window PMU counts: pfc0=L2D_CACHE_REFILL, pfc1=MEM_ACCESS(demand), pfc2=BR_MIS_PRED
  - excess = L2D_CACHE_REFILL - MEM_ACCESS  (refills beyond demand accesses == prefetch traffic)
R independent batches give a spread so stability is visible (>=3 reps requirement).

Pointer tag = (base_ptr_tag + uimm4) % 16 ; base_ptr_tag = (memtag - genuine_uimm4) % 16.
MATCH when pointer tag == memtag; UNCHK (TCMA) when pointer tag in (0,15); else mismatch.

Usage: pmu_diff.py <cfg> <vd> <sus> <gate 0xNN> <probe_sets csv>
                   [--reps N] [--prefix P] [--memtag M] [--repeat R] [--thr T]
                   [--u4 a,b,c | --all] [--u6 K]
"""
import glob
import os
import sys

ADDG_OP = 0x91800000


def residency(ht, nsets=64):
    raw = ht.raw
    return [0.0] * nsets if not raw else [sum((int(r) >> s) & 1 for r in raw) / len(raw) for s in range(nsets)]


def main():
    cfg, vd = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    sus = int(sys.argv[3]); gate = int(sys.argv[4], 16)
    probes = [int(x) for x in sys.argv[5].split(",")]

    def opt(name, default, cast=int):
        return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

    reps = opt("--reps", 2000); P = opt("--prefix", 14); memtag = opt("--memtag", 6)
    R = opt("--repeat", 3); thr = opt("--thr", 0.5, float); u6 = opt("--u6", 0)
    if "--all" in sys.argv:
        u4list = list(range(16))
    elif "--u4" in sys.argv:
        u4list = [int(x) for x in sys.argv[sys.argv.index("--u4") + 1].split(",")]
    else:
        u4list = None  # set to [match, match+1] after we know C

    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    if "--noflush" in sys.argv:
        CONF.enable_pre_run_flush = False  # view_rotation OFF -> no misprediction window
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    wd = "/tmp/pmu_diff_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    ex = fz.executor
    ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))

    def load(i):
        return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0], "rb").read()).input_

    inp = load(sus)
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0]); relocs = list(g.code_reloc)
    k = next(j for j, r in enumerate(relocs) if (r.value & 0xFFC00000) == ADDG_OP and r.offset == gate)
    C = (relocs[k].value >> 10) & 0xF
    base_ptr_tag = (memtag - C) % 16
    if u4list is None:
        u4list = [C, (C + 1) % 16]

    notrain = "--notrain" in sys.argv

    def mk(u4v):
        plan = list(relocs)
        v = plan[k].value
        v = (v & ~(0xF << 10)) | ((u4v & 0xF) << 10)
        v = (v & ~(0x3F << 16)) | ((u6 & 0x3F) << 16)
        plan[k] = Reloc(plan[k].offset, v, plan[k].rtype)
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=g.mte_tags,
                             pac_keys=g.pac_keys,
                             bpu_training=() if notrain else g.bpu_training, env_plan=g.env_plan)

    others = [i for i in range(P + 6) if i != sus][:P]
    prefix = [ex.genuine_variant(load(i)) for i in others]
    if notrain:  # strip per-input BPU mistraining from the whole batch (window-off test)
        prefix = [ExecutorInput(p.input_, code_reloc=p.code_reloc, mte_tags=p.mte_tags,
                                pac_keys=p.pac_keys, bpu_training=(), env_plan=p.env_plan)
                  for p in prefix]

    def one(v):
        hts, _ = ex.trace_test_case(prefix + [v], reps)
        ht = hts[-1]
        res = residency(ht)
        pc = ht.perf_counters  # (reps, 3): L2D_REFILL, MEM_ACCESS, BR_MIS_PRED
        means = [float(pc[:, c].mean()) for c in range(pc.shape[1])]
        lit = [s for s in range(64) if res[s] > thr]
        return res, means, lit

    print(f"{os.path.basename(vd)} sus#{sus} gate=0x{gate:x} genuine_uimm4(match)={C} memtag={memtag} "
          f"u6={u6} reps={reps} prefix={len(prefix)} repeat={R} thr={thr}")
    print(f"ptag=({base_ptr_tag}+u4)%16; MATCH@ptag{memtag}(u4={C}); UNCHK@ptag0/15")
    print("pfc0=L2D_CACHE_REFILL  pfc1=MEM_ACCESS(demand)  pfc2=BR_MIS_PRED   excess=pfc0-pfc1")
    hdr = f"{'u4':>3} {'ptag':>4} {'flag':>8} {'run':>3} | " + " ".join(f"s{p}" for p in probes)
    hdr += " | width | " + f"{'L2D':>9} {'MEMACC':>9} {'BRMIS':>8} {'excess':>9}"
    print(hdr)
    for u4 in u4list:
        ptag = (base_ptr_tag + u4) % 16
        flag = "MATCH" if ptag == memtag else ("UNCHK" if ptag in (0, 15) else "mismtch")
        for r in range(R):
            res, means, lit = one(mk(u4))
            pv = " ".join(f"{res[p]:.2f}" for p in probes)
            width = f"n={len(lit)}[{min(lit) if lit else '-'}..{max(lit) if lit else '-'}]"
            exc = means[0] - means[1]
            print(f"{u4:>3} {ptag:>4} {flag:>8} {r:>3} | {pv} | {width} | "
                  f"{means[0]:9.2f} {means[1]:9.2f} {means[2]:8.2f} {exc:9.2f}")


if __name__ == "__main__":
    main()
