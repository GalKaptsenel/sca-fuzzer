#!/usr/bin/env python3
"""Anchor-move dose-response. Hold a decoyed seal at a MISMATCH tag (uimm4) and sweep its ADDG address
offset (uimm6, in granules) so the sealed speculative access lands in a moving L1D set. Position-controlled
(fixed prefix, tail slot). For each uimm6 read all 64 sets at the tail and print the lit run (>thr) plus its
[min,max] bound and the anchor's own set. If the lit run TRACKS the anchor across uimm6 -> the transmit is
anchored to the sealed speculative access (genuine, speculative). If it stays put -> fixed/architectural.

Usage: anchor_sweep.py <cfg> <vd> <sus> <gate 0xNN> <u4_mismatch> <u6a,u6b,...> [--reps N] [--prefix P]
                       [--repeat R] [--thr 0.5] [--anchor-granule G]
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
    sus = int(sys.argv[3]); gate = int(sys.argv[4], 16); u4 = int(sys.argv[5])
    u6s = [int(x) for x in sys.argv[6].split(",")]
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2000
    P = int(sys.argv[sys.argv.index("--prefix") + 1]) if "--prefix" in sys.argv else 14
    R = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.5
    ag = int(sys.argv[sys.argv.index("--anchor-granule") + 1]) if "--anchor-granule" in sys.argv else None
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    wd = "/tmp/anchor_sweep_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
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

    def mk(u4v, u6v):
        plan = list(relocs)
        v = plan[k].value
        v = (v & ~(0xF << 10)) | ((u4v & 0xF) << 10)
        v = (v & ~(0x3F << 16)) | ((u6v & 0x3F) << 16)
        plan[k] = Reloc(plan[k].offset, v, plan[k].rtype)
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=g.mte_tags,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    others = [i for i in range(P + 6) if i != sus][:P]
    prefix = [ex.genuine_variant(load(i)) for i in others]

    def tail(v):
        hts, _ = ex.trace_test_case(prefix + [v], reps)
        return residency(hts[-1])

    print(f"{os.path.basename(vd)} sus#{sus} gate=0x{gate:x} u4_mismatch={u4} reps={reps} prefix={len(prefix)} "
          f"repeat={R} thr={thr}")
    if ag is not None:
        print(f"anchor granule base={ag}: uimm6 K shifts anchor to granule {ag}+K, set=(({ag}+K)>>2)&63")
    for u6 in u6s:
        for rep in range(R):
            res = tail(mk(u4, u6))
            lit = [s for s in range(64) if res[s] > thr]
            run = f"[{min(lit)}..{max(lit)}]" if lit else "[]"
            aset = (((ag + u6) >> 2) & 63) if ag is not None else "?"
            print(f"  u6={u6:2d} rep{rep} anchor_set={aset} lit>{thr}={run}  n={len(lit)}  sets={lit}")


if __name__ == "__main__":
    main()
