#!/usr/bin/env python3
"""Pointer-tag sweep of one decoyed seal (polarity + tag-check test). Hold the seal's address (uimm6) and
sweep its pointer-tag delta uimm4 0..15; the pointer tag = (base_ptr_tag + uimm4) % 16 where base_ptr_tag
is read from the genuine (uimm4 that matches memory). For each uimm4 read all 64 sets at the tail
(fixed prefix) and report residency at the probe sets + the width of the lit stream. A genuine tag-check
transmit: the stream is ABSENT at the match tag (pointer tag == memory tag) and present at mismatches;
TCMA-unchecked pointer tags 0/15 behave like no-check. Prints the implied pointer tag and match flag.

Usage: tag_sweep.py <cfg> <vd> <sus> <gate 0xNN> <probe_sets csv> [--reps N] [--prefix P] [--repeat R]
                    [--memtag M] [--thr 0.5] [--u6 K]
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
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2000
    P = int(sys.argv[sys.argv.index("--prefix") + 1]) if "--prefix" in sys.argv else 14
    R = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    memtag = int(sys.argv[sys.argv.index("--memtag") + 1]) if "--memtag" in sys.argv else 6
    thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.5
    u6 = int(sys.argv[sys.argv.index("--u6") + 1]) if "--u6" in sys.argv else 0
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    wd = "/tmp/tag_sweep_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
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
    C = (relocs[k].value >> 10) & 0xF          # genuine uimm4 (match)
    base_ptr_tag = (memtag - C) % 16            # pointer tag produced by uimm4=0

    def mk(u4v):
        plan = list(relocs)
        v = plan[k].value
        v = (v & ~(0xF << 10)) | ((u4v & 0xF) << 10)
        v = (v & ~(0x3F << 16)) | ((u6 & 0x3F) << 16)
        plan[k] = Reloc(plan[k].offset, v, plan[k].rtype)
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=g.mte_tags,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    others = [i for i in range(P + 6) if i != sus][:P]
    prefix = [ex.genuine_variant(load(i)) for i in others]

    def tail(v):
        hts, _ = ex.trace_test_case(prefix + [v], reps)
        return residency(hts[-1])

    print(f"{os.path.basename(vd)} sus#{sus} gate=0x{gate:x} genuine_uimm4(match)={C} memtag={memtag} "
          f"u6={u6} reps={reps} repeat={R} thr={thr}")
    print(f"pointer tag = ({base_ptr_tag} + uimm4) % 16 ; MATCH when pointer tag == {memtag} (uimm4={C}); "
          f"TCMA-unchecked when pointer tag in (0,15)")
    print(f"{'u4':>3} {'ptag':>4} {'flag':>9} | " + " ".join(f"s{p}" for p in probes) + " | streamwidth")
    for u4 in range(16):
        ptag = (base_ptr_tag + u4) % 16
        flag = "MATCH" if ptag == memtag else ("UNCHK" if ptag in (0, 15) else "mismatch")
        for rep in range(R):
            res = tail(mk(u4))
            lit = [s for s in range(64) if res[s] > thr]
            pv = " ".join(f"{res[p]:.2f}" for p in probes)
            print(f"{u4:>3} {ptag:>4} {flag:>9} | {pv} | n={len(lit)} [{min(lit) if lit else '-'}..{max(lit) if lit else '-'}]")


if __name__ == "__main__":
    main()
