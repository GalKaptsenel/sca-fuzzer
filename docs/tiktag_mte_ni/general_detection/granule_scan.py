#!/usr/bin/env python3
"""Find which memory granule a decoyed gate's speculative access checks against, by retagging each
candidate granule to T and seeing whether the target set's suppression (match) delta moves from the
default-tag match (delta for tag 6) to the T match. Position-controlled (fixed prefix, tail slot).

Usage: granule_scan.py <cfg> <vd> <suspect_idx> <target_set> <gate-off 0xNN> <gran,gran,...> [--T N] [--reps N] [--prefix P]
"""
import glob, os, sys
ADDG_OP = 0x91800000
DEFAULT = 6


def res1(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    sus, tset = int(sys.argv[3]), int(sys.argv[4])
    gate = int(sys.argv[5], 16)
    grans = [int(x) for x in sys.argv[6].split(",")]
    T = int(sys.argv[sys.argv.index("--T") + 1]) if "--T" in sys.argv else 9
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2000
    P = int(sys.argv[sys.argv.index("--prefix") + 1]) if "--prefix" in sys.argv else 14
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0, repo)
    from src.config import CONF; CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    wd = "/tmp/granule_scan_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", ""); fz.initialize_modules()
    ex = fz.executor; ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
    def load(i): return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0], "rb").read()).input_
    inp = load(sus)
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0]); relocs = list(g.code_reloc)
    kb = next(k for k, r in enumerate(relocs) if (r.value & 0xFFC00000) == ADDG_OP and r.offset == gate)
    C = (relocs[kb].value >> 10) & 0xF
    base_tag = (DEFAULT - C) % 16
    d_def = (DEFAULT - base_tag) % 16   # delta giving pointer tag = 6 (match to default memory)
    d_T = (T - base_tag) % 16           # delta giving pointer tag = T (match if granule retagged to T)
    prefix = [ex.genuine_variant(load(i)) for i in range(P + 6) if i != sus][:P]
    def dput(v, d): return (v & ~(0xF << 10)) | ((d & 0xF) << 10)
    def meas(d, mte):
        plan = list(relocs); plan[kb] = Reloc(relocs[kb].offset, dput(relocs[kb].value, d), relocs[kb].rtype)
        v = ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=mte, pac_keys=g.pac_keys,
                          bpu_training=g.bpu_training, env_plan=g.env_plan)
        hts, _ = ex.trace_test_case(prefix + [v], reps); return res1(hts[-1], tset)
    print(f"{os.path.basename(vd)} set={tset} gate=0x{gate:x} C={C} base_tag={base_tag} "
          f"d(tag6)={d_def} d(tag{T})={d_T} reps={reps}")
    print(f"retag each granule->{T}; MATCH should move set {tset} suppression from d={d_def} to d={d_T}")
    for gi in grans:
        mte = list(g.mte_tags); mte[gi] = T
        a = meas(d_def, mte); b = meas(d_T, mte)
        tag = "  <-- TRACKS (match moved to d_T)" if (b < a - 0.3 and b < 0.3) else ""
        print(f"  granule {gi:4d} -> T={T}: set{tset}@d{d_def}(tag6)={a:.2f}  set{tset}@d{d_T}(tag{T})={b:.2f}{tag}")


if __name__ == "__main__":
    main()
