#!/usr/bin/env python3
"""Position-controlled all-64-set probe. Fixed prefix of genuine variants of OTHER inputs; the suspect
variant sits at the tail (one fixed slot => batch position held constant). Build the tail variant from
the genuine baseline by editing individual seal relocations (tag uimm4 and/or address uimm6) and/or by
retagging memory granules, then read residency of every L1D set at the tail. Prints the genuine baseline
once, the variant, and the per-set delta, so a single-seal perturbation or an address (anchor) move can be
read directly against the full set vector.

Edits:
  --edit 0xNN:u4=V      set the ADDG at code offset 0xNN to pointer-tag delta V (uimm4)
  --edit 0xNN:u6=V      set the ADDG at code offset 0xNN to address offset V granules (uimm6)
  --retag G:T           set memory tag of granule G to T (applies to every variant incl. prefix? no: tail only via mte)
  --name LABEL          label for the variant row
Several --edit/--retag may be given. Memory retags are applied to the tail variant's mte_tags only.

Usage: allsets.py <cfg> <vd> <suspect_idx> [--reps N] [--prefix P] [--edit ...] [--retag ...] [--name L]
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
    sus = int(sys.argv[3])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2000
    P = int(sys.argv[sys.argv.index("--prefix") + 1]) if "--prefix" in sys.argv else 14
    name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "variant"
    edits = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--edit"]
    retags = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--retag"]
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    wd = "/tmp/allsets_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    ex = fz.executor
    ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))

    def load(i):
        return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0], "rb").read()).input_

    inp = load(sus)
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0])
    relocs = list(g.code_reloc)

    def find_addg(off):
        ks = [k for k, r in enumerate(relocs) if (r.value & 0xFFC00000) == ADDG_OP and r.offset == off]
        if not ks:
            raise SystemExit(f"no ADDG reloc at offset 0x{off:x}; offsets="
                             f"{sorted(hex(r.offset) for r in relocs if (r.value & 0xFFC00000) == ADDG_OP)}")
        return ks[0]

    # apply edits to a fresh plan
    plan = list(relocs)
    applied = []
    for e in edits:
        off_s, kv = e.split(":")
        off = int(off_s, 16)
        field, val = kv.split("=")
        val = int(val)
        k = find_addg(off)
        v = plan[k].value
        if field == "u4":
            v = (v & ~(0xF << 10)) | ((val & 0xF) << 10)
        elif field == "u6":
            v = (v & ~(0x3F << 16)) | ((val & 0x3F) << 16)
        else:
            raise SystemExit(f"bad field {field}")
        plan[k] = Reloc(plan[k].offset, v, plan[k].rtype)
        applied.append(f"0x{off:x}:{field}={val}")

    mte = list(g.mte_tags)
    for rt in retags:
        gg, tt = rt.split(":")
        mte[int(gg)] = int(tt)
        applied.append(f"retag g{gg}->{tt}")

    variant = ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=mte,
                            pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    others = [i for i in range(P + 6) if i != sus][:P]
    prefix = [ex.genuine_variant(load(i)) for i in others]

    def tail(v):
        hts, _ = ex.trace_test_case(prefix + [v], reps)
        return residency(hts[-1])

    rg = tail(g)
    rv = tail(variant)
    print(f"{os.path.basename(vd)} suspect#{sus} reps={reps} prefix={len(prefix)} name={name}")
    print(f"edits: {applied}")
    hotg = [(s, round(rg[s], 2)) for s in range(64) if rg[s] > 0.15]
    hotv = [(s, round(rv[s], 2)) for s in range(64) if rv[s] > 0.15]
    print(f"  genuine resident(>0.15): {hotg}")
    print(f"  {name:10s} resident(>0.15): {hotv}")
    block = [1, 4, 5, 6, 8, 9, 10, 11]
    print(f"  block{block}: genuine={[round(rg[s],2) for s in block]}")
    print(f"  block{block}: {name}={[round(rv[s],2) for s in block]}")
    diff = [(s, round(rg[s], 2), round(rv[s], 2)) for s in range(64) if abs(rg[s] - rv[s]) > 0.15]
    print(f"  diff |Δ|>0.15 (set, genuine, {name}): {diff}")
    print(f"  FULL genuine: {[round(x,2) for x in rg]}")
    print(f"  FULL {name}: {[round(x,2) for x in rv]}")


if __name__ == "__main__":
    main()
