#!/usr/bin/env python3
"""Position-controlled two-sided track of the DECOYED speculative seal. Fixed prefix of genuine variants
of other inputs; the suspect sits at the tail (one fixed slot, so batch position is held constant). For
each memory tag T of the tracked granule, sweep the decoyed seal's pointer tag (the ADDG at a decoyed
offset tagging the speculative access's base) 0..15 and read the target set at the tail slot. A genuine
speculative pointer-vs-memory tag check makes the target set's match point (gain-flavor: the dip, where
the access executes instead of being squashed) move with T. Only decoyed (speculative=True) seals are
sweepable; architectural seals are refused.

Usage: ctrl_store_track.py <cfg> <vd> <suspect_idx> <target_set> <granule> [--reps N] [--prefix P] [--tags a,b,c] [--gate-off 0xNN]
"""
import glob
import os
import sys

ADDG_OP = 0x91800000


def res1(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    sus, tset, gran = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3000
    P = int(sys.argv[sys.argv.index("--prefix") + 1]) if "--prefix" in sys.argv else 14
    tags = [int(x) for x in sys.argv[sys.argv.index("--tags") + 1].split(",")] if "--tags" in sys.argv else [3, 6, 9, 12]
    force = int(sys.argv[sys.argv.index("--gate-off") + 1], 16) if "--gate-off" in sys.argv else None
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    from src.aarch64.aarch64_executor import NIVariant
    wd = "/tmp/ctrl_store_track_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    ex = fz.executor
    ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))

    def load(i):
        return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0], "rb").read()).input_

    inp = load(sus)
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0]); relocs = list(g.code_reloc)
    cell_tag = int(g.mte_tags[gran])
    # decoyed (speculative) seal offsets
    rsv = ex._resolve(inp)
    decoy_offsets = set()
    for r in rsv._entries:
        if r.speculative and r.alts:
            offs = rsv._offsets.get(id(r.sealing))
            if offs:
                decoy_offsets.update(offs)
    # candidate ADDG gates = ADDG relocs at a decoyed offset
    addg_dec = [(k, relocs[k]) for k in range(len(relocs))
                if (relocs[k].value & 0xFFC00000) == ADDG_OP and relocs[k].offset in decoy_offsets]
    if force is not None:
        addg_dec = [(k, r) for (k, r) in addg_dec if r.offset == force]
    if not addg_dec:
        print(f"REFUSE: no decoyed ADDG gate (decoy offsets={sorted(hex(o) for o in decoy_offsets)}). Abort.")
        return

    prefix = [ex.genuine_variant(load(i)) for i in range(P + 6) if i != sus][:P]

    def dput(val, d):
        return (val & ~(0xF << 10)) | ((d & 0xF) << 10)

    def ei(kb, d, mte):
        plan = list(relocs)
        plan[kb] = Reloc(relocs[kb].offset, dput(relocs[kb].value, d), relocs[kb].rtype)
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=mte,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    def tail_set(variant):
        hts, _ = ex.trace_test_case(prefix + [variant], reps)
        return res1(hts[-1], tset)

    # pick the decoyed gate whose confined-at-tail knockout moves the target set most
    base_tail = tail_set(g)
    C = (relocs[addg_dec[0][0]].value >> 10) & 0xF
    knock = []
    for (k, r) in addg_dec:
        c = (r.value >> 10) & 0xF
        v = tail_set(ei(k, (c + 1) % 16, g.mte_tags))
        knock.append((abs(base_tail - v), k, c))
    knock.sort(reverse=True)
    _, KB, C = knock[0]
    print(f"{os.path.basename(vd)} suspect#{sus} set={tset} granule={gran} cell_tag={cell_tag} "
          f"reps={reps} prefix={len(prefix)} gate=0x{relocs[KB].offset:x} correct={C} base@tail={base_tail:.2f}")
    print("rows=memtag T (STG granule), cols=store-pointer delta 0..15 at TAIL; genuine match delta per T = (C+T-cell_tag)%16")
    for T in tags:
        mte = list(g.mte_tags); mte[gran] = T
        Tdelta = (C + T - cell_tag) % 16
        row = {d: tail_set(ei(KB, d, mte)) for d in range(16)}
        base_tag = (cell_tag - C) % 16
        checked = [d for d in range(16) if (base_tag + d) % 16 not in (0, 15)]
        dip = min(checked, key=lambda d: row[d])
        peak = max(checked, key=lambda d: row[d])
        print(f"  T={T:2d} match_delta={Tdelta:2d} dip={dip:2d} peak={peak:2d}  "
              + " ".join(f"{d}:{row[d]:.2f}" for d in range(16)))


if __name__ == "__main__":
    main()
