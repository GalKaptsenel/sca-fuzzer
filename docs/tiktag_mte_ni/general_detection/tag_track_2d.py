#!/usr/bin/env python3
"""Two-sided 2D tag track. Set the relevant speculative granule's MEMORY tag to T (device STG via
mte_tags[granule]) and, for each T, sweep the speculative POINTER tag at the gating seal (re-converged:
before=d, after=(Tdelta-d) so the pointer nets to cell-tag T and the perturbation is one access). A real
pointer-vs-memory tag check makes the target set SUPPRESS at the delta where the pointer tag equals T, and
that match-delta MOVES with T. Prints, per T, the per-delta residency, the measured dip, and the predicted
dip Tdelta=(C+T-cell_tag)%16.

Usage: tag_track_2d.py <cfg> <violation-dir> <suspect_idx> <target_set> <granule> [--reps N] [--gate-off 0xNN] [--tags a,b,c]
"""
import glob
import os
import sys

ADDG_OP = 0x91800000


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    sus, tset, gran = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1500
    force = int(sys.argv[sys.argv.index("--gate-off") + 1], 16) if "--gate-off" in sys.argv else None
    tags = [int(x) for x in sys.argv[sys.argv.index("--tags") + 1].split(",")] if "--tags" in sys.argv \
        else [3, 6, 9, 12]
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    wd = "/tmp/tag_track_2d_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm"); ex = fz.executor; ex.load_test_case(tc)
    inp = deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()).input_
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0]); relocs = list(g.code_reloc)
    cell_tag = int(g.mte_tags[gran])
    # DECOYED-GATE GUARD: only sweep seals the NI decoy actually perturbs (speculative-eligible). Sweeping
    # an architectural seal measures an architectural MTE effect, not a speculative leak (the 160128/021502
    # defect). Memoized resolve -> offsets match these relocs.
    rsv = ex._resolve(inp)
    decoy_offsets = set()
    for r in rsv._entries:
        if r.speculative and r.alts:
            offs = rsv._offsets.get(id(r.sealing))
            if offs:
                decoy_offsets.update(offs)

    def dput(val, d):
        return (val & ~(0xF << 10)) | ((d & 0xF) << 10)

    def ei(plan, mte):
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=mte,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    # pair seals: before(delta=c!=0) + after at +8 (same Xd, delta 0) = base-preserved; else persist single
    addg = [(k, r) for k, r in enumerate(relocs) if (r.value & 0xFFC00000) == ADDG_OP]
    by_off = {r.offset: k for k, r in addg}
    seals, used = [], set()
    for k, r in addg:
        if k in used:
            continue
        c = (r.value >> 10) & 0xF
        reg = r.value & 0x1f
        kaft = by_off.get(r.offset + 8)
        if c != 0 and kaft is not None and (relocs[kaft].value & 0x1f) == reg and ((relocs[kaft].value >> 10) & 0xF) == 0:
            seals.append((k, kaft, c, reg)); used.add(k); used.add(kaft)
        else:
            seals.append((k, None, c, reg)); used.add(k)

    def plan_set(kb, ka, correct, db):
        # before=db, after=(correct-db) -> net pointer tag = base+correct = cell tag `correct` implies
        p = list(relocs)
        p[kb] = Reloc(relocs[kb].offset, dput(relocs[kb].value, db), relocs[kb].rtype)
        if ka is not None:
            p[ka] = Reloc(relocs[ka].offset, dput(relocs[ka].value, (correct - db) % 16), relocs[ka].rtype)
        return p

    # choose the gate: forced, or the base-preserved seal with the largest confined effect on the set
    if force is not None:
        if force not in decoy_offsets:
            print(f"REFUSE: forced gate 0x{force:x} is NOT a decoyed (speculative) seal; decoy offsets="
                  f"{sorted(hex(o) for o in decoy_offsets)}. Sweeping it measures an architectural MTE "
                  f"effect, not a speculative leak. Aborting.")
            return
        sel = [(kb, ka, c, reg) for (kb, ka, c, reg) in seals if relocs[kb].offset == force]
        if not sel:
            print(f"forced gate 0x{force:x} is not a seal before-ADDG; abort"); return
        KB, KA, C, REG = sel[0]
    else:
        cand = [(kb, ka, c, reg) for (kb, ka, c, reg) in seals
                if ka is not None and relocs[kb].offset in decoy_offsets]  # base-preserved AND decoyed
        if not cand:
            print(f"REFUSE: no base-preserved DECOYED seal; decoy offsets="
                  f"{sorted(hex(o) for o in decoy_offsets)}. A 2D sweep needs a re-convergeable seal the "
                  f"NI decoy perturbs. Aborting.")
            return
        base_res = res(ex.trace_test_case([g], reps)[0][0], tset)
        knock = []
        for (kb, ka, c, reg) in cand:
            r = res(ex.trace_test_case([ei(plan_set(kb, ka, c, (c + 1) % 16), g.mte_tags)], reps)[0][0], tset)
            knock.append((abs(base_res - r), kb, ka, c, reg))
        knock.sort(reverse=True)
        _, KB, KA, C, REG = knock[0]
    print(f"{os.path.basename(vd)} set={tset} granule={gran} cell_tag={cell_tag} reps={reps}")
    print(f"gate=paired seal off=0x{relocs[KB].offset:x} x{REG} correct={C}")
    # TCMA tags 0 and 15 are unchecked; exclude the DELTAS that map to those pointer tags (tag space, not
    # delta space). During-access pointer tag at delta d = base_tag + d, base_tag = (cell_tag - C) % 16.
    base_tag = (cell_tag - C) % 16
    print(f"2D: rows=memory tag T (STG granule), cols=pointer delta 0..15; dip = tag MATCH "
          f"(base_tag={base_tag}, TCMA-excluded deltas={sorted(d for d in range(16) if (base_tag+d)%16 in (0,15))})")
    for T in tags:
        mte = list(g.mte_tags); mte[gran] = T
        Tdelta = (C + T - cell_tag) % 16
        row = {}
        for d in range(16):
            row[d] = res(ex.trace_test_case([ei(plan_set(KB, KA, Tdelta, d), mte)], reps)[0][0], tset)
        checked = [d for d in range(16) if (base_tag + d) % 16 not in (0, 15)]
        dip = min(checked, key=lambda d: row[d])
        print(f"  T={T:2d} predicted_match_delta={Tdelta:2d} measured_dip={dip:2d}  "
              + " ".join(f"{d}:{row[d]:.2f}" for d in range(16)))


if __name__ == "__main__":
    main()
