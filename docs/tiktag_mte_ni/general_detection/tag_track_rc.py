#!/usr/bin/env python3
"""Re-converge-aware pointer-tag sweep. The MTE sealer wraps a base-preserved access with a BEFORE ADDG
(#delta) and an AFTER ADDG (#(correct-delta)) so the tag re-converges to the cell tag after the access
(perturbation confined to that one access). A sweep MUST move BOTH: set before=d and after=(correct-d),
else the swept tag persists downstream and muddies everything. This script pairs each seal's before/after
relocs, knocks out each (confined) to find which gates the target set, and sweeps that seal's pointer tag
(re-converged) over 0..15. Persist/writeback seals (no after) are swept before-only and flagged.

Usage: tag_track_rc.py <cfg> <violation-dir> <suspect_idx> <target_set> [--reps N]
"""
import glob
import os
import sys

ADDG_OP = 0x91800000


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd, sus, tset = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), \
        int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2000
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    wd = "/tmp/tag_track_rc_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm"); ex = fz.executor; ex.load_test_case(tc)
    inp = deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()).input_
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0]); relocs = list(g.code_reloc)
    # DECOYED-GATE GUARD: the speculative-exception decoy only ever perturbs speculative-eligible seals. Sweeping any other
    # (architectural) seal measures an architectural MTE effect the violation never exercised — the defect
    # that produced a false "genuine" on 160128 and 021502. Restrict the swept gate to the offsets of
    # seals the sealer marks speculative (decoy-eligible); the resolved object is memoized so its offsets
    # match these relocs' coordinate system.
    rsv = ex._resolve(inp)
    decoy_offsets = set()
    for r in rsv._entries:
        if r.speculative and r.alts:
            offs = rsv._offsets.get(id(r.sealing))
            if offs:
                decoy_offsets.update(offs)

    def dput(val, d):
        return (val & ~(0xF << 10)) | ((d & 0xF) << 10)

    def ei(plan):
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=g.mte_tags,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    # index ADDG relocs by offset
    addg = [(k, r) for k, r in enumerate(relocs) if (r.value & 0xFFC00000) == ADDG_OP]
    by_off = {r.offset: k for k, r in addg}
    # a seal = before ADDG (delta=correct!=0) + after ADDG at before_off+8, same Xd, delta 0 (base-preserved).
    # else persist/writeback single.
    seals = []  # (kbefore, kafter_or_None, correct, reg)
    used = set()
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

    base_res = res(ex.trace_test_case([g], reps)[0][0], tset)
    print(f"{os.path.basename(vd)} set={tset} reps={reps} baseline(set)={base_res:.2f}; {len(seals)} seals")

    # knockout each seal CONFINED (before+1, after-1) and measure the target set
    def plan_set(kb, ka, c, db):
        p = list(relocs)
        p[kb] = Reloc(relocs[kb].offset, dput(relocs[kb].value, db), relocs[kb].rtype)
        if ka is not None:
            p[ka] = Reloc(relocs[ka].offset, dput(relocs[ka].value, (c - db) % 16), relocs[ka].rtype)
        return p

    decoyed_seals = [(kb, ka, c, reg) for (kb, ka, c, reg) in seals if relocs[kb].offset in decoy_offsets]
    if not decoyed_seals:
        print(f"REFUSE: no decoyed (speculative) seal among the seals; decoy offsets="
              f"{sorted(hex(o) for o in decoy_offsets)}. Sweeping an architectural seal would measure an "
              f"architectural MTE effect, not a speculative leak. Aborting.")
        return
    knock = []
    for (kb, ka, c, reg) in decoyed_seals:
        r = res(ex.trace_test_case([ei(plan_set(kb, ka, c, (c + 1) % 16))], reps)[0][0], tset)
        knock.append((abs(base_res - r), kb, ka, c, reg))
    knock.sort(reverse=True)
    print("top gates by |effect| on set (confined knockout):")
    for d, kb, ka, c, reg in knock[:4]:
        print(f"  off=0x{relocs[kb].offset:x} x{reg} paired={ka is not None} correct={c} |d|={d:+.2f}")

    _, KB, KA, C, REG = knock[0]
    print(f"SWEEP seal off=0x{relocs[KB].offset:x} x{REG} paired={KA is not None} (re-converged): "
          f"set {tset} residency by pointer delta 0..15")
    sweep = {d: res(ex.trace_test_case([ei(plan_set(KB, KA, C, d))], reps)[0][0], tset) for d in range(16)}
    print("   " + " ".join(f"{d}:{sweep[d]:.2f}" for d in range(16)))
    print(f"   genuine(match) delta={C}: {sweep[C]:.2f} ; max other: {max(v for d,v in sweep.items() if d!=C):.2f}")


if __name__ == "__main__":
    main()
