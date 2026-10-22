#!/usr/bin/env python3
"""Decisive 16-tag sweep for a saved NI violation, on the REAL violation input.

Loads the violation's generated.asm as the TC and the suspect input from its .reif, builds the genuine
sealed variant, then (phase 1) knocks out each sealed-load retag ADDG one at a time to find which load
feeds the reported diverging set, and (phase 2) sweeps THAT load's tag over all 16 deltas, reporting the
diverging-set residency per delta. GENUINE tag-check = the matching delta (and TCMA-unchecked 0/15) stand
out, every checked mismatch suppressed. NOISE/footprint = fires ~uniformly or decoy gains residency.

Usage: triage_sweep.py <cfg> <violation-dir> <suspect_idx> <diverging_set> [--reps N]
"""
import glob
import os
import sys

ADDG_OP = 0x91800000          # ADDG Xd,Xn,#0,#d : top opcode bits; uimm4 (delta) in [13:10]


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd, sus, dset = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1200
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    from src.aarch64.aarch64_executor import NIVariant
    wd = "/tmp/triage_sweep_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm"); ex = fz.executor; ex.load_test_case(tc)
    blob = open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()
    inp = deserialize(blob).input_
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0])
    relocs = list(g.code_reloc)

    def ei(plan):
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=g.mte_tags,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    # all ADDG retag relocs (candidate per-load before/after slots)
    addg = [(k, r) for k, r in enumerate(relocs) if (r.value & 0xFFC00000) == ADDG_OP]
    base = res(ex.trace_test_case([g], reps)[0][0], dset)
    print(f"{vd.split('/')[-1]} suspect#{sus} divset={dset} reps={reps}  genuine baseline res={base:.2f}")
    print(f"  {len(addg)} ADDG retag relocs")

    # phase 1: knock out each slot (delta -> genuine+1) and find which drops the diverging set
    knock = []
    for k, r in addg:
        d0 = (r.value >> 10) & 0xF
        plan = list(relocs); plan[k] = Reloc(r.offset, (r.value & ~(0xF << 10)) | (((d0 + 1) & 0xF) << 10), r.rtype)
        rr = res(ex.trace_test_case([ei(plan)], reps)[0][0], dset)
        knock.append((base - rr, k, r, d0, rr))
    knock.sort(reverse=True)
    print("  top knockouts (drop, off, genuine_delta, res):")
    for drop, k, r, d0, rr in knock[:4]:
        print(f"    off=0x{r.offset:x} dgen={d0} res={rr:.2f} drop={drop:+.2f}")

    # phase 2: full 16-delta sweep on the strongest-knockout slot
    drop, k, r, d0, _ = knock[0]
    sweep = {}
    for d in range(16):
        plan = list(relocs); plan[k] = Reloc(r.offset, (r.value & ~(0xF << 10)) | (d << 10), r.rtype)
        sweep[d] = res(ex.trace_test_case([ei(plan)], reps)[0][0], dset)
    print(f"  SWEEP leaking off=0x{r.offset:x} (genuine delta={d0}): divset {dset} residency by delta 0..15:")
    print("   " + " ".join(f"{d}:{sweep[d]:.2f}" for d in range(16)))


if __name__ == "__main__":
    main()
