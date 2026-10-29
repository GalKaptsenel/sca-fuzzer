#!/usr/bin/env python3
"""Decisive 16-tag sweep for a saved speculative-exception violation, on the REAL violation input.

Starts from the genuine sealed variant (which already carries the sandbox-clamp relocations), finds the
sealed site feeding the reported diverging set (phase 1 knockout), then sweeps THAT site's tag over all
16 deltas (phase 2) and reports the diverging-set residency per delta.

RE-MERGE (how the sealer does it, post the CE tag-preservation fix): a sealed site is a before-ADDG that
retags the base to the ACCESS's tag, then the access, then an after-ADDG that re-converges the base back
to the CORRECT cell tag -- so downstream runs AS IF the access used the correct tag; only the access
sees the perturbed tag. So when we sweep a site's before-ADDG to delta d we MUST also set its paired
after-ADDG so the net tag stays = correct (after = net_genuine - d). Perturbing only the before-ADDG
corrupts the base tag for every downstream instruction and confounds the measurement.

Priming: the batch's other inputs run as a prefix before the suspect (cross-input priming), suspect read
at the last slot. `--no-prime` disables it.

GENUINE tag-check = the matching delta (and TCMA 0/15) stand out, every checked mismatch suppressed.
NOISE/footprint = fires ~uniformly or the genuine tag does not stand out.

Usage: triage_sweep.py <cfg> <violation-dir> <suspect_idx> <diverging_set> [--reps N] [--no-prime]
"""
import glob
import os
import re as _re
import sys

ADDG_OP = 0x91800000          # ADDG Xd,Xn,#0,#d : opcode bits; uimm4 (tag delta) in [13:10]


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def _uimm(v):
    return (v >> 10) & 0xF


def _with_uimm(v, u):
    return (v & ~(0xF << 10)) | ((u & 0xF) << 10)


def main():
    cfg, vd, sus, dset = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1200
    noprime = "--no-prime" in sys.argv
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    wd = "/tmp/triage_sweep_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm"); ex = fz.executor; ex.load_test_case(tc)
    inp = deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()).input_
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0])
    relocs = list(g.code_reloc)
    by_off = {r.offset: r for r in relocs}

    # before->after pairing + the set of sealed-site before offsets, from the resolver's slot offsets.
    res_ = ex._resolve(inp)
    pair = {}          # before_off -> after_off  (2-slot sites: base preserved -> has a re-merge ADDG)
    before_offs = []   # all sealed-site before offsets
    for r in res_._entries:
        offs = res_._offsets.get(id(r.sealing))
        if not offs:
            continue
        before_offs.append(offs[0])
        if len(offs) == 2:
            pair[offs[0]] = offs[1]
    before_offs = [o for o in before_offs if o in by_off and (by_off[o].value & 0xFFC00000) == ADDG_OP]

    def plan_site(before_off, d):
        """relocs with site `before_off` set to tag delta d, its paired after-ADDG re-merged so the net
        tag stays genuine (downstream sees the correct tag). Everything else genuine."""
        out = list(relocs)
        idx = {r.offset: i for i, r in enumerate(out)}
        b = by_off[before_off]; bu = _uimm(b.value)
        out[idx[before_off]] = Reloc(before_off, _with_uimm(b.value, d), b.rtype)
        aoff = pair.get(before_off)
        if aoff is not None and aoff in by_off:
            a = by_off[aoff]; net = (bu + _uimm(a.value)) & 0xF
            out[idx[aoff]] = Reloc(aoff, _with_uimm(a.value, (net - d) & 0xF), a.rtype)
        return out

    def ei(plan):
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=g.mte_tags,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    prefix = []
    if not noprime:
        others = sorted({int(m.group(1)) for p in glob.glob(f"{vd}/input_0*.*")
                         for m in [_re.search(r'input_(\d+)\.', os.path.basename(p))] if m})
        for k in others:
            if k == sus:
                continue
            kb = open(glob.glob(f"{vd}/input_{k:04d}.*")[0], "rb").read()
            prefix.append(ex.genuine_variant(deserialize(kb).input_))
    slot = -1 if prefix else 0
    print(f"  priming prefix: {len(prefix)} prior inputs; measuring suspect at slot {slot}; "
          f"{len(before_offs)} sealed sites ({len(pair)} with a re-merge ADDG)")

    base = res(ex.trace_test_case(prefix + [g], reps)[0][slot], dset)
    print(f"{vd.split('/')[-1]} suspect#{sus} divset={dset} reps={reps}  genuine baseline res={base:.2f}")

    # phase 1: knock each site to (genuine+1), RE-MERGED, find which drops the diverging set
    knock = []
    for bo in before_offs:
        d0 = _uimm(by_off[bo].value)
        rr = res(ex.trace_test_case(prefix + [ei(plan_site(bo, (d0 + 1) & 0xF))], reps)[0][slot], dset)
        knock.append((base - rr, bo, d0, rr))
    knock.sort(reverse=True)
    print("  top knockouts (drop, off, genuine_delta, res):")
    for drop, bo, d0, rr in knock[:4]:
        print(f"    off=0x{bo:x} dgen={d0} res={rr:.2f} drop={drop:+.2f} {'(has re-merge)' if bo in pair else '(no re-merge)'}")

    # phase 2: full 16-delta sweep on the strongest site, RE-MERGED
    _, bo, d0, _ = knock[0]
    sweep = {d: res(ex.trace_test_case(prefix + [ei(plan_site(bo, d))], reps)[0][slot], dset) for d in range(16)}
    print(f"  SWEEP site off=0x{bo:x} (genuine delta={d0}, {'re-merged' if bo in pair else 'no re-merge slot'}): "
          f"divset {dset} residency by delta 0..15:")
    print("   " + " ".join(f"{d}:{sweep[d]:.2f}" for d in range(16)))


if __name__ == "__main__":
    main()
