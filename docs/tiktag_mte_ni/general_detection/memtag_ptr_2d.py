#!/usr/bin/env python3
"""Full 2D tag-tracking control: for the relevant speculative granule, set its MEMORY tag to several
values T and, for each, sweep the POINTER tag (the ADDG that tags the register the granule's speculative
load uses) over 0..15, measuring the diverging set. A genuine pointer-vs-memory tag check shows the
leaking set peak at pointer tag == T, and the peak MOVES as T changes (tracking).

Phase A finds the relevant granule empirically (retag each spec granule, pick the one that moves the set).
Then it finds the speculative load that reads that granule and the DECOYED ADDG that tags that load's base
register (never an architectural gate). Nothing is claimed; it prints the grid.

Usage: memtag_ptr_2d.py <cfg> <violation-dir> <suspect_idx> <diverging_set> [--reps N]
"""
import glob
import os
import sys

ADDG_OP = 0x91800000


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd, sus, dset = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), \
        int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1500
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    from src.aarch64.aarch64_relocations import apply_relocations
    from src.aarch64.aarch64_contract_executor import ExecutionClause
    from src.aarch64.aarch64_trace import _cache_sets
    from src.aarch64.aarch64_mte import MTE_GRANULE, MTE_INITIAL_DEFAULT_TAG
    wd = "/tmp/memtag_ptr_2d_wd"
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm")
    ex = fz.executor
    ex.load_test_case(tc)
    inp = deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()).input_
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0])
    relocs = list(g.code_reloc)
    if ex._sandbox_base is None:
        ex._sandbox_base, _ = ex.read_base_addresses()
    default = MTE_INITIAL_DEFAULT_TAG
    ntags = len(g.mte_tags)
    MASK = (1 << 56) - 1

    def ei(plan, mte):
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=mte,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    # CE: speculative accesses with (granule, pc_offset, base_reg)
    sealed = ex._sealer.seal(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
    rsv = sealed.resolve(inp)
    code = apply_relocations(rsv.object_code, rsv.genuine())
    cer = ex._contract_executor.run(ex._make_ce_execution(
        code, inp, ex._sandbox_base, CONF.model_max_nesting, CONF.model_max_spec_window,
        ExecutionClause.COND | ExecutionClause.BPAS, mte_tags=ex._mte_tags_for(inp),
        pac_keys=ex._input_pac_keys(inp), pac_profile=ex._pac_profile_value))
    cb = cer[0].cpu.pc
    all_gr = "--all-granules" in sys.argv   # include architecturally-touched granules in Phase A
    spec_acc = []  # (granule, off, base_reg)
    phase_a_gr = set()
    for ite in cer:
        spec = int(ite.metadata.speculation_nesting) != 0
        off = ite.cpu.pc - cb
        if not (0 <= off < len(code)):
            continue
        rn = (int.from_bytes(code[off:off + 4], "little") >> 5) & 0x1f
        base = ite.cpu.gpr[29]
        for ma in ite.metadata.accesses():
            gi = (((ma.effective_address & MASK) & ~(MTE_GRANULE - 1)) - ((base & MASK) & ~(MTE_GRANULE - 1))) // MTE_GRANULE
            if 0 <= gi < ntags:
                if spec:
                    spec_acc.append((gi, off, rn))
                if spec or all_gr:
                    phase_a_gr.add(gi)
    spec_gr = sorted(phase_a_gr)

    # Phase A: relevant granule = the one whose mismatch moves the set most
    base_res = res(ex.trace_test_case([g], reps)[0][0], dset)
    mism = (default + 7) % 16
    eff = []
    for gi in spec_gr:
        mte = list(g.mte_tags); mte[gi] = mism
        r = res(ex.trace_test_case([ei(relocs, mte)], reps)[0][0], dset)
        eff.append((abs(base_res - r), gi))
    eff.sort(reverse=True)
    print(f"{os.path.basename(vd)} set={dset} reps={reps} baseline={base_res:.2f} "
          f"relevant_granule_effects(top3)={eff[:3]}")
    if not eff or eff[0][0] < 0.10:
        print("no granule isolates the set via memory tag -> 2D tracking not applicable here.")
        return
    gstar = eff[0][1]

    # find the speculative load that reads gstar, and the DECOYED ADDG tagging its base register
    en = sorted(glob.glob(f"{vd}/enacted_test*.bin"))
    bb = open(en[0], "rb").read()
    diff = set()
    for d in en[1:]:
        ob = open(d, "rb").read()
        for i in range(0, min(len(bb), len(ob)), 4):
            if bb[i:i + 4] != ob[i:i + 4]:
                diff.add(i)
    load_off = min((off for (gi, off, rn) in spec_acc if gi == gstar), default=None)
    load_rn = next((rn for (gi, off, rn) in spec_acc if gi == gstar), None)
    force = int(sys.argv[sys.argv.index("--gate-off") + 1], 16) if "--gate-off" in sys.argv else None
    gate = None
    if force is not None:
        for k, r in enumerate(relocs):
            if (r.value & 0xFFC00000) == ADDG_OP and r.offset == force:
                gate = k
        if gate is None:
            print(f"forced gate 0x{force:x} is not an ADDG reloc; abort")
            return
    else:
        for k, r in enumerate(relocs):
            if (r.value & 0xFFC00000) == ADDG_OP and r.offset in diff and (r.value & 0x1f) == load_rn \
                    and r.offset < load_off:
                if gate is None or r.offset > relocs[gate].offset:
                    gate = k
    if gate is None:
        print(f"granule {gstar}: no decoyed ADDG tags its load's base x{load_rn} before off 0x{load_off:x}; "
              "cannot sweep the pointer for this granule.")
        return
    R = relocs[gate]
    D0 = (R.value >> 10) & 0xF
    print(f"relevant granule={gstar}; its spec load base=x{load_rn} at off 0x{load_off:x}; "
          f"pointer gate=decoyed ADDG off 0x{R.offset:x} (genuine delta {D0})")

    # 2D: memtag T x pointer-delta d
    print("2D grid rows=memtag T, cols=pointer delta 0..15 (divset residency); expected peak d=(D0+T-def)%16")
    for T in (3, 6, 9, 12):
        mte = list(g.mte_tags); mte[gstar] = T
        row = {}
        for d in range(16):
            plan = list(relocs)
            plan[gate] = Reloc(R.offset, (R.value & ~(0xF << 10)) | (d << 10), R.rtype)
            row[d] = res(ex.trace_test_case([ei(plan, mte)], reps)[0][0], dset)
        checked = [d for d in range(16) if d not in (0, 15)]
        peak = max(checked, key=lambda d: row[d])
        exp = (D0 + (T - default)) % 16
        print(f"  T={T:2d} exp_peak={exp:2d} peak={peak:2d}  " + " ".join(f"{d}:{row[d]:.2f}" for d in range(16)))


if __name__ == "__main__":
    main()
