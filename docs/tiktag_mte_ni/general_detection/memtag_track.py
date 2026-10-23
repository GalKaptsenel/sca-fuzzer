#!/usr/bin/env python3
"""Clean memory-tag control: (A) empirically find the relevant SPECULATIVE granule (retag each granule
touched speculatively, one at a time, to a mismatch and see which one moves the diverging set), then
(B) retag ONLY that granule's memory tag and sweep it 0..15 with the pointer left at its genuine value,
measuring the diverging set. If the set responds at exactly the memtag that equals the genuine pointer
tag (and differs elsewhere), the leak is the pointer-vs-memory tag comparison. No gate guessing; nothing
claimed until the numbers are in.

Usage: memtag_track.py <cfg> <violation-dir> <suspect_idx> <diverging_set> [--reps N]
"""
import glob
import os
import sys


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd, sus, dset = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), \
        int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2000
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
    wd = "/tmp/memtag_track_wd"
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm")
    ex = fz.executor
    ex.load_test_case(tc)
    inp = deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()).input_
    g = ex.genuine_variant(inp)
    if ex._sandbox_base is None:
        ex._sandbox_base, _ = ex.read_base_addresses()
    default = MTE_INITIAL_DEFAULT_TAG
    ntags = len(g.mte_tags)

    def ei(mte):
        return ExecutorInput(g.input_, code_reloc=g.code_reloc, mte_tags=mte,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    # enumerate SPECULATIVE granules from the CE (nesting>0), tag byte masked off
    sealed = ex._sealer.seal(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
    code = apply_relocations(sealed.resolve(inp).object_code, sealed.resolve(inp).genuine())
    cer = ex._contract_executor.run(ex._make_ce_execution(
        code, inp, ex._sandbox_base, CONF.model_max_nesting, CONF.model_max_spec_window,
        ExecutionClause.COND | ExecutionClause.BPAS, mte_tags=ex._mte_tags_for(inp),
        pac_keys=ex._input_pac_keys(inp), pac_profile=ex._pac_profile_value))
    MASK = (1 << 56) - 1
    spec_gr = []
    for ite in cer:
        if int(ite.metadata.speculation_nesting) == 0:
            continue
        base = ite.cpu.gpr[29]
        for ma in ite.metadata.accesses():
            gi = (((ma.effective_address & MASK) & ~(MTE_GRANULE - 1)) - ((base & MASK) & ~(MTE_GRANULE - 1))) // MTE_GRANULE
            if 0 <= gi < ntags and gi not in spec_gr:
                spec_gr.append(gi)
    base_res = res(ex.trace_test_case([g], reps)[0][0], dset)
    print(f"{os.path.basename(vd)} suspect#{sus} divset={dset} reps={reps} default_memtag={default}")
    print(f"genuine baseline divset res={base_res:.2f}; speculative granules={sorted(spec_gr)}")

    # PHASE A: which speculative granule, retagged to a mismatch, moves the diverging set?
    mism = (default + 7) % 16
    print("PHASE A (retag one spec granule -> mismatch, |delta| on divset):")
    effects = []
    for gi in sorted(spec_gr):
        mte = list(g.mte_tags)
        mte[gi] = mism
        r = res(ex.trace_test_case([ei(mte)], reps)[0][0], dset)
        effects.append((abs(base_res - r), gi, r))
        print(f"  granule {gi:3d}: memtag {default}->{mism}  divset {base_res:.2f}->{r:.2f}  |d|={abs(base_res-r):.2f}")
    effects.sort(reverse=True)
    if not effects or effects[0][0] < 0.10:
        print("PHASE A: no speculative granule moves the diverging set (>=0.10) -> leak is NOT gated by a "
              "speculative memory-tag check at this set (per this control).")
        return
    gstar = effects[0][1]
    print(f"relevant speculative granule = {gstar}")

    # PHASE B: retag ONLY that granule, sweep its memtag 0..15 (pointer left genuine)
    print(f"PHASE B (retag granule {gstar} memtag 0..15, pointer genuine; divset residency):")
    sweep = {}
    for T in range(16):
        mte = list(g.mte_tags)
        mte[gstar] = T
        sweep[T] = res(ex.trace_test_case([ei(mte)], reps)[0][0], dset)
    print("   " + " ".join(f"{T}:{sweep[T]:.2f}" for T in range(16)))
    # the memtag matching the genuine pointer tag should be the outlier (== baseline level)
    at_default = sweep[default]
    others = [sweep[T] for T in range(16) if T != default and T not in (0, 15)]
    print(f"   memtag=default({default}) -> {at_default:.2f} ; mean(other checked) -> "
          f"{sum(others)/len(others):.2f} ; min/max other -> {min(others):.2f}/{max(others):.2f}")


if __name__ == "__main__":
    main()
