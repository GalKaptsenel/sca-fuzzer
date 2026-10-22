#!/usr/bin/env python3
"""GRANULE-TAG TRACKING CONTROL — the decisive genuineness proof for an MTE/TikTag leak.

A pointer-tag sweep alone shows the leak depends on the POINTER tag. This proves it is the tag CHECK
(pointer-vs-memory) by changing the MEMORY (granule) tag of the guess load and confirming the detected
match delta MOVES with it. If the peak tracks the granule tag (peak_delta = d0 + (T_mem - T_default)
mod 16), the signal is genuinely the architectural tag comparison, not a fixed pointer-tag artifact.

Steps:
  1. Locate the guess load's granule: CE-run (COND|BPAS, max nesting) the genuine code and find the
     speculative access that hits the diverging set; granule index = (eff_addr - base)/16 (the executor
     seeds all granules with one uniform tag; index runs over the main|faulty span).
  2. Find the pointer retag ADDG that gates the diverging set (per-reloc knockout) -> reloc k, genuine
     delta d0 (the pointer delta matching the default memory tag).
  3. For each memory tag T_mem in a set of CHECKED tags: patch ONLY that granule to T_mem (others stay
     default, so architectural sealed accesses still match), sweep the pointer delta 0..15, report the
     peak. GENUINE iff the peak tracks: peak(T_mem) == (d0 + T_mem - T_default) mod 16.

Usage: tag_track.py <cfg> <violation-dir> <suspect_idx> <diverging_set> [--reps N]
"""
import glob
import os
import sys

REPO = "/home/gal_k_1_1998/revizor/sca-fuzzer"
ADDG_OP = 0x91800000


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd, sus, dset = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), \
        int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2000
    sys.path.insert(0, REPO)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    from src.aarch64.aarch64_relocations import apply_relocations
    from src.aarch64.aarch64_contract_executor import ExecutionClause
    from src.aarch64.aarch64_trace import _cache_sets
    from src.aarch64.aarch64_mte import MTE_GRANULE, MTE_INITIAL_DEFAULT_TAG

    wd = "/tmp/tag_track_wd"
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(REPO, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm")
    ex = fz.executor
    ex.load_test_case(tc)
    blob = open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()
    inp = deserialize(blob).input_
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0])
    relocs = list(g.code_reloc)
    if ex._sandbox_base is None:
        ex._sandbox_base, _ = ex.read_base_addresses()

    ntags = len(g.mte_tags) if g.mte_tags else 0
    print(f"{os.path.basename(vd)} suspect#{sus} divset={dset} reps={reps} ntags={ntags}")

    def ei(plan, mte):
        return ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=mte,
                             pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)

    # --- Step 1: find the pointer retag ADDG gating the diverging set (knockout at default tag) ---
    addg = [(k, r) for k, r in enumerate(relocs) if (r.value & 0xFFC00000) == ADDG_OP]
    base_res = res(ex.trace_test_case([g], reps)[0][0], dset)
    knock = []
    for k, r in addg:
        d0 = (r.value >> 10) & 0xF
        plan = list(relocs)
        plan[k] = Reloc(r.offset, (r.value & ~(0xF << 10)) | (((d0 + 1) & 0xF) << 10), r.rtype)
        rr = res(ex.trace_test_case([ei(plan, g.mte_tags)], reps)[0][0], dset)
        knock.append((base_res - rr, k, r, d0))
    knock.sort(reverse=True)
    _, K, R, D0 = knock[0]
    vreg = R.value & 0x1f  # ADDG Xd = the sealed GUESS pointer register the checked load uses as base
    print(f"gating pointer ADDG off=0x{R.offset:x} Xd=x{vreg} genuine_delta d0={D0} "
          f"(baseline divset res={base_res:.2f})")

    # --- Step 2: find the GUESS LOAD's granule = the speculative LOAD whose base reg == Xd (the
    # tag-CHECKED access), NOT the downstream probe that hits the diverging set. ---
    sealed = ex._sealer.seal(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
    rsv = sealed.resolve(inp)
    code = apply_relocations(rsv.object_code, rsv.genuine())
    cer = ex._contract_executor.run(ex._make_ce_execution(
        code, inp, ex._sandbox_base, CONF.model_max_nesting, CONF.model_max_spec_window,
        ExecutionClause.COND | ExecutionClause.BPAS, mte_tags=ex._mte_tags_for(inp),
        pac_keys=ex._input_pac_keys(inp), pac_profile=ex._pac_profile_value))
    cb = cer[0].cpu.pc
    MASK = (1 << 56) - 1
    gidx = None
    for ite in cer:
        if int(ite.metadata.speculation_nesting) == 0:
            continue
        off = ite.cpu.pc - cb
        if not (0 <= off < len(code)):
            continue
        rn = (int.from_bytes(code[off:off + 4], "little") >> 5) & 0x1f
        accs = list(ite.metadata.accesses())
        if accs and rn == vreg and off > R.offset:   # the first use of the retagged pointer
            base = ite.cpu.gpr[29]
            ga = (accs[0].effective_address & MASK) & ~(MTE_GRANULE - 1)
            gb = (base & MASK) & ~(MTE_GRANULE - 1)
            gidx = (ga - gb) // MTE_GRANULE
            break
    if gidx is None or not (0 <= gidx < ntags):
        print(f"could not locate the guess load's granule (base x{vreg}); gidx={gidx}; abort")
        return
    print(f"guess-load granule index={gidx} (default mem tag={MTE_INITIAL_DEFAULT_TAG})")

    # --- Step 3: move the MEMORY tag, re-sweep the pointer, check the peak TRACKS and the patch is REAL ---
    default = MTE_INITIAL_DEFAULT_TAG
    mem_tags = [default, 3, 9, 12]  # checked tags (avoid TCMA-unchecked 0/15)
    sweeps = {}
    for T in mem_tags:
        mte = list(g.mte_tags)
        mte[gidx] = T
        sw = {}
        for d in range(16):
            plan = list(relocs)
            plan[K] = Reloc(R.offset, (R.value & ~(0xF << 10)) | (d << 10), R.rtype)
            sw[d] = res(ex.trace_test_case([ei(plan, mte)], reps)[0][0], dset)
        sweeps[T] = sw
    print("mem_tag | expected_peak | pointer-delta sweep (divset residency) | peak | d0_res | TRACKS?")
    d0_default = sweeps[default][D0]
    track_ok = True
    patch_effective = False
    for T in mem_tags:
        sw = sweeps[T]
        # the tracked match is a CHECKED tag; TCMA-unchecked deltas 0/15 are always resident, exclude them
        checked = [d for d in sw if d not in (0, 15)]
        peak = max(checked, key=lambda d: sw[d])
        expected = (D0 + (T - default)) % 16
        tracks = (peak == expected)
        track_ok = track_ok and tracks
        if T != default and sw[D0] <= 0.5 * d0_default:   # old match collapsed -> patch really moved the mem tag
            patch_effective = True
        cells = " ".join(f"{d}:{sw[d]:.2f}" for d in range(16))
        print(f"  T={T:2d} | exp {expected:2d} | {cells} | peak {peak} | d0={sw[D0]:.2f} | {'YES' if tracks else 'NO'}")
    if not patch_effective:
        print("VALIDITY: FAIL — changing the granule tag did NOT move delta d0; the mem-tag patch did not "
              "reach the CHECKED granule (control invalid, NOT evidence either way).")
    elif track_ok:
        print("VERDICT: GENUINE — the detected match delta TRACKS the memory tag (pointer-vs-memory check).")
    else:
        print("VERDICT: SUSPECT — patch moved the granule (d0 collapsed) but the peak did NOT track the "
              "new memory tag; not a clean pointer-vs-memory match.")


if __name__ == "__main__":
    main()
