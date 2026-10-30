#!/usr/bin/env python3
"""GRANULE-TAG TRACKING CONTROL — the decisive genuineness proof for an MTE/TikTag leak.

A pointer-tag sweep alone shows the leak depends on the POINTER tag. This proves it is the tag CHECK
(pointer-vs-memory) by changing the MEMORY (granule) tag of the guess load and confirming the detected
match tag MOVES with it. If the peak tracks the granule tag (peak == (d0 + (T_mem - T_default)) mod 16),
the signal is the architectural tag comparison, not a fixed pointer-tag artifact.

GST (GlobalSealTable) model: the code is byte-identical across variants; a variant = its GST image.
A decoyed gate is an eligible MteSealing site (speculative, with alternative tags); sweeping it = setting
its APPLIED-tag lane while the re-converge lane stays genuine. The guess load is the site's own access.

Steps:
  1. Among the decoyed (speculative) MTE sites, knock each to (genuine+1) and pick the one gating the
     diverging set -> suspect site, pointer register Xd, genuine tag d0 (= the uniform default mem tag).
  2. Locate that site's guess-load granule: CE-run (COND|BPAS, max nesting) the genuine code and find the
     speculative access whose base register == Xd; granule index = (eff_addr - base)/MTE_GRANULE.
  3. For each memory tag T_mem (checked tags only): patch ONLY that granule to T_mem (others stay
     default, so architectural sealed accesses still match), sweep the pointer's applied tag 0..15, report
     the peak. GENUINE iff the peak tracks: peak(T_mem) == (d0 + T_mem - T_default) mod 16.

Usage: tag_track.py <cfg> <violation-dir> <suspect_idx> <diverging_set> [--reps N] [--gate-reg xN]
"""
import dataclasses
import glob
import os
import sys

REPO = "/home/gal_k_1_1998/revizor/sca-fuzzer"


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd, sus, dset = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), \
        int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2000
    gate_reg = sys.argv[sys.argv.index("--gate-reg") + 1] if "--gate-reg" in sys.argv else None
    sys.path.insert(0, REPO)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize
    from src.aarch64.aarch64_contract_executor import ExecutionClause
    from src.aarch64.aarch64_mte import MTE_GRANULE, MTE_INITIAL_DEFAULT_TAG

    wd = "/tmp/tag_track_wd"
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(REPO, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm")
    ex = fz.executor
    ex.load_test_case(tc)
    inp = deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()).input_
    g = ex.genuine_variant(inp)
    rsv = ex._resolve(inp)
    if ex._sandbox_base is None:
        ex._sandbox_base, _ = ex.read_base_addresses()
    ntags = len(g.mte_tags) if g.mte_tags else 0
    print(f"{os.path.basename(vd)} suspect#{sus} divset={dset} reps={reps} ntags={ntags}")

    def build_image(sus_sealing, applied):
        image = rsv._gst.blank_image()
        for r in rsv._entries:
            r.sealing.store_into(image, applied if r.sealing is sus_sealing else r.value, r.value)
        return image.to_section().content

    def ei(sus_sealing, applied, mte):
        return dataclasses.replace(g, gst=build_image(sus_sealing, applied), mte_tags=tuple(mte))

    # --- Step 1: the decoyed (speculative) MTE gates; knock each to (genuine+1) to find the one gating
    # the diverging set. Architectural sites (not speculative / no alts) are never swept. ---
    decoyed = [r for r in rsv._entries
               if r.sealing.__class__.__name__ == "MteSealing" and r.speculative and r.alts]
    if not decoyed:
        print("no decoyed (speculative) MTE gate found; abort")
        return
    print(f"decoyed (speculative) gate regs: {sorted(r.sealing.value_reg for r in decoyed)}")
    base_res = res(ex.trace_test_case([g], reps)[0][0], dset)
    if gate_reg is not None:
        sel = [r for r in decoyed if r.sealing.value_reg == gate_reg]
        if not sel:
            print(f"gate {gate_reg} is not a decoyed MTE site; abort")
            return
        R = sel[0]
    else:
        knock = []
        for r in decoyed:
            rr = res(ex.trace_test_case([ei(r.sealing, (r.value + 1) & 0xF, g.mte_tags)], reps)[0][0], dset)
            knock.append((base_res - rr, r, rr))
        knock.sort(key=lambda t: t[0], reverse=True)
        _, R, _ = knock[0]
    vreg = int(R.sealing.value_reg[1:])        # Xd = the sealed GUESS pointer register the checked load uses
    D0 = R.value                               # genuine tag = the uniform default memory tag
    print(f"gating pointer site Xd=x{vreg} genuine_tag d0={D0} (baseline divset res={base_res:.2f})")

    # --- Step 2: the guess-load granule = the speculative access whose base register == Xd ---
    code = rsv.object_code
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
        if accs and rn == vreg:
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
        sweeps[T] = {d: res(ex.trace_test_case([ei(R.sealing, d, mte)], reps)[0][0], dset) for d in range(16)}
    print("mem_tag | expected_peak | pointer-tag sweep (divset residency) | peak | d0_res | TRACKS?")
    d0_default = sweeps[default][D0]
    track_ok = True
    patch_effective = False
    for T in mem_tags:
        sw = sweeps[T]
        checked = [d for d in sw if d not in (0, 15)]   # TCMA-unchecked tags 0/15 always resident
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
        print("VERDICT: GENUINE — the detected match tag TRACKS the memory tag (pointer-vs-memory check).")
    else:
        print("VERDICT: SUSPECT — patch moved the granule (d0 collapsed) but the peak did NOT track the "
              "new memory tag; not a clean pointer-vs-memory match.")


if __name__ == "__main__":
    main()
