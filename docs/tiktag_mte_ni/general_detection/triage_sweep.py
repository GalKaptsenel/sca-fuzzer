#!/usr/bin/env python3
"""Decisive 16-tag sweep for a saved speculative-exception violation, on the REAL violation input.

Starts from the genuine sealed variant, finds the sealed site feeding the reported diverging set
(phase 1 knockout), then sweeps THAT site's applied tag over all 16 values (phase 2) and reports the
diverging-set residency per tag.

GST (GlobalSealTable) model: the test-case code is byte-identical across every variant; a variant is
selected purely by its GST image. Each MTE sealed site has an APPLIED-tag lane (the tag BFI'd into the
pointer before the access) and, when the access preserves its base, a re-converge lane that always
carries the genuine cell tag -- so downstream runs as if the access used the correct tag and only the
sealed access sees the perturbed tag. Sweeping a site = rewriting its applied lane; the re-converge lane
stays genuine automatically (no net-tag bookkeeping needed, unlike the old relocation sweep).

Priming: the batch's other inputs run as a prefix before the suspect (cross-input priming), suspect read
at the last slot. `--no-prime` disables it.

GENUINE tag-check = the matching tag (and TCMA 0/15) stand out, every checked mismatch suppressed.
NOISE/footprint = fires ~uniformly or the genuine tag does not stand out.

Usage: triage_sweep.py <cfg> <violation-dir> <suspect_idx> <diverging_set> [--reps N] [--no-prime]
"""
import dataclasses
import glob
import os
import re as _re
import sys


def res(ht, s):
    raw = ht.raw
    return sum((int(r) >> s) & 1 for r in raw) / len(raw) if raw else 0.0


def main():
    cfg, vd, sus, dset = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1200
    noprime = "--no-prime" in sys.argv
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize
    wd = "/tmp/triage_sweep_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm"); ex = fz.executor; ex.load_test_case(tc)
    inp = deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()).input_
    g = ex.genuine_variant(inp)
    res_ = ex._resolve(inp)

    mte_sites = [r for r in res_._entries if r.sealing.__class__.__name__ == "MteSealing"]
    if not mte_sites:
        print("no MteSealing sites in this test case"); return

    def build_image(sus_sealing, applied):
        """GST image: genuine for every seal except `sus_sealing`, whose APPLIED lane is set to `applied`
        (its re-converge lane stays genuine, so only the sealed access sees the perturbed tag)."""
        image = res_._gst.blank_image()
        for r in res_._entries:
            r.sealing.store_into(image, applied if r.sealing is sus_sealing else r.value, r.value)
        return image.to_section().content

    def ei(image):
        return dataclasses.replace(g, gst=image)

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
          f"{len(mte_sites)} MTE sealed sites")

    base = res(ex.trace_test_case(prefix + [g], reps)[0][slot], dset)
    print(f"{vd.split('/')[-1]} suspect#{sus} divset={dset} reps={reps}  genuine baseline res={base:.2f}")

    # phase 1: knock each site to (genuine cell tag + 1), find which drops the diverging set most
    knock = []
    for r in mte_sites:
        gtag = r.value
        rr = res(ex.trace_test_case(prefix + [ei(build_image(r.sealing, (gtag + 1) & 0xF))], reps)[0][slot], dset)
        knock.append((base - rr, r, gtag, rr))
    knock.sort(key=lambda t: t[0], reverse=True)
    print("  top knockouts (drop, reg, genuine_tag, res):")
    for drop, r, gtag, rr in knock[:4]:
        print(f"    reg={r.sealing.value_reg} gtag={gtag} res={rr:.2f} drop={drop:+.2f}")

    # phase 2: full 16-tag sweep on the strongest site (applied tag absolute 0..15)
    _, r, gtag, _ = knock[0]
    sweep = {d: res(ex.trace_test_case(prefix + [ei(build_image(r.sealing, d))], reps)[0][slot], dset)
             for d in range(16)}
    print(f"  SWEEP site reg={r.sealing.value_reg} (genuine delta={gtag}): divset {dset} residency by delta 0..15:")
    print("   " + " ".join(f"{d}:{sweep[d]:.2f}" for d in range(16)))


if __name__ == "__main__":
    main()
