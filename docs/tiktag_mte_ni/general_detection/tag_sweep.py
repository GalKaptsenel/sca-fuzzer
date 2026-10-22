#!/usr/bin/env python3
"""Genuineness control (verify-mte-tag-leak tag-tracking): take the fuzzer's genuine sealed variant of
one input, and patch ONLY the speculative guess-load's retag ADDG uimm4 to each delta 0..15, measuring
the probe (set 32) residency per delta. A real tag-CHECK leak keeps the probe present only when the
retagged tag MATCHES the granule (and for the TCMA-unchecked tags 0/15 whose check is skipped), and
absent for every checked mismatch. Noise / a non-tag artifact would fire uniformly across deltas.

Usage: tag_sweep.py -s base.json -c <hw.yml> -t <template.asm> [--retag-off 0x90] [--reps 3000]
"""
import argparse
import os
import sys

ADDG_X5_BASE = 0x918000A5   # ADDG x5, x5, #0, #0 ; uimm4 in bits [13:10]


def residency32(htrace):
    raw = htrace.raw
    if not raw:
        return 0.0
    return sum((int(r) >> 32) & 1 for r in raw) / len(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--spec", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-t", "--template", required=True)
    ap.add_argument("--retag-off", default="0x90")
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--repo", default="/home/gal_k_1_1998/revizor/sca-fuzzer")
    a = ap.parse_args()
    off = int(a.retag_off, 16)

    repo = os.path.abspath(a.repo)
    sys.path.insert(0, repo)
    spec, config, template = map(os.path.abspath, (a.spec, a.config, a.template))

    from src.config import CONF
    CONF.load(config)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import ExecutorInput

    wd = "/tmp/tag_sweep_wd"
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)

    fuzzer = factory.get_fuzzer(spec, wd, template, "")
    fuzzer.initialize_modules()
    tc = fuzzer.generator.create_test_case_from_template(template)
    ex = fuzzer.executor
    ex.load_test_case(tc)
    inputs = fuzzer.input_gen.generate(a.n)

    for ii, inp in enumerate(inputs):
        gen = ex.genuine_variant(inp)
        relocs = list(gen.code_reloc)
        # find the retag reloc at the speculative guess-load's before-slot
        idx = [k for k, r in enumerate(relocs) if r.offset == off]
        if not idx:
            addg = [(k, hex(r.offset), hex(r.value & 0xffffffff)) for k, r in enumerate(relocs)
                    if (r.value & 0xFFC003FF) == (ADDG_X5_BASE & 0xFFC003FF)]
            print(f"input {ii}: no reloc at off {hex(off)}; addg-x5 relocs: {addg}")
            continue
        k = idx[0]
        Reloc = type(relocs[k])
        print(f"\ninput {ii} seed={getattr(inp,'seed','?')} reps={a.reps} retag_off={hex(off)}")
        res = {}
        for delta in range(16):
            r2 = list(relocs)
            r2[k] = Reloc(off, ADDG_X5_BASE | (delta << 10), relocs[k].rtype)
            ei = ExecutorInput(gen.input_, code_reloc=tuple(r2), mte_tags=gen.mte_tags,
                               pac_keys=gen.pac_keys, bpu_training=gen.bpu_training,
                               env_plan=gen.env_plan)
            ht, _ = ex.trace_test_case([ei], a.reps)
            res[delta] = residency32(ht[0])
        print("  probe32 by retag delta: " + " ".join(f"{d}:{res[d]:.2f}" for d in range(16)))


if __name__ == "__main__":
    main()
