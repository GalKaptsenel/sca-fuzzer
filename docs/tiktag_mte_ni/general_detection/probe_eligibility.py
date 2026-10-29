#!/usr/bin/env python3
"""Device-free CE-eligibility probe for the MTE speculative-exception pipeline.

Builds the real speculative-exception pipeline (get_fuzzer speculative-exception + initialize_modules), parses an
ASM template, seals it, and for several generated inputs prints, per MTE sealing site: the resolved tag
delta (value), the decoy alternatives (alts), and whether the slot is speculative (decoy-eligible). Also
prints whether the input has any decoy and the genuine-vs-decoy relocation difference. The contract
executor runs locally, so this needs no device; executor._sandbox_base is preset to the live base.

Usage:
  probe_eligibility.py -s <base.json> -c <config.yml> -t <template.asm> [--base 0xHEX] [--n 4]
"""
import argparse
import os
import sys


def disasm_reloc(plan):
    """Summarize a relocation plan as a list of (offset, encoded-word-hex). Relocation is a NamedTuple
    (offset, value, rtype); value is the instruction word baked at offset."""
    return [(rel.offset, f"0x{rel.value & 0xffffffff:08x}") for rel in plan]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--spec", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-t", "--template", required=True)
    ap.add_argument("--base", default="0xffffff8873821000")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--repo", default="/home/gal_k_1_1998/revizor/sca-fuzzer")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)
    spec = os.path.abspath(args.spec)
    config = os.path.abspath(args.config)
    template = os.path.abspath(args.template)

    from src.config import CONF
    CONF.load(config)
    from src import factory

    workdir = os.path.join("/tmp", "probe_elig_wd")
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    fuzzer = factory.get_fuzzer(spec, workdir, template, "")
    fuzzer.initialize_modules()

    test_case = fuzzer.generator.create_test_case_from_template(template)
    executor = fuzzer.executor
    executor.load_test_case(test_case)
    executor._sandbox_base = int(args.base, 16)

    print(f"TEMPLATE {template}")
    print(f"primitives = {getattr(executor, '_primitives', '?')}")
    inputs = fuzzer.input_gen.generate(args.n)

    any_decoy = False
    for i, inp in enumerate(inputs):
        try:
            has = executor.has_decoy(inp)
            resolved = executor._resolve(inp)
        except Exception as e:
            print(f"\ninput {i}: resolve FAILED: {type(e).__name__}: {e}")
            continue
        any_decoy = any_decoy or has
        print(f"\ninput {i} seed={getattr(inp, 'seed', '?')}  has_decoy={has}")
        entries = resolved._entries
        for j, r in enumerate(entries):
            sealing = type(r.sealing).__name__
            # only print the non-sandbox (value) sealings with interesting state
            if sealing == "SandboxSealing":
                continue
            print(f"  site[{j}] {sealing}: value(delta)={r.value} alts={r.alts} "
                  f"speculative={r.speculative}")
        g = resolved.genuine()
        try:
            import random
            d = resolved.decoy(random.Random(12345))
        except Exception as e:
            d = None
            print(f"  decoy() FAILED: {type(e).__name__}: {e}")
        gd = disasm_reloc(g)
        dd = disasm_reloc(d) if d is not None else []
        if gd != dd:
            print(f"  GENUINE relocs: {gd}")
            print(f"  DECOY   relocs: {dd}")
        else:
            print(f"  decoy == genuine (NULL decoy)  [{len(gd)} relocs]")

    print(f"\nSUMMARY {os.path.basename(template)}: any input decoy-eligible = {any_decoy}")


if __name__ == "__main__":
    main()
