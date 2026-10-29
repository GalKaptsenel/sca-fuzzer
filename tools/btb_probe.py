"""Probe: for the BTB template, check per-input whether a branch-target seal slot exists, whether it is
decoy-eligible (speculative-only), and whether genuine vs decoy binaries differ."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONF

cfg, tmpl, n_inputs = sys.argv[1], sys.argv[2], int(sys.argv[3])
CONF.load(cfg)
from src.factory import get_fuzzer
from src.aarch64.template.runner import load_template, build_test_case

fuzzer = get_fuzzer("base.json", "eval_out/btb", "", "")
fuzzer.initialize_modules()
template = load_template(tmpl)
tc = build_test_case(fuzzer.generator, template)
fuzzer.input_gen.n_actors = len(tc.actors)
inputs = fuzzer.input_gen.generate(n_inputs)
ex = fuzzer.executor
ex.load_test_case(tc)

n_decoy = 0
n_diff = 0
spec_blr = 0
arch_blr = 0
for i, inp in enumerate(inputs):
    resolved = ex._resolve(inp)
    hd = resolved.has_decoy()
    if hd:
        n_decoy += 1
    # entries: list of _Resolved; check branch_target sealings and speculative flag
    for e in resolved._entries:
        cls = type(e.sealing).__name__
        if "BranchTarget" in cls:
            if e.speculative:
                spec_blr += 1
            else:
                arch_blr += 1
    variants = ex.variants_for_input(inp)
    names = list(variants.keys())
    base = variants[names[0]]
    # compare genuine vs each decoy enacted code
    from src.aarch64.seal.sealer import apply_relocations
    gbytes = apply_relocations(resolved.object_code, list(base.code_reloc))
    for nm in names[1:]:
        dbytes = apply_relocations(resolved.object_code, list(variants[nm].code_reloc))
        if dbytes != gbytes:
            n_diff += 1
            break
    if i < 5:
        seals = [(type(e.sealing).__name__, e.speculative, e.value, e.alts) for e in resolved._entries]
        print(f"inp {i}: has_decoy={hd}  n_entries={len(resolved._entries)}")
        for s in seals:
            print(f"    {s}")

print(f"\nhas_decoy: {n_decoy}/{n_inputs}   genuine!=decoy binaries: {n_diff}/{n_inputs}")
print(f"BranchTarget seal slots: speculative={spec_blr}  architectural={arch_blr}")
