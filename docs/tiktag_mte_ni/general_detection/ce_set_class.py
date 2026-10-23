#!/usr/bin/env python3
"""Local CE set classifier. Run the contract executor on the genuine variant of a violation's suspect
input and, for every L1D set, report whether it is touched architecturally (speculation_nesting==0) or
ONLY speculatively (nesting>0). Prints the accesses (pc offset, nesting, load/store, effective address,
granule) that land in each queried set. No device.

Usage: ce_set_class.py <cfg> <violation-dir> <suspect_idx> <set,set,...>
"""
import glob
import os
import sys


def main():
    cfg, vd = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    sus = int(sys.argv[3])
    query = set(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    repo = "/home/gal_k_1_1998/revizor/sca-fuzzer"
    sys.path.insert(0, repo)
    from src.config import CONF
    CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize
    from src.aarch64.aarch64_relocations import apply_relocations
    from src.aarch64.aarch64_contract_executor import ExecutionClause
    from src.aarch64.aarch64_mte import MTE_GRANULE
    wd = "/tmp/ce_set_class_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(repo, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    ex = fz.executor
    ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
    inp = deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0], "rb").read()).input_
    if ex._sandbox_base is None:
        ex._sandbox_base, _ = ex.read_base_addresses()
    g = ex.genuine_variant(inp)
    sealed = ex._sealer.seal(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
    rsv = sealed.resolve(inp)
    code = apply_relocations(rsv.object_code, rsv.genuine())
    cer = ex._contract_executor.run(ex._make_ce_execution(
        code, inp, ex._sandbox_base, CONF.model_max_nesting, CONF.model_max_spec_window,
        ExecutionClause.COND | ExecutionClause.BPAS, mte_tags=ex._mte_tags_for(inp),
        pac_keys=ex._input_pac_keys(inp), pac_profile=ex._pac_profile_value))
    cb = cer[0].cpu.pc
    MASK = (1 << 56) - 1
    arch_sets, spec_sets = set(), set()
    hits = {}  # set -> list of (nesting, ld/st, off, ea, granule)
    for ite in cer:
        nest = int(ite.metadata.speculation_nesting)
        off = ite.cpu.pc - cb
        base = ite.cpu.gpr[29]
        for ma in ite.metadata.accesses():
            ea = ma.effective_address & MASK
            cset = (ea >> 6) & 63
            gran = ((ea & ~(MTE_GRANULE - 1)) - ((base & MASK) & ~(MTE_GRANULE - 1))) // MTE_GRANULE
            (arch_sets if nest == 0 else spec_sets).add(cset)
            hits.setdefault(cset, []).append((nest, off, ea, gran))
    spec_only = sorted(spec_sets - arch_sets)
    print(f"{os.path.basename(vd)} suspect#{sus}")
    print(f"architectural sets (nesting0): {sorted(arch_sets)}")
    print(f"speculative-only sets (nesting>0, NOT arch): {spec_only}")
    if query:
        for s in sorted(query):
            klass = "ARCH" if s in arch_sets else ("SPEC-ONLY" if s in spec_sets else "UNTOUCHED-by-CE")
            print(f"\nset {s}: {klass}")
            for (nest, off, ea, gran) in hits.get(s, []):
                print(f"   nesting={nest} off=0x{off:x} ea=0x{ea:x} granule={gran}")


if __name__ == "__main__":
    main()
