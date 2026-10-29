#!/usr/bin/env python3
"""All-16-tags sweep on a TEMPLATE (no violation dir). For the suspect input, hold the fixed prefix and
sweep each ADDG seal's pointer-tag delta uimm4 0..15, reading the given probe sets at the tail. Genuine
tag-check: the probe is sharply separated at the MATCH tag (peaked, one tag stands out). Prefetcher/noise:
flat or inverted across tags.

Usage: sweep_v2_template.py -s base.json -c <cfg> -t <template> --sus N --probes a,b,c [--reps N] [--prefix P] [--memtag M]
"""
import argparse, os, sys

ADDG_OP = 0x91800000


def residency(ht, nsets=64):
    raw = ht.raw
    return [0.0] * nsets if not raw else [sum((int(r) >> s) & 1 for r in raw) / len(raw) for s in range(nsets)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--spec", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-t", "--template", required=True)
    ap.add_argument("--sus", type=int, default=0)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--prefix", type=int, default=12)
    ap.add_argument("--memtag", type=int, default=6)
    ap.add_argument("--repo", default="/home/gal_k_1_1998/revizor/sca-fuzzer")
    a = ap.parse_args()
    probes = [int(x) for x in a.probes.split(",")]
    sys.path.insert(0, os.path.abspath(a.repo))
    spec, config, template = map(os.path.abspath, (a.spec, a.config, a.template))
    from src.config import CONF
    CONF.load(config)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import ExecutorInput
    wd = "/tmp/sweep_v2_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(spec, wd, template, "")
    fz.initialize_modules()
    tc = fz.generator.create_test_case_from_template(template)
    ex = fz.executor
    ex.load_test_case(tc)
    inputs = fz.input_gen.generate(a.prefix + 6)
    inp = inputs[a.sus]
    g = ex.genuine_variant(inp)
    Reloc = type(g.code_reloc[0]); relocs = list(g.code_reloc)
    addgs = sorted([j for j, r in enumerate(relocs) if (r.value & 0xFFC00000) == ADDG_OP],
                   key=lambda j: relocs[j].offset)

    def u4of(j):
        return (relocs[j].value >> 10) & 0xF

    # Pair each before-ADDG with its re-converge ADDG (the next ADDG within 16 bytes). The before carries
    # the match delta C; its re-converge carries (correct-delta), = 0 in the genuine variant (C+0=C).
    pairs = []
    used = set()
    for a_i, j in enumerate(addgs):
        if j in used:
            continue
        rj = None
        for k2 in addgs[a_i + 1:]:
            if 0 < relocs[k2].offset - relocs[j].offset <= 16:
                rj = k2; break
        pairs.append((j, rj))
        used.add(j)
        if rj is not None:
            used.add(rj)
    print(f"ADDG gates(idx@off): {[(j, hex(relocs[j].offset), u4of(j)) for j in addgs]}")
    print(f"before/reconverge pairs (before_idx, reconv_idx): {pairs}")
    others = [i for i in range(a.prefix + 6) if i != a.sus][:a.prefix]
    prefix = [ex.genuine_variant(inputs[i]) for i in others]

    def tail(v):
        hts, _ = ex.trace_test_case(prefix + [v], a.reps)
        return residency(hts[-1])

    def setu4(plan, idx, u):
        plan[idx] = Reloc(plan[idx].offset, (plan[idx].value & ~(0xF << 10)) | ((u & 0xF) << 10), plan[idx].rtype)

    for (k, rk) in pairs:
        C = u4of(k)
        base_ptr_tag = (a.memtag - C) % 16
        print(f"\n=== sweep before=idx{k}@{hex(relocs[k].offset)} reconv="
              f"{('idx'+str(rk)+'@'+hex(relocs[rk].offset)) if rk is not None else 'NONE'} "
              f"genuine_uimm4(match)={C} memtag={a.memtag} ===")
        print(f"{'u4':>3} {'ptag':>4} {'flag':>9} | " + " ".join(f"s{p}" for p in probes))
        for u4 in range(16):
            plan = list(relocs)
            setu4(plan, k, u4)
            if rk is not None:
                setu4(plan, rk, (C - u4) % 16)   # re-converge so the pointer ends at the cell tag
            var = ExecutorInput(g.input_, code_reloc=tuple(plan), mte_tags=g.mte_tags,
                                pac_keys=g.pac_keys, bpu_training=g.bpu_training, env_plan=g.env_plan)
            res = tail(var)
            ptag = (base_ptr_tag + u4) % 16
            flag = "MATCH" if ptag == a.memtag else ("UNCHK" if ptag in (0, 15) else "mismatch")
            print(f"{u4:>3} {ptag:>4} {flag:>9} | " + " ".join(f"{res[p]:.2f}" for p in probes))


if __name__ == "__main__":
    main()
