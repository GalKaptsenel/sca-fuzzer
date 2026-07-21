#!/usr/bin/env python3
"""General contract-level triage of a Revizor AArch64 violation.

Classifies a saved violation-* as a genuine Spectre leak (and which variant) vs noise, by running the
contract executor (CE, local subprocess) under a configurable set of speculation contracts and comparing
its speculative cache sets to the hardware htrace parsed from report.txt. No hardware re-run (that would
train the BPU and hide the leak) -- for HW confirmation use verify.py.

Usage:
  triage.py <violation-dir> [--repo DIR] [--pair A B] [--base 0xHEX]
            [--contracts seq,cond,bpas,cond-bpas] [--nest 5]

Bulletproof about the base: the `ct` trace is base-dependent, so a wrong base yields spurious seq
DISAGREE. The script tries --base, /sys/executor/print_sandbox_base, then 0, and PICKS whichever makes
the CE seq hash equal report.txt's `Contract trace (hash)`; it aborts if none matches.
"""
import argparse, os, re, sys, collections

def find_repo(*starts):
    for start in starts:
        if not start:
            continue
        d = os.path.abspath(start)
        while d != "/":
            if os.path.isdir(os.path.join(d, "src", "aarch64")):
                return d
            d = os.path.dirname(d)
    sys.exit("could not locate repo root (no src/aarch64); pass --repo")

def parse_pair(report):
    idx = [int(m) for m in re.findall(r'Input #(\d+)\n\* Hardware trace:', report)]
    if len(idx) < 2:
        sys.exit("could not find >=2 counterexample inputs in report.txt")
    return idx[0], idx[1]

def report_hash(report, idx):
    m = re.search(rf'Input #{idx}\b.*?Contract trace \(hash\):\s*(\d+)', report, re.S)
    return int(m.group(1)) if m else None

def htrace_freq(report, idx):
    """Per-SET '^' frequency over the FULL distribution (every `[count]` line, not just the dominant).
    report.txt strings are BIT-REVERSED: string position j == cache set (63-j). We key by set so the
    result is directly comparable to the CE's (addr>>6)&63 sets and to verify.py's raw-int bins."""
    m = re.search(rf'Input #{idx}\n\* Hardware trace:(.*?)(?:\* Contract|Input #\d|\Z)', report, re.S)
    acc, tot, W = collections.defaultdict(int), 0, 0
    for mm in re.finditer(r'([.\^]{40,})\s*\[(\d+)\]', m.group(1) if m else ""):
        t, c = mm.group(1), int(mm.group(2)); tot += c; W = max(W, len(t))
        for j, ch in enumerate(t):
            if ch == '^': acc[(W - 1) - j] += c            # position -> set (reverse)
    return {b: acc[b] / tot for b in acc} if tot else {}

def input_paths(vd, a, b):
    for ext in ("reif", "bin"):
        pa, pb = f"{vd}/input_{a:04d}.{ext}", f"{vd}/input_{b:04d}.{ext}"
        if os.path.exists(pa) and os.path.exists(pb):
            return pa, pb
    sys.exit(f"input files for #{a}/#{b} not found (.reif/.bin)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vd")
    ap.add_argument("--repo"); ap.add_argument("--pair", nargs=2, type=int)
    ap.add_argument("--base"); ap.add_argument("--nest", type=int, default=5)
    ap.add_argument("--contracts", default="seq,cond,bpas,cond-bpas")
    a = ap.parse_args()
    vd = os.path.abspath(a.vd)
    repo = find_repo(a.repo, vd, __file__)
    sys.path.insert(0, repo)
    from src.config import CONF; CONF.load(f"{vd}/reproduce.yaml")
    from src.factory import get_input_generator
    from src.aarch64.aarch64_generator import Aarch64Generator
    from src.aarch64.aarch64_contract_executor import (
        ContractExecution, ContractExecutorService, ExecutionClause as E, SimArch,
        SUPPORTED_EXECUTION_CLAUSES)
    from src.aarch64.aarch64_trace import compute_ctrace
    from src.aarch64.aarch64_executor import _ce_memory_regs

    # Canonical name for any supported clause combination (built from the flag bits, not hardcoded).
    def clause_name(cl):
        bits = [f.name.lower() for f in E if f and int(f) and (cl & f) == f]
        return "-".join(bits) if bits else "seq"
    CLAUSE = {clause_name(cl): (cl, 0 if cl == E.SEQ else a.nest) for cl in SUPPORTED_EXECUTION_CLAUSES}
    CLAUSE["seq"] = (E.SEQ, 0)
    if a.contracts.strip() == "all":
        names = ["seq"] + sorted(n for n in CLAUSE if n != "seq")
    else:
        names = [c.strip() for c in a.contracts.split(",") if c.strip()]
        unknown = [n for n in names if n not in CLAUSE]
        if unknown:                                   # no silent fallback
            sys.exit(f"unknown contract(s) {unknown}; known: {sorted(CLAUSE)} (or 'all')")
        if "seq" not in names: names = ["seq"] + names   # seq is the violation premise

    report = open(f"{vd}/report.txt").read()
    A, B = a.pair if a.pair else parse_pair(report)
    tc = Aarch64Generator.in_memory_assemble(open(f"{vd}/sandboxed_test_case.asm").read())
    pa, pb = input_paths(vd, A, B)
    inA, inB = get_input_generator(0).load([pa, pb])
    ce = ContractExecutorService(os.path.join(repo, "src/aarch64/contract_executor/contract_executor"))

    def ct(inp, clause, nest, base):
        m, r = _ce_memory_regs(inp)
        ex = ContractExecution(tc, m, r, SimArch.RVZR_ARCH_AARCH64, nest,
                               CONF.model_max_spec_window, req_mem_base_virt=base, execution_clauses=clause)
        return compute_ctrace(ce.run(ex))

    # --- pick a base that reproduces the report's ct-seq hash (base-dependent!) ---
    cands = []
    if a.base: cands.append(int(a.base, 16))
    try: cands.append(int(open("/sys/executor/print_sandbox_base").read(), 16))
    except Exception: pass
    cands += [0x0, 0x1000000]
    want = report_hash(report, A)
    base = None
    for c in cands:
        if ct(inA, E.SEQ, 0, c).hash_ == want or want is None:
            base = c; break
    if base is None:
        sys.exit(f"no candidate base reproduces report hash {want} for in#{A}; pass the real --base "
                 f"(read /sys/executor/print_sandbox_base on the device). tried={[hex(c) for c in cands]}")
    print(f"# {os.path.basename(vd)}  in#{A} vs in#{B}  base={hex(base)}"
          + ("  (validated vs report hash)" if want else "  (report hash absent, unvalidated)"))

    f0, f1 = htrace_freq(report, A), htrace_freq(report, B)
    hdiff = sorted(s for s in set(f0) | set(f1) if abs(f0.get(s, 0) - f1.get(s, 0)) >= 0.10)
    print(f"OBSERVED htrace-diff sets (>=10%): {hdiff}")
    for nm in names:
        cl, nest = CLAUSE[nm]
        cA, cB = ct(inA, cl, nest, base), ct(inB, cl, nest, base)
        if cA.hash_ == cB.hash_:
            print(f"{nm:>10}: AGREE"); continue
        bA = {(v >> 6) & 63 for v in cA.raw}; bB = {(v >> 6) & 63 for v in cB.raw}
        dv = sorted(bA ^ bB)
        tag = "  <-- MATCHES htrace" if set(dv) & set(hdiff) else "  (no htrace match)"
        print(f"{nm:>10}: DISAGREE  divergent-sets={dv}{tag}")

if __name__ == "__main__":
    main()
