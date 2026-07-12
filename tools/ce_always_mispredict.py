#!/usr/bin/env python3
"""Run the contract executor under ALWAYS_MISPREDICT on two inputs of a violation dir,
and save the thorough per-instruction CE traces (same format as input_*_trace_log.txt).

    python3 tools/ce_always_mispredict.py <violation_dir> <idxA> <idxB> [config.yml]

Model only (no hardware / no BPU retraining). Prints ctrace hashes + arch/spec cache
sets, and writes <dir>/input_<idx>_ce_always_mispredict.txt for each input.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
D, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = sys.argv[4] if len(sys.argv) > 4 else "configs/spectre_v1_pp.yml"
from contextlib import redirect_stdout
from src.config import CONF
CONF.load(cfg, None)
CONF.contract_execution_clause = ["cond"]    # -> ALWAYS_MISPREDICT
from src.factory import get_fuzzer
from src.aarch64.aarch64_trace import show_context
fz = get_fuzzer("base.json", "/tmp/ce_work", None, "")
fz.initialize_modules(); ex = fz.executor
tc = fz.asm_parser.parse_file(f"{D}/generated.asm"); ex.load_test_case(tc)
def _inp(i):  # saved inputs are input_NNNN_nzcv_scheme.bin (old: input_NNNN.bin)
    base = f"{D}/input_{i:04d}"
    return next(base + s for s in ("_nzcv_scheme.bin", ".bin") if os.path.exists(base + s))
inputs = fz.input_gen.load([_inp(a), _inp(b)])
ctraces, _, traces = ex.trace_test_case_with_taints(inputs, 5)
print(f"contract = ALWAYS_MISPREDICT   inputs #{a}, #{b}")
print(f"ctrace #{a} = {ctraces[0]}")
print(f"ctrace #{b} = {ctraces[1]}")
print(f"ctraces EQUAL: {ctraces[0] == ctraces[1]}")
def sets(tr):
    arch, spec = set(), set()
    for ite in tr:
        m = ite.metadata
        if m.has_memory_access:
            s = ((m.memory_access.effective_address - ite.cpu.gpr[29]) // 64) % 64
            (arch if m.speculation_nesting == 0 else spec).add(s)
    return sorted(arch), sorted(spec)
for idx, tr in ((a, traces[0]), (b, traces[1])):
    arch, spec = sets(tr)
    print(f"  #{idx}: arch sets {arch} | spec-only {sorted(set(spec) - set(arch))}")
    out = f"{D}/input_{idx:04d}_ce_always_mispredict.txt"
    with open(out, "w") as f, redirect_stdout(f):
        show_context(tr, -1); print(); tr.pretty_print()
    print(f"  -> saved thorough trace: {out}")
