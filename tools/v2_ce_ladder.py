#!/usr/bin/env python3
"""CE-only contract-ladder triage for the v2 (indirect dispatch) violation.

Passes data_size (dispatch-table bytes) so the CE does not misparse the appended
table as code and SIGBUS on the indirect BLR.

    python3 tools/v2_ce_ladder.py <violation-dir> <A> <B>
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
D, A, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

from src.config import CONF
CONF.load(f"{D}/reproduce.yaml", None)
from src.factory import get_input_generator
from src.aarch64.aarch64_generator import Aarch64Generator
from src.aarch64.aarch64_contract_executor import (
    ContractExecution, ContractExecutorService, ExecutionClause as E, BranchPredictor, SimArch)
from src.aarch64.aarch64_trace import compute_ctrace
from src.aarch64.aarch64_executor import _ce_memory_regs

asm = open(f"{D}/sandboxed_test_case.asm").read()
tc_bytes = Aarch64Generator.in_memory_assemble(asm)
# trailing dispatch-table bytes = 4 * number of .word entries after the dispatch label
ndisp = len(re.findall(r'^\.word ', asm, re.M))
data_size = 4 * ndisp
print(f"# tc_bytes={len(tc_bytes)}  dispatch .word entries={ndisp}  data_size={data_size}")

inA, inB = get_input_generator(0).load([f"{D}/input_{A:04d}.reif", f"{D}/input_{B:04d}.reif"])
ce = ContractExecutorService(os.path.join(ROOT, "src/aarch64/contract_executor/contract_executor"))

def run(inp, clause, bp, nest, base):
    mem, regs = _ce_memory_regs(inp)
    exn = ContractExecution(tc_bytes, mem, regs, SimArch.RVZR_ARCH_AARCH64, nest,
                            CONF.model_max_spec_window, req_mem_base_virt=base,
                            execution_clauses=clause, branch_predictor=bp, data_size=data_size)
    return ce.run(exn)

rep = open(f"{D}/report.txt").read()
m = re.search(rf'Input #{A}\b.*?Contract trace \(hash\):\s*(\d+)', rep, re.S)
want = int(m.group(1)) if m else None
print(f"# report contract hash (run contract cond-bpas nest30) for #{A} = {want}")

cands = [0xffff400095321000, 0x0, 0x1000000, 0x10000000, 0x400095321000]
try:
    cands.insert(0, int(open("/sys/executor/print_sandbox_base").read(), 16))
except Exception as e:
    print(f"# (/sys base read skipped: {e})")

base = None
for c in cands:
    try:
        h = compute_ctrace(run(inA, E.COND | E.BPAS, BranchPredictor.NONE, 30, c)).hash_
    except Exception as e:
        print(f"#  base {hex(c)}: CE error {repr(e)[:80]}"); continue
    print(f"#  base {hex(c)}: cond-bpas nest30 hash {h} {'== report (USE)' if h==want else '!= report'}")
    if h == want:
        base = c; break
if base is None:
    sys.exit("no candidate base reproduced the report's cond-bpas hash")

LADDER = [
    ("seq",       E.SEQ,            BranchPredictor.NONE, 0),
    ("cond",      E.COND,           BranchPredictor.NONE, 30),
    ("bpas",      E.BPAS,           BranchPredictor.NONE, 30),
    ("cond-bpas", E.COND | E.BPAS,  BranchPredictor.NONE, 30),
    ("bpu_n3",    E.BPU,            BranchPredictor.NEOVERSE_N3, 30),
]

def sets(cer):
    arch, spec = set(), set()
    for ite in cer:
        md = ite.metadata
        if md.has_sandbox_memory_access:
            s = ((md.memory_access.effective_address - ite.cpu.gpr[29]) // 64) % 64
            (arch if md.speculation_nesting == 0 else spec).add(s)
    return sorted(arch), sorted(spec)

print(f"\n# base = {hex(base)}   pair #{A} vs #{B}   HW htrace-diff sets = [3, 9]\n")
for nm, cl, bp, nest in LADDER:
    cA, cB = run(inA, cl, bp, nest, base), run(inB, cl, bp, nest, base)
    hA, hB = compute_ctrace(cA).hash_, compute_ctrace(cB).hash_
    aA, sA = sets(cA); aB, sB = sets(cB)
    allA, allB = set(aA) | set(sA), set(aB) | set(sB)
    dv = sorted(allA ^ allB)
    tag = "AGREE   " if hA == hB else "DISAGREE"
    match = "  <-- divergent sets include HW {3,9}" if ({3, 9} & set(dv)) else ""
    print(f"{nm:>10}: {tag}  divergent-sets={dv}{match}")
    if nm == "bpu_n3":
        print(f"             #{A}: arch={aA}  spec-only={sorted(set(sA)-set(aA))}")
        print(f"             #{B}: arch={aB}  spec-only={sorted(set(sB)-set(aB))}")
