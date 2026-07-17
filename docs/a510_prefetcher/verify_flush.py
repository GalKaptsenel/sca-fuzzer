#!/usr/bin/env python3
"""Experiment F, done correctly: show the DEMAND set too.

verify_claims.py excluded the demand set from its output, which hid the very line the flush is supposed to
remove. Here nothing is filtered except the harness background sets, so both effects are visible:
  - the demand line disappears when the victim flushes it   (the flush works)
  - the prefetched lines do NOT                              (the flush is not an eraser)
Single-input batches only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_claims as V   # reuses assemble/victim/ex/BASE/REPS/N_TRIALS

BG = V.BG

def lit_alone(addr, flush_demand):
    """All lit sets (background excluded, demand set KEPT)."""
    tc_bytes = V.assemble(V.victim((), flush_demand))
    _, patched = V.ex._sandboxed_tc()
    V.ex._sandboxed_cache = (tc_bytes, patched)
    inp = V.BASE.copy()
    inp[0]['gpr'][1] = addr
    htraces, _ = V.ex.trace_test_case([V.ex.as_executor_input(inp)], V.REPS)
    raw = htraces[0].raw; n = len(raw) or 1
    fr = [sum((r >> b) & 1 for r in raw)/n for b in range(64)]
    return [b for b in range(64) if fr[b] > 0.5 and b not in BG], fr

print(f"### F (corrected): demand set INCLUDED. 1 input/batch, {V.REPS} reps, {V.N_TRIALS} trials.")
print("### bg sets 15/23/31 excluded (harness background).\n")
for addr in (0x1000, 0x1080, 0x10c0, 0x1100):
    d = (addr // 64) % 64
    print(f"--- demand {addr:#06x} = faulty+{addr-0x1000:#05x}, demand set {d}")
    for trial in range(1, V.N_TRIALS + 1):
        no, frn = lit_alone(addr, False)
        ye, fry = lit_alone(addr, True)
        gone  = sorted(set(no) - set(ye))
        stayed = sorted(set(no) & set(ye))
        print(f"  trial {trial}: without={str(no):<28} with={str(ye):<26} "
              f"| demand-set freq {frn[d]:.2f} -> {fry[d]:.2f} | removed={gone} survived={stayed}")
