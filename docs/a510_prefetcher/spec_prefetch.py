#!/usr/bin/env python3
"""Does a MISPREDICTED branch produce a prefetch for a load that never architecturally executes?

branch_over_load.py showed no prefetch when the load is skipped -- but there the branch was constant over
all reps, so the BPU predicted it correctly and the load never issued at all. Here we force a real
misprediction with the per-input BPU-training section (byte_offset, train_taken):

  body:  0: and x1,x1,#0x1fff   4: add x0,x29,x1   8: cbz x2,.Lskip   12: ldr x3,[x0]
        16: dc civac,x0        20: .Lskip: dsb ish  24: isb

  x2 = 0            -> cbz is architecturally TAKEN  -> the load is SKIPPED
  train_taken = 0   -> BPU trained NOT-taken         -> fall-through predicted -> load issues SPECULATIVELY

If the footprint appears, the prefetcher leaves a durable, flush-surviving trace of an address the program
never architecturally touched. Single-input batches.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_claims as V
from src.aarch64.aarch64_executor_input_encoder import ExecutorInput

BG, ADDR, BR_OFF = V.BG, 0x1080, 8

BODY = ['and x1, x1, #0x1fff', 'add x0, x29, x1',
        'cbz x2, .Lskip', 'ldr x3, [x0]', 'dc civac, x0', '.Lskip:', 'dsb ish', 'isb', '']

def run(x2, training, tag):
    tc = V.assemble('\n'.join(BODY))
    _, patched = V.ex._sandboxed_tc()
    V.ex._sandboxed_cache = (tc, patched)
    res = []
    for _ in range(V.N_TRIALS):
        inp = V.BASE.copy()
        inp[0]['gpr'][1] = ADDR
        inp[0]['gpr'][2] = x2
        ei = ExecutorInput(input_=inp, bpu_training=training)
        ht, _ = V.ex.trace_test_case([ei], V.REPS)
        raw = ht[0].raw; n = len(raw) or 1
        fr = [sum((r >> b) & 1 for r in raw)/n for b in range(64)]
        res.append([b for b in range(64) if fr[b] > 0.5 and b not in BG])
    same = all(r == res[0] for r in res)
    print(f"  {tag:<52} {str(res[0]):<30} {'stable' if same else '*** UNSTABLE: ' + str(res[1:])}")

print(f"### 1 input/batch, {V.REPS} reps, {V.N_TRIALS} trials. demand {ADDR:#06x} (set 2).")
print(f"### executing the load gives [18,26,34,42,50]; branch at byte offset {BR_OFF}\n")

print("=== controls ===")
run(1, (), 'x2=1 -> load EXECUTES architecturally      (no training)')
run(0, (), 'x2=0 -> load SKIPPED, branch well-predicted (no training)')

print("\n=== forced MISPREDICTION: arch = taken (skip), BPU trained not-taken ===")
run(0, ((BR_OFF, False),), 'x2=0, train_taken=0  -> SPECULATIVE load  (<-- the question)')

print("\n=== training-direction control: train toward the architectural direction ===")
run(0, ((BR_OFF, True),),  'x2=0, train_taken=1  -> predicted correctly, no spec')
