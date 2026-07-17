#!/usr/bin/env python3
"""Does the prefetch fire when the load is BRANCHED OVER (never architecturally executed)?

x2 comes from the input and decides the branch; x1 is the demand address. The load+flush sit on the
skipped path. If the prefetched footprint appears with x2 set to SKIP, the access happened
speculatively (wrong-path) and the prefetcher left a durable, flush-surviving trace of an address the
program never architecturally touched.

The config has enable_pre_run_flush=0, so the branch is NOT PHR-flushed and mistrains naturally.
Single-input batches.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_claims as V

BG = V.BG
ADDR = 0x1080          # faulty+0x080, demand set 2 -> [18,26,34,42,50] when it executes

def run(asm_body, x1, x2, tag):
    asm = '\n'.join(['and x1, x1, #0x1fff', 'add x0, x29, x1'] + asm_body + ['dsb ish', 'isb', ''])
    tc = V.assemble(asm)
    _, patched = V.ex._sandboxed_tc()
    V.ex._sandboxed_cache = (tc, patched)
    res = []
    for _ in range(V.N_TRIALS):
        inp = V.BASE.copy()
        inp[0]['gpr'][1] = x1
        inp[0]['gpr'][2] = x2
        ht, _ = V.ex.trace_test_case([V.ex.as_executor_input(inp)], V.REPS)
        raw = ht[0].raw; n = len(raw) or 1
        fr = [sum((r >> b) & 1 for r in raw)/n for b in range(64)]
        d = (x1 // 64) % 64
        res.append(([b for b in range(64) if fr[b] > 0.5 and b not in BG], fr[d]))
    lits = [r[0] for r in res]
    same = all(l == lits[0] for l in lits)
    print(f"  {tag:<46} {str(lits[0]):<30} demand_freq={res[0][1]:.2f} "
          f"{'stable' if same else '*** UNSTABLE: ' + str(lits[1:]) + ' ***'}")

# cbz x2, skip  => x2==0 SKIPS the load; x2!=0 EXECUTES it
GUARDED = ['cbz x2, .Lskip', 'ldr x3, [x0]', 'dc civac, x0', '.Lskip:']

print(f"### 1 input/batch, {V.REPS} reps, {V.N_TRIALS} trials. demand set INCLUDED.")
print(f"### demand {ADDR:#06x} (set {(ADDR//64)%64}); executing the load gives [18,26,34,42,50]\n")

print("=== controls (no branch) ===")
run(['ldr x3, [x0]', 'dc civac, x0'], ADDR, 0, 'unguarded load + flush          (must prefetch)')
run(['nop'],                          ADDR, 0, 'no load at all                  (must be empty)')

print("\n=== the load is behind a branch ===")
run(GUARDED, ADDR, 1, 'cbz x2 / x2=1 -> load EXECUTES  (arch path)')
run(GUARDED, ADDR, 0, 'cbz x2 / x2=0 -> load SKIPPED   (<-- the question)')

print("\n=== same, but the address is only computed on the taken path (tighter) ===")
TIGHT = ['cbz x2, .Ls2', 'ldr x3, [x0]', 'dc civac, x0', '.Ls2:']
run(TIGHT, 0x1200, 0, 'SKIPPED, demand outside trigger zone (set 8)')
run(TIGHT, ADDR,   0, 'SKIPPED, demand set 2')
