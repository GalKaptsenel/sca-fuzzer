#!/usr/bin/env python3
"""Speculative prefetch? Train the branch NATURALLY (the BPU-training section is broken).

TC:  and x1,x1,#0x1fff ; add x0,x29,x1 ; cbz x2,.Lskip ; ldr x3,[x0] ; dc civac,x0 ; .Lskip: dsb; isb
     x2 != 0 -> branch NOT taken -> load EXECUTES
     x2 == 0 -> branch TAKEN     -> load SKIPPED

A batch runs its inputs back-to-back and repeats, so N training inputs placed before the test input train
the BPU each round. The trainers load from set 8 -- OUTSIDE the trigger zone -- so they steer the branch
without prefetching anything themselves (otherwise we would measure their contamination, not speculation).

  MISTRAIN : trainers x2=1 (fall through) -> BPU learns NOT-taken.
             test x2=0 -> architecturally SKIPS, but is predicted fall-through -> the load issues
             SPECULATIVELY at the trigger address. Prefetch => speculative, flush-surviving trace of an
             address never architecturally accessed.
  CONTROL  : trainers x2=0 (taken) -> BPU learns taken -> test predicted correctly -> no speculation.

Only the last (test) input is measured.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_claims as V

BG        = V.BG
TRIG      = 0x1080     # faulty+0x080, set 2 -> [18,26,34,42,50] when the load executes
NOTRIG    = 0x1200     # set 8 -> prefetches nothing (trainer address)
N_TRAIN   = int(os.environ.get('NTRAIN', '20'))

BODY = ['and x1, x1, #0x1fff', 'add x0, x29, x1',
        'cbz x2, .Lskip', 'ldr x3, [x0]', 'dc civac, x0', '.Lskip:', 'dsb ish', 'isb', '']

def mk(x1, x2):
    inp = V.BASE.copy()
    inp[0]['gpr'][1] = x1
    inp[0]['gpr'][2] = x2
    return V.ex.as_executor_input(inp)

def run(train_x2, test_x1, test_x2, tag):
    tc = V.assemble('\n'.join(BODY))
    _, patched = V.ex._sandboxed_tc()
    V.ex._sandboxed_cache = (tc, patched)
    res = []
    for _ in range(V.N_TRIALS):
        batch = [mk(NOTRIG, train_x2) for _ in range(N_TRAIN)] + [mk(test_x1, test_x2)]
        ht, _ = V.ex.trace_test_case(batch, V.REPS)
        raw = ht[-1].raw; n = len(raw) or 1          # measure ONLY the test input
        fr = [sum((r >> b) & 1 for r in raw)/n for b in range(64)]
        res.append([b for b in range(64) if fr[b] > 0.5 and b not in BG])
    same = all(r == res[0] for r in res)
    print(f"  {tag:<56} {str(res[0]):<30} {'stable' if same else '*** UNSTABLE: ' + str(res[1:])}")

print(f"### {N_TRAIN} trainers (@set 8) + 1 test, {V.REPS} reps, {V.N_TRIALS} trials, measuring the test only.")
print(f"### test addr {TRIG:#06x} (set 2); an executed load gives [18,26,34,42,50]\n")

print("=== sanity: the batch shape itself does not manufacture the footprint ===")
run(0, NOTRIG, 1, 'trainers taken;     test x2=1 load EXECUTES @set8 (must be [])')
run(0, TRIG,   1, 'trainers taken;     test x2=1 load EXECUTES @set2 (must prefetch)')

print("\n=== the question ===")
run(1, TRIG, 0, 'trainers FALL THROUGH; test x2=0 SKIPS @set2  <-- MISTRAINED, speculative?')
run(0, TRIG, 0, 'trainers TAKEN;       test x2=0 SKIPS @set2  <-- control, predicted right')
