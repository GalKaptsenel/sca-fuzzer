#!/usr/bin/env python3
"""Adversarial tests: try to REFUTE "the prefetched lines survive the flush", plus an access-type
matrix and multi-bit address combinations. Single-input batches, N trials each."""
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_claims as V

BG = V.BG

def run_asm(body, addr, keep_demand=True):
    """body: list of asm lines; x0 = x29 + (x1 & 0x1fff) is set up first."""
    asm = '\n'.join(['and x1, x1, #0x1fff', 'add x0, x29, x1'] + body + ['dsb ish', 'isb', ''])
    tc = V.assemble(asm)
    _, patched = V.ex._sandboxed_tc()
    V.ex._sandboxed_cache = (tc, patched)
    inp = V.BASE.copy(); inp[0]['gpr'][1] = addr
    ht, _ = V.ex.trace_test_case([V.ex.as_executor_input(inp)], V.REPS)
    raw = ht[0].raw; n = len(raw) or 1
    fr = [sum((r >> b) & 1 for r in raw)/n for b in range(64)]
    d = (addr // 64) % 64
    lit = [b for b in range(64) if fr[b] > 0.5 and b not in BG and (keep_demand or b != d)]
    return lit, fr[d]

def show(tag, body, addr, keep_demand=True):
    res = [run_asm(body, addr, keep_demand) for _ in range(V.N_TRIALS)]
    lits = [r[0] for r in res]
    same = all(l == lits[0] for l in lits)
    print(f"  {tag:<44} {str(lits[0]):<30} demand_freq={res[0][1]:.2f} "
          f"{'stable' if same else '*** UNSTABLE: ' + str(lits[1:]) + ' ***'}")

DELAY = ['nop'] * 400
print(f"### 1 input/batch, {V.REPS} reps, {V.N_TRIALS} trials. demand set INCLUDED in lit.\n")

print("=== H1 REFUTATION: is it a RACE (prefetch lands after the flush) rather than survival? ===")
print("    If a long delay before the flush still leaves the prefetched lines, it is NOT a race.")
show('ldr; dc civac                  (baseline)', ['ldr x3, [x0]', 'dc civac, x0'], 0x1080)
show('ldr; dsb; isb; 400 nops; dc civac',
     ['ldr x3, [x0]', 'dsb ish', 'isb'] + DELAY + ['dc civac, x0'], 0x1080)
show('ldr; dsb; isb; 400 nops (NO flush)',
     ['ldr x3, [x0]', 'dsb ish', 'isb'] + DELAY, 0x1080)

print("\n=== H2 REFUTATION: does the F+R reload walk itself create the lines? ===")
show('NO memory access at all (control)',        ['nop'], 0x1080)
show('dc civac only, never accessed (control)',  ['dc civac, x0'], 0x1080)
show('ldr OUTSIDE trigger zone (set 8)',         ['ldr x3, [x0]', 'dc civac, x0'], 0x1200)
show('ldr at set 2 (positive control)',          ['ldr x3, [x0]', 'dc civac, x0'], 0x1080)

print("\n=== H3: does the flush actually remove the demand line? (demand_freq must go to 0) ===")
show('ldr, no flush',                            ['ldr x3, [x0]'], 0x1080)
show('ldr, flush',                               ['ldr x3, [x0]', 'dc civac, x0'], 0x1080)

print("\n=== G: ACCESS-TYPE MATRIX (all with dc civac of the demand line) ===")
TYPES = [
    ('ldr  x3,[x0]   64-bit load',  ['ldr x3, [x0]']),
    ('ldr  w3,[x0]   32-bit load',  ['ldr w3, [x0]']),
    ('ldrh w3,[x0]   16-bit load',  ['ldrh w3, [x0]']),
    ('ldrb w3,[x0]    8-bit load',  ['ldrb w3, [x0]']),
    ('ldur x3,[x0]   unscaled',     ['ldur x3, [x0, #0]']),
    ('ldp  x3,x4,[x0] pair load',   ['ldp x3, x4, [x0]']),
    ('str  x3,[x0]   64-bit store', ['str x3, [x0]']),
    ('strb w3,[x0]    8-bit store', ['strb w3, [x0]']),
    ('stp  x3,x4,[x0] pair store',  ['stp x3, x4, [x0]']),
    ('prfm pldl1keep  SW prefetch', ['prfm pldl1keep, [x0]']),
    ('ldar x3,[x0]   acquire load', ['ldar x3, [x0]']),
]
for addr in (0x1000, 0x1080):
    print(f"  --- demand {addr:#06x} (set {(addr//64)%64})")
    for tag, body in TYPES:
        show('    ' + tag, body + ['dc civac, x0'], addr)

print("\n=== J: MULTIPLE BITS SET in the demand offset ===")
for addr in (0x1000, 0x1040, 0x1080, 0x10c0, 0x1140, 0x1180, 0x11c0,
             0x1240, 0x1480, 0x1840, 0x10c1, 0x10ff):
    d = (addr // 64) % 64
    show(f'    V={addr:#06x} bits[11:6]={((addr>>6)&0x3f):#04x} set={d:<2}',
         ['ldr x3, [x0]', 'dc civac, x0'], addr, keep_demand=False)
