#!/usr/bin/env python3
"""Exhaustive verification of the A510 prefetcher covert channel.

The channel's entire single-symbol input space is the demand SET = offset bits [11:6] of the faulty-page
access. Bits [5:0] are ignored (same cache line), so there are exactly 64 distinguishable inputs
(demand sets 0..63). This sweeps ALL 64, measures each footprint, and checks it against the
characterized catalog (KNOWN_PATTERNS). Every mismatch is reported.

For each demand set s:  V = FAULTY | (s << 6)  ->  ldr [x29+V] ; dc civac  ->  htrace footprint.

Env: REPS (default 100)  PASSES (default 3, to expose instability)  OUT (csv path).
"""
import os, sys, csv

os.chdir('/home/gal_k_1_1998/sca-fuzzer')
sys.path.insert(0, 'docs/a510_prefetcher')
from poc_covert_channel import (footprint, hot, pin, KNOWN_PATTERNS, characterized_footprint, FAULTY, BG)

REPS   = int(os.environ.get('REPS', '100'))
PASSES = int(os.environ.get('PASSES', '3'))
OUT    = os.environ.get('OUT', 'docs/a510_prefetcher/verify_all_results.csv')


def main():
    pin(0)
    print(f"=== EXHAUSTIVE VERIFICATION: all 64 demand sets, {PASSES} passes x {REPS} reps ===")
    print(f"    footprint = prefetched cache sets (background {sorted(BG)} excluded), demand line flushed\n")
    rows = []
    n_ok = n_bad = n_unstable = 0
    print(f"  {'set':>3}  {'V':>7}  {'expected (catalog)':<26}  {'observed':<26}  verdict")
    print("  " + "-" * 92)
    for s in range(64):
        V = FAULTY | (s << 6)
        obs = [hot(footprint(V, REPS), 0.5) for _ in range(PASSES)]
        stable = len(set(obs)) == 1
        merged = obs[0] if stable else frozenset().union(*obs)
        exp = KNOWN_PATTERNS[s]
        match = all(o == exp for o in obs)
        if match:
            verdict = 'OK'; n_ok += 1
        elif stable:
            verdict = f'MISMATCH (catalog says {sorted(exp)})'; n_bad += 1
        else:
            verdict = f'UNSTABLE {[sorted(o) for o in obs]}'; n_unstable += 1
        rows.append([s, hex(V), sorted(exp), [sorted(o) for o in obs], stable, verdict])
        # print only the interesting band + any anomaly
        if s <= 9 or not match:
            print(f"  {s:>3}  {V:#07x}  {str(sorted(exp)):<26}  {str(sorted(merged)):<26}  {verdict}")

    with open(OUT, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['demand_set', 'V', 'expected', 'observed_passes', 'stable', 'verdict'])
        w.writerows(rows)

    print(f"\n=== SUMMARY: {n_ok}/64 match catalog exactly, {n_bad} mismatch, {n_unstable} unstable ===")
    print(f"  full per-set log: {OUT}")
    silent = [s for s in range(8, 64) if KNOWN_PATTERNS[s] == frozenset()
              and all(r[0] != s or r[5] == 'OK' for r in rows)]
    print(f"  demand sets 8..63 expected silent: {len([r for r in rows if r[0] >= 8 and r[5]=='OK'])}/56 verified silent")


if __name__ == '__main__':
    main()
