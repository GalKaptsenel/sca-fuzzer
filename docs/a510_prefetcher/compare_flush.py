#!/usr/bin/env python3
"""EXP1 with vs without the demand-line flush, same inputs.
  flushed   : lit sets == prefetcher-filled only (demand line evicted by us)
  unflushed : lit sets == demand line + prefetcher-filled"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['NORUN'] = '1'
from fast_sweep import run

V = [int(x,16) for x in os.environ.get('VALUES','0x1000,0x1040,0x1080,0x10c0,0x1100,0x1001,0x1020').split(',')]
noflush = dict(run(V, (), 'EXP1-noflush: demand line NOT flushed', flush_demand=False))
flushed = dict(run(V, (), 'EXP1-flushed: demand line flushed',     flush_demand=True))

print("\n=== COMPARISON (bg sets 15/23/31 excluded) ===")
print(f"{'x1':>8} {'demand set':>10} | {'WITHOUT flush (demand+prefetch)':<34} | {'WITH flush (prefetch only)':<28} | delta")
print("-"*118)
for v in V:
    d = (v//64) % 64
    a, b = sorted(noflush[v]), sorted(flushed[v])
    gone = sorted(set(a) - set(b))
    print(f"{v:#08x} {d:>10} | {str(a):<34} | {str(b):<28} | -{gone}")
