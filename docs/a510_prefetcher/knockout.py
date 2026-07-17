#!/usr/bin/env python3
"""EXP2: demand-load at faulty+0x1000 prefetches sets 2-6. Flush SOME of those prefetched lines
(dc civac) and confirm exactly those sets go dark -- proving the lines are prefetcher-filled and
that F+R is reading real cache residency, not a walk artifact."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_sweep import run

# demand x1=0x1000 (set 0) -> prefetches sets 2..6 == offsets 0x1080,0x10c0,0x1100,0x1140,0x1180
run([0x1000], (),                         'EXP2a: knockout NONE (baseline)')
run([0x1000], (0x1080,),                  'EXP2b: knockout set 2      (flush 0x1080)')
run([0x1000], (0x1080, 0x10c0),           'EXP2c: knockout sets 2,3   (flush 0x1080,0x10c0)')
run([0x1000], (0x1080, 0x10c0, 0x1100),   'EXP2d: knockout sets 2,3,4 (flush 0x1080,0x10c0,0x1100)')
