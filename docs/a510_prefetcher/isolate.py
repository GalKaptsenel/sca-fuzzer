#!/usr/bin/env python3
"""Is the prefetch result contaminated by batch neighbours? Measure the SAME address alone in its own
super-batch vs. inside a batch, repeatedly. If alone-vs-grouped differ, batch order is a confound."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['NORUN'] = '1'
from fast_sweep import run

for trial in (1, 2, 3):
    print(f"\n#################### TRIAL {trial}: each address ALONE in its own batch")
    for V in (0x0080, 0x1080, 0x0000, 0x1000):
        run([V], (), f'alone V={V:#06x}')
print("\n#################### SAME addresses, all together in ONE batch")
run([0x0080, 0x1080, 0x0000, 0x1000], (), 'grouped')
