#!/usr/bin/env python3
"""Same probe, run on the PRE-REFACTOR harness (git 92b0b61^): P+P uses a long unrolled STATIC chain of
loads (no rbit loop), and F+R is the ORIGINAL 2-page (main+faulty) walk.

Two questions at once:
  1. Does an independent, structurally different probe (old P+P) see the same prefetch footprint?
  2. Under the ORIGINAL 2-page F+R, measured ALONE, does main behave like faulty? (the confound:
     every earlier "main == faulty" result used the 4-region harness AND single-input batches.)

P+P semantics are INVERTED vs F+R: prime the eviction region, run the victim, probe -> a set is lit when
the victim EVICTED it. A prefetch into set b therefore lights b, same as an F+R hit does.
"""
import os, sys
os.chdir('/home/gal_k_1_1998/sca-fuzzer')
sys.path.insert(0, '/home/gal_k_1_1998/sca-fuzzer')
MODE = os.environ.get('MODE', 'F+R')
from src.config import CONF
CONF.load('configs/spectre_v1_fr_remote.yml')
CONF.executor_mode = MODE
import verify_claims as V   # builds the executor with the mode above

print(f"### PRE-REFACTOR harness, measurement_mode={MODE}, 1 input/batch, "
      f"{V.REPS} reps, {V.N_TRIALS} trials\n")
print("=== faulty page (demand line flushed by the victim) ===")
for s in (0, 2, 3, 8):
    V.show(f'faulty set {s:<2} (V={0x1000 + s*0x40:#06x})', V.trials(0x1000 + s * 0x40))
print("\n=== main page -- ALONE, on the ORIGINAL 2-page harness ===")
for s in (0, 2, 3, 8):
    V.show(f'main set {s:<2} (V={s*0x40:#06x})', V.trials(s * 0x40))
