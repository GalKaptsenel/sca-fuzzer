#!/usr/bin/env python3
"""The decisive test for violation-260716-181940: measure #9 and #28 each ALONE in its own
super-batch (identical, empty predecessor context) and compare against the grouped 29-input batch
the violation was originally found in.

If #9 and #28 agree when alone but differ when grouped => the "violation" is batch/predecessor
contamination of the L1 prefetcher, not an input-dependent leak. See RESULTS.md EXP3.
"""
import os, sys
os.chdir('/home/gal_k_1_1998/sca-fuzzer')
sys.path.insert(0, '/home/gal_k_1_1998/sca-fuzzer')
from src.config import CONF
CONF.load('configs/spectre_v1_fr_remote.yml')
from src.isa_loader import InstructionSet
from src import factory

VD    = 'violations_fr/violation-260716-181940'
REPS  = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BLOCK = (2, 3, 4, 5, 6)

isa = InstructionSet('base.json', CONF.instruction_categories)
gen = factory.get_program_generator(isa, CONF.program_generator_seed)
ig  = factory.get_input_generator(CONF.input_gen_seed)
ex  = factory.get_executor()
ap  = factory.get_asm_parser(gen)
ex.load_test_case(ap.parse_file(os.path.join(VD, 'generated.asm')))
all_inputs = ig.load([os.path.join(VD, f'input_{i:04d}.reif') for i in range(29)])

def stats(h):
    raw = h.raw; n = len(raw) or 1
    fr  = [sum((r >> b) & 1 for r in raw)/n for b in range(64)]
    return sum(fr[b] for b in BLOCK)/len(BLOCK), [b for b in range(64) if fr[b] > 0.5]

def measure(indices, tag):
    exec_inputs = [ex.as_executor_input(all_inputs[i]) for i in indices]
    htraces, _ = ex.trace_test_case(exec_inputs, REPS)
    print(f"\n=== {tag} ({REPS} reps, batch of {len(indices)}) ===")
    for i, h in zip(indices, htraces):
        blk, lit = stats(h)
        print(f"  #{i:<3} block2-6={blk:.2f}  lit={lit}")
    return {i: stats(h) for i, h in zip(indices, htraces)}

for trial in (1, 2, 3):
    print(f"\n################ TRIAL {trial}")
    a9  = measure([9],  'ALONE #9')
    a28 = measure([28], 'ALONE #28')
    print(f"  -> alone: #9 block={a9[9][0]:.2f}  #28 block={a28[28][0]:.2f}  "
          f"same-lit={a9[9][1] == a28[28][1]}")

g = measure(list(range(29)), 'GROUPED (original 29-input batch, as the violation was found)')
print(f"\n  -> grouped: #9 block={g[9][0]:.2f}  #28 block={g[28][0]:.2f}  same-lit={g[9][1] == g[28][1]}")
