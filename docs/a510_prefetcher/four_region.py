import os, sys
sys.path.insert(0, os.getcwd())
from src.config import CONF
CONF.load('configs/spectre_v1_fr_remote.yml')
from src.isa_loader import InstructionSet
from src import factory

VD = "violations_fr/violation-260716-181940"
N_REPS = 300
isa = InstructionSet('base.json', CONF.instruction_categories)
gen = factory.get_program_generator(isa, CONF.program_generator_seed)
ig  = factory.get_input_generator(CONF.input_gen_seed)
ex  = factory.get_executor()
ap  = factory.get_asm_parser(gen)
ex.load_test_case(ap.parse_file(os.path.join(VD, 'generated.asm')))
inputs = ig.load([os.path.join(VD, f'input_{i:04d}.reif') for i in range(29)])
exec_inputs = [ex.as_executor_input(i) for i in inputs]

def freq(ht, b):
    raw = ht.raw; n = len(raw) or 1
    return sum((r >> b) & 1 for r in raw) / n

ht, _ = ex.trace_test_case(exec_inputs, N_REPS)
print(f"=== 4-region harness (flush+reload lower->main->faulty->upper; record main+faulty) {N_REPS} reps ===")
for idx, tag in ((9, "#9  (ldp->faulty)"), (28, "#28 (ldp->main)  ")):
    h = ht[idx]
    blk = sum(freq(h, b) for b in (2,3,4,5,6))/5
    lit = [b for b in range(64) if freq(h, b) > 0.5]
    print(f"  {tag}: block2-6={blk:.2f}  lit={lit}")
