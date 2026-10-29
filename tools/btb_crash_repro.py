"""Sweep program seeds through the real NI boost (CE) path to find a seed that reproduces the
kaddr2uaddr out-of-sandbox CE crash, then dump the offending program's asm. Device-free except the
one-time sandbox-base read. Usage: python -I tools/btb_crash_repro.py <config.yml> <n_seeds> <n_inputs>"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONF

cfg, n_seeds, n_inputs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
CONF.load(cfg)
from src.factory import get_fuzzer

fuzzer = get_fuzzer("base.json", "eval_out/btb", "", "")
fuzzer.initialize_modules()

base_seed = 1000
for s in range(base_seed, base_seed + n_seeds):
    fuzzer.generator.set_seed(s)
    fuzzer.input_gen.set_seed(s * 7 + 1)
    tc = fuzzer.generation_function("") if False else fuzzer.generator.create_test_case("eval_out/btb/_s.asm")
    fuzzer.input_gen.n_actors = len(tc.actors)
    inputs = fuzzer.input_gen.generate(n_inputs)
    fuzzer.executor.load_test_case(tc)
    try:
        boosted, ctraces = fuzzer._boost_inputs(inputs, CONF.model_max_nesting)
    except Exception as e:
        msg = str(e)
        if "kaddr2uaddr" in msg or "outside sandbox" in msg or "crashed" in msg:
            print(f"\n*** CRASH at program seed {s}: {msg[:200]}")
            with open("eval_out/btb/_s.asm") as f:
                print(f.read())
            sys.exit(42)
        print(f"seed {s}: other exception: {msg[:120]}")
        continue
    print(f"seed {s}: ok ({len(boosted)} boosted)")
print("no crash found in sweep")
