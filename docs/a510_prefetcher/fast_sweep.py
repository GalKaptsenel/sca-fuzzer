#!/usr/bin/env python3
"""A510 prefetcher probe, BATCHED (one super-batch => one round trip; ~100x faster than per-rep adb).

TC (hand-assembled, has `dc civac` which the fuzzer ISA lacks):
    and x1, x1, #0x1fff ; add x0, x29, x1 ; ldr x3, [x0] ; dc civac, x0 ; dsb ish ; isb
x1 comes ONLY from each .reif input, so the demand address is input-driven.
The demand line is flushed, so any lit set == prefetcher-filled.

Usage: fast_sweep.py [reps]        (env VALUES="0x1000,0x1040,..." to override the sweep points)
"""
import os, sys, subprocess, tempfile
sys.path.insert(0, '/home/gal_k_1_1998/sca-fuzzer')
from src.config import CONF
CONF.load('/home/gal_k_1_1998/sca-fuzzer/configs/spectre_v1_fr_remote.yml')
from src.isa_loader import InstructionSet
from src import factory
from src.aarch64.aarch64_executor_input_encoder import ExecutorInput

os.chdir('/home/gal_k_1_1998/sca-fuzzer')
VD   = '/home/gal_k_1_1998/sca-fuzzer/violations_fr/violation-260716-181940'
BG   = (15, 23, 31)                       # harness background sets (present with no matching load)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
VALUES = [int(v, 16) for v in os.environ.get(
    'VALUES', '0x1000,0x1040,0x1080,0x10c0,0x1100,0x1001,0x1020').split(',')]

def assemble(asm: str) -> bytes:
    """Assemble raw asm -> machine code (native aarch64 as/objcopy)."""
    with tempfile.TemporaryDirectory() as d:
        s, o, b = f'{d}/a.s', f'{d}/a.o', f'{d}/a.bin'
        open(s, 'w').write(asm)
        subprocess.run(['as', '-march=armv9-a+memtag', '-o', o, s], check=True)
        subprocess.run(['objcopy', '-O', 'binary', o, b], check=True)
        return open(b, 'rb').read()

def probe_asm(extra_flushes=(), flush_demand=True) -> str:
    lines = ['and x1, x1, #0x1fff', 'add x0, x29, x1', 'ldr x3, [x0]']
    if flush_demand:
        lines.append('dc civac, x0')
    for off in extra_flushes:                       # knockout: flush specific prefetched lines
        lines += [f'movz x2, #{off:#x}', 'add x2, x29, x2', 'dc civac, x2']
    return '\n'.join(lines + ['dsb ish', 'isb', ''])

isa = InstructionSet('base.json', CONF.instruction_categories)
gen = factory.get_program_generator(isa, CONF.program_generator_seed)
ig  = factory.get_input_generator(CONF.input_gen_seed)
ex  = factory.get_executor()
ap  = factory.get_asm_parser(gen)

# Load any TC to populate executor state, then swap in our hand-assembled bytes.
ex.load_test_case(ap.parse_file(os.path.join(VD, 'generated.asm')))

def run(values, extra_flushes, tag, flush_demand=True):
    tc_bytes = assemble(probe_asm(extra_flushes, flush_demand))
    _, patched = ex._sandboxed_tc()
    ex._sandboxed_cache = (tc_bytes, patched)       # <-- the seam: our bytes, batched transport

    base = ig.load([os.path.join(VD, 'input_0009.reif')])[0]
    inputs = []
    for V in values:
        inp = base.copy()
        inp[0]['gpr'][1] = V                        # x1 = V; all else identical
        inputs.append(inp)
    exec_inputs = [ex.as_executor_input(i) for i in inputs]
    htraces, _ = ex.trace_test_case(exec_inputs, REPS)   # ONE super-batch

    out = []
    print(f"\n=== {tag}  ({REPS} reps, {len(values)} inputs, 1 round-trip) ===")
    for V, h in zip(values, htraces):
        raw = h.raw; n = len(raw) or 1
        fr  = [sum((r >> b) & 1 for r in raw)/n for b in range(64)]
        pf  = [b for b in range(64) if fr[b] > 0.5 and b not in BG]
        print(f"  x1={V:#06x}  bits[11:6]={((V>>6)&0x3f):#04x}  demand-set={(V//64)%64:2}  prefetched={pf}")
        out.append((V, pf))
    return out

if __name__ == '__main__' and os.environ.get('NORUN') != '1':
    run(VALUES, (), 'EXP1: input-driven demand address, demand line flushed')
