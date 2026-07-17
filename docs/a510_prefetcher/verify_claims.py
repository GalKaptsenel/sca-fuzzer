#!/usr/bin/env python3
"""Re-establish EVERY claim in SUPERVISOR_SUMMARY.md using single-input batches only.

Batch neighbours contaminate the prefetcher (see RESULTS.md EXP3/EXP4), so every measurement here is ONE
input in its own super-batch. Each point is repeated N_TRIALS times so instability is visible rather than
averaged away.

  A  cross-core           -- A510 (cpu0) vs A715 (cpu4) vs X3 (cpu8)
  B  data-independence    -- fixed address, varying memory CONTENTS
  C  faulty demand sweep  -- sets 0..15 alone (tests the "no prefetch for set >= 8" cutoff)
  D  main demand sweep    -- sets 0..7 alone (does main really behave like faulty?)
  E  page disambiguation  -- for demand set 2, are the prefetched lines in faulty or main?
  F  flush vs no-flush    -- the headline leak claim, alone
"""
import os, sys, subprocess, tempfile
os.chdir('/home/gal_k_1_1998/sca-fuzzer')
sys.path.insert(0, '/home/gal_k_1_1998/sca-fuzzer')
from src.config import CONF
CONF.load('configs/spectre_v1_fr_remote.yml')
from src.isa_loader import InstructionSet
from src import factory

VD       = 'violations_fr/violation-260716-181940'
BG       = (15, 23, 31)          # harness background sets (lit even with no matching load)
REPS     = int(os.environ.get('REPS', '100'))
N_TRIALS = int(os.environ.get('TRIALS', '3'))

isa = InstructionSet('base.json', CONF.instruction_categories)
gen = factory.get_program_generator(isa, CONF.program_generator_seed)
ig  = factory.get_input_generator(CONF.input_gen_seed)
ex  = factory.get_executor()
ap  = factory.get_asm_parser(gen)
ex.load_test_case(ap.parse_file(os.path.join(VD, 'generated.asm')))
BASE = ig.load([os.path.join(VD, 'input_0009.reif')])[0]
conn, sysfs = ex.device._conn, ex.device._cfg.sysfs


def assemble(asm):
    with tempfile.TemporaryDirectory() as d:
        s, o, b = f'{d}/a.s', f'{d}/a.o', f'{d}/a.bin'
        open(s, 'w').write(asm)
        subprocess.run(['as', '-march=armv9-a+memtag', '-o', o, s], check=True)
        subprocess.run(['objcopy', '-O', 'binary', o, b], check=True)
        return open(b, 'rb').read()


def victim(extra_flushes=(), flush_demand=True):
    lines = ['and x1, x1, #0x1fff', 'add x0, x29, x1', 'ldr x3, [x0]']
    if flush_demand:
        lines.append('dc civac, x0')
    for off in extra_flushes:
        lines += [f'movz x2, #{off:#x}', 'add x2, x29, x2', 'dc civac, x2']
    return '\n'.join(lines + ['dsb ish', 'isb', ''])


def measure_alone(V, extra_flushes=(), flush_demand=True, fill=None):
    """ONE input, ONE super-batch. Returns the prefetched sets (demand set and background excluded)."""
    tc_bytes = assemble(victim(extra_flushes, flush_demand))
    _, patched = ex._sandboxed_tc()
    ex._sandboxed_cache = (tc_bytes, patched)

    inp = BASE.copy()
    inp[0]['gpr'][1] = V
    if fill is not None:
        inp[0]['main'][:] = fill
        inp[0]['faulty'][:] = fill
    htraces, _ = ex.trace_test_case([ex.as_executor_input(inp)], REPS)
    raw = htraces[0].raw
    n = len(raw) or 1
    fr = [sum((r >> b) & 1 for r in raw) / n for b in range(64)]
    demand = (V // 64) % 64
    return [b for b in range(64) if fr[b] > 0.5 and b not in BG and b != demand]


def trials(V, **kw):
    return [measure_alone(V, **kw) for _ in range(N_TRIALS)]


def show(label, res):
    same = all(r == res[0] for r in res)
    stable = 'stable' if same else '*** UNSTABLE ***'
    print(f"  {label:<34} {str(res[0]):<26} {stable}")
    if not same:
        for i, r in enumerate(res[1:], 2):
            print(f"  {'  trial ' + str(i):<34} {r}")


def pin(core):
    conn.shell(f'echo {core} > {sysfs}/pin_to_core', privileged=True)


def main():
    print(f"### All measurements: 1 input per super-batch, {REPS} reps, {N_TRIALS} trials.\n")

    print("=== A: CROSS-CORE (demand faulty+0x000, prefetch expected [2,3,4,5,6]) ===")
    for core, name in ((0, 'cpu0 A510 (in-order)'), (4, 'cpu4 A715 (OoO)'), (8, 'cpu8 X3 (OoO)')):
        pin(core)
        show(name, trials(0x1000))
    pin(0)

    print("\n=== B: DATA-INDEPENDENCE (address FIXED at faulty+0x000; memory CONTENTS varied) ===")
    for fill, name in ((0x0000000000000000, 'all zeros'),
                       (0xffffffffffffffff, 'all ones'),
                       (0x5a5a5a5a5a5a5a5a, 'pattern 0x5a..'),
                       (0x0000000000001080, 'value = a sandbox offset')):
        show(name, trials(0x1000, fill=fill))

    print("\n=== C: FAULTY demand sets 0..15 ALONE (does the 'set>=8 never prefetches' cutoff hold?) ===")
    for s in range(16):
        show(f'faulty set {s:<2} (V={0x1000 + s*0x40:#06x})', trials(0x1000 + s * 0x40))

    print("\n=== D: MAIN demand sets 0..7 ALONE (contaminated batches said 'main never streams') ===")
    for s in range(8):
        show(f'main set {s:<2} (V={s*0x40:#06x})', trials(s * 0x40))

    print("\n=== E: PAGE DISAMBIGUATION (demand faulty+0x080 = set 2 -> prefetch [18,26,34,42,50]) ===")
    print("    set 18 == faulty+0x480 OR main+0x480 (the two pages alias onto the same 64 sets).")
    show('no extra flush',                 trials(0x1080))
    show('flush faulty+0x480 (set 18)',    trials(0x1080, extra_flushes=(0x1480,)))
    show('flush main+0x480   (set 18)',    trials(0x1080, extra_flushes=(0x0480,)))

    print("\n=== F: FLUSH vs NO-FLUSH, ALONE (the headline claim) ===")
    for V in (0x1000, 0x1080, 0x10c0):
        d = (V // 64) % 64
        print(f"  --- V={V:#06x} (demand set {d})")
        show('  without dc civac', trials(V, flush_demand=False))
        show('  with dc civac',    trials(V, flush_demand=True))


if __name__ == '__main__':
    main()
