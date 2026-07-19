#!/usr/bin/env python3
"""PoC: a covert channel over the Cortex-A510 L1 hardware prefetcher.

The victim is experiment 1 from the morning -- one load, then a flush of the *same* register:

    and x1, x1, #0x1fff        ; x1 carries the hidden value
    add x0, x29, x1            ; demand address = sandbox_base + offset
    ldr x3, [x0]              ; the single demand load
    dc  civac, x0             ; flush the demand line (same register)
    dsb ish ; isb

`dc civac, x0` invalidates ONLY the demand line, by VA, exactly as documented. It does NOT touch the
extra lines the HW prefetcher pulled in as a side effect of the load. So after the victim runs, the
cache footprint is entirely the prefetcher's -- and that footprint is a deterministic function of the
demand address's offset bits [11:6] (the demand SET). That is the channel.

SYMBOL ALPHABET
---------------
The transmitter chooses a demand SET s in [0..63] by setting offset = FAULTY | (s << 6). The A510
prefetcher's footprint (background sets excluded) is a characterized function of s -- the full catalog
is KNOWN_PATTERNS below (measured this session, see RESULTS.md):

    s = 0        -> [2,3,4,5,6]                 (page-start contiguous mode; unique)
    s = 1        -> [17,25,33,41,49]            (stride-8 mode; UNSTABLE across runs)
    s = 2..7     -> [s+16,s+24,s+32,s+40,s+48]  (stride-8 mode; s=7 shows [39,47,55], 23/31 under bg)
    s >= 8       -> []                           (past the ~512B trigger zone -> silence)

So the distinguishable footprints are demand sets 0..7 plus "silence" (any s>=8) = up to 9 symbols.
Set 1 is unstable, so the RELIABLE alphabet is 8 (sets 0,2,3,4,5,6,7 + silence). POC_ALPHABET selects:
    full     : sets 0..7 + silence            (9 symbols, includes the flaky set 1)
    reliable : sets 0,2,3,4,5,6,7 + silence    (8 symbols, drops set 1)   [default]
    compact  : the original 5-symbol scheme    (low/bit6/bit7/bit8/bit9)

NOVELTY REPORTING
-----------------
The decoder is aware of the full KNOWN_PATTERNS catalog. A measured footprint is decoded to the nearest
calibrated symbol ONLY if it is within POC_REJECT (default 3) of that symbol's template; otherwise it is
reported as UNKNOWN, together with the nearest catalog pattern (which demand set it most resembles) so a
novel/unexpected prefetch behaviour is surfaced rather than silently misclassified.

CAPACITY
--------
Each transmitted input carries log2(reliable_alphabet) bits. With A symbols reliably resolved, k inputs
retrieve any integer in [0, A**k). The run prints the measured reliable alphabet and the resulting
bits/input and numbers/k-inputs.

Env knobs:  POC_N (default 10000)  REPS (default 40)  CALIB_REPS (default 100)  POC_SEED (default 1)
            POC_ALPHABET (reliable|full|compact)  POC_REJECT (default 3)
Results are streamed to docs/a510_prefetcher/poc_results.csv so a crash never loses the run.
"""
import os, sys, subprocess, tempfile, csv, time, math

os.chdir('/home/gal_k_1_1998/sca-fuzzer')
sys.path.insert(0, '/home/gal_k_1_1998/sca-fuzzer')
from src.config import CONF
CONF.load('configs/spectre_v1_fr_remote.yml')
from src.isa_loader import InstructionSet
from src import factory

VD         = 'violations_fr/violation-260716-181940'
BG         = frozenset((15, 23, 31))          # harness background sets (lit regardless of the load)
FAULTY     = 0x1000
REPS       = int(os.environ.get('REPS', '40'))
CALIB_REPS = int(os.environ.get('CALIB_REPS', '100'))
N          = int(os.environ.get('POC_N', '10000'))
SEED       = int(os.environ.get('POC_SEED', '1'))
ALPHABET   = os.environ.get('POC_ALPHABET', 'reliable')
REJECT     = int(os.environ.get('POC_REJECT', '3'))
OUT_CSV    = 'docs/a510_prefetcher/poc_results.csv'


def characterized_footprint(demand_set):
    """The A510 prefetch footprint for a faulty-page demand set, background excluded (the catalog)."""
    if demand_set == 0:
        base = {2, 3, 4, 5, 6}
    elif 1 <= demand_set <= 7:
        base = {demand_set + 16, demand_set + 24, demand_set + 32, demand_set + 40, demand_set + 48}
    else:
        base = set()
    return frozenset(base - BG)


# full catalog the decoder is "aware of": every demand set 0..63 (8+ all collapse to silence).
KNOWN_PATTERNS = {s: characterized_footprint(s) for s in range(64)}


def demand_set_symbols():
    """symbol name -> list of x1 offset values that all encode that symbol."""
    if ALPHABET == 'compact':
        return {
            'low':  [FAULTY | (1 << 3), FAULTY | (1 << 4), FAULTY | (1 << 5)],
            'bit6': [FAULTY | (1 << 6)], 'bit7': [FAULTY | (1 << 7)],
            'bit8': [FAULTY | (1 << 8)], 'bit9': [FAULTY | (1 << 9)],
        }
    sets = range(8) if ALPHABET == 'full' else [0, 2, 3, 4, 5, 6, 7]
    syms = {f'set{s}': [FAULTY | (s << 6)] for s in sets}
    syms['silence'] = [FAULTY | (8 << 6), FAULTY | (20 << 6)]   # any demand set >= 8
    return syms


SYMBOLS   = demand_set_symbols()
SYM_NAMES = list(SYMBOLS)


def build_rng(seed):
    import random
    return random.Random(seed)


isa = InstructionSet('base.json', CONF.instruction_categories)
gen = factory.get_program_generator(isa, CONF.program_generator_seed)
ig  = factory.get_input_generator(CONF.input_gen_seed)
ex  = factory.get_executor()
ap  = factory.get_asm_parser(gen)
ex.load_test_case(ap.parse_file(os.path.join(VD, 'generated.asm')))
BASE_INPUT = ig.load([os.path.join(VD, 'input_0009.reif')])[0]
conn, sysfs = ex.device._conn, ex.device._cfg.sysfs


def assemble(asm):
    with tempfile.TemporaryDirectory() as d:
        s, o, b = f'{d}/a.s', f'{d}/a.o', f'{d}/a.bin'
        open(s, 'w').write(asm)
        subprocess.run(['as', '-march=armv9-a+memtag', '-o', o, s], check=True)
        subprocess.run(['objcopy', '-O', 'binary', o, b], check=True)
        return open(b, 'rb').read()


def victim():
    return '\n'.join([
        'and x1, x1, #0x1fff',
        'add x0, x29, x1',
        'ldr x3, [x0]',
        'dc civac, x0',
        'dsb ish', 'isb', ''])


VICTIM_BYTES = assemble(victim())
_, _PATCHED = ex._sandboxed_tc()


def footprint(V, reps):
    """ONE input, ONE super-batch. Return {set: frequency} for every htrace bit, background excluded."""
    ex._sandboxed_cache = (VICTIM_BYTES, _PATCHED)
    inp = BASE_INPUT.copy()
    inp[0]['gpr'][1] = V
    htraces, _ = ex.trace_test_case([ex.as_executor_input(inp)], reps)
    raw = htraces[0].raw
    n = len(raw) or 1
    return {b: sum((r >> b) & 1 for r in raw) / n for b in range(64) if b not in BG}


def hot(freqs, thresh):
    return frozenset(b for b, f in freqs.items() if f > thresh)


def pin(core):
    conn.shell(f'echo {core} > {sysfs}/pin_to_core', privileged=True)


def calibrate():
    print(f"=== CALIBRATION (alphabet={ALPHABET}, {CALIB_REPS} reps/symbol) ===")
    templates = {}
    for name in SYM_NAMES:
        sigs = {V: hot(footprint(V, CALIB_REPS), 0.5) for V in SYMBOLS[name]}
        union = frozenset().union(*sigs.values())
        agreed = frozenset(b for b in union if all(b in s for s in sigs.values()))
        templates[name] = agreed
        dset = (next(iter(SYMBOLS[name])) & 0x1fff) // 64
        collapse = 'OK' if len(set(sigs.values())) == 1 else '*** bits disagree ***'
        exp = sorted(KNOWN_PATTERNS[dset])
        match = 'matches catalog' if agreed == KNOWN_PATTERNS[dset] else f'!= catalog {exp}'
        print(f"  {name:<8} demand_set={dset:<2} footprint={sorted(agreed)!s:<24} {collapse:<20} {match}")

    print("\n  pairwise footprint distance (symmetric-difference size; 0 == indistinguishable):")
    print("        " + "".join(f"{n:>9}" for n in SYM_NAMES))
    collisions = []
    for a in SYM_NAMES:
        row = []
        for b in SYM_NAMES:
            d = len(templates[a] ^ templates[b])
            row.append(d)
            if a < b and d == 0:
                collisions.append((a, b))
        print(f"  {a:<8}" + "".join(f"{d:>9}" for d in row))
    if collisions:
        print(f"  WARNING: indistinguishable symbol pairs {collisions} -- channel cannot resolve these.")
    return templates


def nearest_known(sig):
    """Best-matching characterized demand-set pattern for an unrecognized footprint."""
    s = min(range(64), key=lambda k: (len(sig ^ KNOWN_PATTERNS[k]), k))
    return s, len(sig ^ KNOWN_PATTERNS[s])


def decode(sig, templates):
    """Nearest calibrated symbol within POC_REJECT; else 'UNKNOWN' (novel/unexpected footprint)."""
    name = min(SYM_NAMES, key=lambda n: (len(sig ^ templates[n]), SYM_NAMES.index(n)))
    if len(sig ^ templates[name]) > REJECT:
        return 'UNKNOWN'
    return name


def run_game(templates):
    rng = build_rng(SEED)
    plan = [rng.choice(SYM_NAMES) for _ in range(N)]
    labels = SYM_NAMES + ['UNKNOWN']
    print(f"\n=== TRANSMIT + DECODE ({N} inputs, {REPS} reps each, seed={SEED}, reject>{REJECT}) ===")
    correct = 0
    unknowns = []
    confusion = {a: {b: 0 for b in labels} for a in SYM_NAMES}
    t0 = time.time()
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['i', 'sent', 'V', 'footprint', 'decoded', 'ok', 'nearest_known_set', 'nk_dist'])
        for i, sent in enumerate(plan):
            V = rng.choice(SYMBOLS[sent])
            sig = hot(footprint(V, REPS), 0.5)
            got = decode(sig, templates)
            ok = (got == sent)
            correct += ok
            confusion[sent][got] += 1
            nks, nkd = nearest_known(sig)
            if got == 'UNKNOWN':
                unknowns.append((i, sent, V, sorted(sig), nks, nkd))
                print(f"  [NOVEL @#{i}] sent={sent} V={V:#06x} footprint={sorted(sig)} "
                      f"-> matches no known symbol; nearest catalog = demand set {nks} (dist {nkd})")
            w.writerow([i, sent, hex(V), sorted(sig), got, int(ok), nks, nkd])
            if (i + 1) % 500 == 0 or i + 1 == N:
                rate = (i + 1) / (time.time() - t0)
                fh.flush()
                print(f"  {i+1:>6}/{N}  acc={correct/(i+1):6.3f}  unknown={len(unknowns)}  "
                      f"{rate:5.1f} inp/s  ETA {(N-i-1)/rate/60 if rate else 0:5.1f} min")

    print(f"\n=== RESULT: {correct}/{N} recovered  ({100*correct/N:.2f}%),  {len(unknowns)} reported NOVEL ===")
    print("  confusion (rows = sent, cols = decoded):")
    print("        " + "".join(f"{n:>9}" for n in labels))
    reliable = []
    for a in SYM_NAMES:
        acc = confusion[a][a] / max(sum(confusion[a].values()), 1)
        if acc >= 0.99:
            reliable.append(a)
        print(f"  {a:<8}" + "".join(f"{confusion[a][b]:>9}" for b in labels) + f"   acc={acc:.3f}")

    A = len(reliable)
    print(f"\n=== CAPACITY ===")
    print(f"  reliable symbols (per-symbol acc >= 0.99): {A}  {reliable}")
    if A >= 2:
        bits = math.log2(A)
        print(f"  bits per transmitted input: log2({A}) = {bits:.2f}")
        print(f"  k inputs retrieve any integer in [0, {A}**k):")
        for k in (1, 2, 4, 8):
            print(f"    k={k:<2} -> {A**k:>20,} distinct values  ({k*bits:.1f} bits)")
    print(f"  full per-input log: {OUT_CSV}")
    return reliable


def main():
    pin(0)  # the channel lives only on the A510 (cpu0); the OoO cores show nothing.
    templates = calibrate()
    if all(len(t) == 0 for t in templates.values()):
        print("\nAll footprints empty -- prefetcher silent. Is the phone on cpu0 and the victim loaded?")
        return
    run_game(templates)


if __name__ == '__main__':
    main()
