#!/usr/bin/env python3
"""
Triage an AArch64 Revizor violation: GENUINE Spectre-style leak vs measurement noise.

Compares the contract executor's speculative (wrong-path) cache lines against the
hardware htrace divergence recorded in the violation report — WITHOUT re-running the
test on hardware (which would train the BPU and hide the leak).

Usage:
    source <your-revizor-venv>/bin/activate
    sudo chmod 777 /dev/executor
    python3 triage_violation.py <path/to/violation-YYMMDD-HHMMSS>

Verdict per violation:
    GENUINE      - every stably-divergent HW bit is a CE-predicted speculative line.
    NOISE        - no stable HW divergence (or no CE speculative divergence).
    INVESTIGATE  - a stable HW-divergent bit is not in the CE set (deeper nesting / CE gap).
"""
import os, sys, re

def _find_repo_root():
    """Repo root = nearest ancestor containing revizor.py (override with $REVIZOR_ROOT)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "revizor.py")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


REPO = os.environ.get("REVIZOR_ROOT") or _find_repo_root()
FREQ_DIVERGENCE_THRESHOLD = 0.3   # |freq_A - freq_B| above this == a stable per-bit divergence


def per_bit_freq(report_text, input_idx):
    """Per-bit set-frequency across ALL htrace patterns of one counterexample input."""
    m = re.search(rf'^Input #{input_idx}\s*$(.*?)Contract trace', report_text, re.M | re.S)
    if not m:
        return None
    total = 0
    count = [0] * 64
    for pattern, c in re.findall(r'([.\^]{60,})\s*\[(\d+)\]', m.group(1)):
        c = int(c)
        total += c
        for pos, ch in enumerate(pattern):
            if ch == '^':
                count[pos] += c
    return [c / total for c in count] if total else None


def main(vdir):
    vdir = os.path.abspath(vdir)   # resolve before chdir
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    from src.config import CONF
    CONF.load('config.yml')
    CONF.contract_execution_clause = ['cond']   # -> ALWAYS_MISPREDICT (explore wrong path)
    from src.isa_loader import InstructionSet
    from src import factory

    isa = InstructionSet('base.json', CONF.instruction_categories)
    gen = factory.get_program_generator(isa, CONF.program_generator_seed)
    ig = factory.get_input_generator(CONF.input_gen_seed)
    ex = factory.get_executor()
    ap = factory.get_asm_parser(gen)
    sandbox_base, _ = ex.read_base_addresses()
    K = (sandbox_base // 64) % 64

    report = open(os.path.join(vdir, 'report.txt')).read()
    cex = list(dict.fromkeys(re.findall(r'^Input #(\d+)\s*$', report, re.M)))[:2]
    if len(cex) < 2:
        print(f"could not parse 2 counterexample inputs from {vdir}")
        return 2

    tc = ap.parse_file(os.path.join(vdir, 'generated.asm'))
    ex.load_test_case(tc)

    def _inp(i):  # saved inputs are now input_NNNN_nzcv_scheme.bin (old: input_NNNN.bin)
        base = os.path.join(vdir, f'input_{int(i):04d}')
        for suffix in ('_nzcv_scheme.bin', '.bin'):
            if os.path.exists(base + suffix):
                return base + suffix
        raise FileNotFoundError(f'no input file for #{i} in {vdir}')
    inputs = ig.load([_inp(i) for i in cex])

    ctr, _, cer, _ = ex.trace_test_case_with_taints(inputs, 5)

    # arch vs speculative cache lines per input
    def split_lines(c):
        arch, spec = set(), set()
        for ite in c:
            if ite.metadata.has_memory_access:
                line = ((ite.metadata.memory_access.effective_address - ite.cpu.gpr[29]) // 64) % 64
                (arch if ite.metadata.speculation_nesting == 0 else spec).add(line)
        return arch, spec
    a0, s0 = split_lines(cer[0])
    a1, s1 = split_lines(cer[1])

    # CE divergent HW bits (string is bit-reversed: pos = 63 - set)
    def to_bits(lines):
        return set(63 - ((l + K) % 64) for l in lines)
    ce_div = to_bits(set(ctr[0].raw) ^ set(ctr[1].raw))

    fa, fb = per_bit_freq(report, cex[0]), per_bit_freq(report, cex[1])
    hw_div = sorted(b for b in range(64) if fa and fb and abs(fa[b] - fb[b]) > FREQ_DIVERGENCE_THRESHOLD)

    print(f"=== {os.path.basename(vdir)}  inputs #{cex[0]} vs #{cex[1]}  (K={K}) ===")
    print(f"  arch lines:  #{cex[0]}={sorted(a0)}  #{cex[1]}={sorted(a1)}  "
          f"{'IDENTICAL' if a0 == a1 else 'DIFFER (not arch-equivalent!)'}")
    print(f"  spec lines:  #{cex[0]}={sorted(s0)}  #{cex[1]}={sorted(s1)}")
    print(f"  CE divergent HW bits = {sorted(ce_div)}")
    print(f"  HW per-bit divergent = {hw_div}")
    for b in hw_div:
        tag = "CE-predicted" if b in ce_div else "NOT in CE"
        print(f"     bit {b}: freq#{cex[0]}={fa[b]:.2f} freq#{cex[1]}={fb[b]:.2f}  ({tag})")

    if not hw_div:
        verdict = "NOISE (no stable HW divergence)"
    elif set(hw_div).issubset(ce_div):
        verdict = "GENUINE (HW divergence explained by CE speculation)"
    else:
        verdict = "INVESTIGATE (HW bit(s) not in CE set: deeper nesting or CE gap)"
    print(f"  => {verdict}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
