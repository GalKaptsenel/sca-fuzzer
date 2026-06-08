#!/usr/bin/env python3
"""Conscious CE cross-validation of a Revizor violation.

Dumps, for BOTH counterexample inputs and BOTH paths, every memory access with:
disassembly (from the raw encoding), cache set, and the loaded/stored value
(memory_access.before). Each line carries a `hand==CE` check (re-derives the set
from the effective address). Architectural sets should be IDENTICAL between the two
inputs; speculative sets should diverge at the gadget (a load whose address is a
previously-loaded value), and those divergent sets must equal the HW-divergent bits.

Usage:
    source <your-revizor-venv>/bin/activate
    sudo chmod 777 /dev/executor
    python3 manual_emulate.py <path/to/violation-YYMMDD-HHMMSS>
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


def main(VD):
    VD = os.path.abspath(VD)   # resolve before chdir
    os.chdir(REPO); sys.path.insert(0, REPO)
    from src.config import CONF; CONF.load('config.yml')
    CONF.contract_execution_clause = ['cond']          # ALWAYS_MISPREDICT
    from src.isa_loader import InstructionSet
    from src import factory
    from src.aarch64.aarch64_disasm import disassemble_instruction
    isa = InstructionSet('base.json', CONF.instruction_categories)
    gen = factory.get_program_generator(isa, CONF.program_generator_seed)
    ig = factory.get_input_generator(CONF.input_gen_seed); ex = factory.get_executor()
    ap = factory.get_asm_parser(gen)
    report = open(os.path.join(VD, 'report.txt')).read()
    cex = list(dict.fromkeys(re.findall(r'^Input #(\d+)\s*$', report, re.M)))[:2]
    tc = ap.parse_file(os.path.join(VD, 'generated.asm')); ex.load_test_case(tc)
    def _inp(i):  # saved inputs are input_NNNN_nzcv_scheme.bin (old: input_NNNN.bin)
        b = os.path.join(VD, f'input_{int(i):04d}')
        return next(b + s for s in ('_nzcv_scheme.bin', '.bin') if os.path.exists(b + s))
    inp = ig.load([_inp(i) for i in cex])
    _, _, cer, _ = ex.trace_test_case_with_taints(inp, 5)

    def collect(c):
        arch, spec, seen = [], [], set()
        for ite in c:
            m = ite.metadata
            if not m.has_memory_access:
                continue
            ea = m.memory_access.effective_address; off = (ea - ite.cpu.gpr[29]) & 0xffffffff
            cs = (off // 64) % 64
            dis = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc)
            k = (m.speculation_nesting, dis, off)
            if k in seen:
                continue
            seen.add(k)
            rec = (dis, m.memory_access.is_write, off, cs, m.memory_access.before & 0xffff)
            (arch if m.speculation_nesting == 0 else spec).append(rec)
        return arch, spec

    a0, s0 = collect(cer[0]); a1, s1 = collect(cer[1])

    def show(title, L):
        print(f"\n{title}")
        for dis, w, off, cs, bef in L:
            hand = (off // 64) % 64
            ok = 'OK' if hand == cs else 'MISMATCH'
            print(f"   {'ST' if w else 'LD'} off=0x{off & 0xffff:04x} set={cs:<2} "
                  f"(hand {hand} {ok})  val=0x{bef:04x}  {dis}")

    print("=" * 70 + f"\nARCHITECTURAL PATH (nest 0)  inputs #{cex[0]} vs #{cex[1]}")
    show(f"#{cex[0]} arch:", a0); show(f"#{cex[1]} arch:", a1)
    print(f"\n  arch sets #{cex[0]}={[r[3] for r in a0]}  #{cex[1]}={[r[3] for r in a1]}  "
          f"{'IDENTICAL (contract-equivalent)' if [r[3] for r in a0] == [r[3] for r in a1] else 'DIFFER -> NOT a real violation'}")
    print("\n" + "=" * 70 + "\nSPECULATIVE PATH (nest>0, mispredicted)")
    show(f"#{cex[0]} spec:", s0); show(f"#{cex[1]} spec:", s1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
