"""Low-level correctness check of generated sealed test cases: PAC, MTE, and PAC+MTE.

  PAC:  per AUT* slot, the seal's sig == ground-truth HW pac_sign top16 AND sign->auth round-trips;
        the enacted code runs FPAC-free.
  SPEC options: across many decoy variants, each SPECULATIVE PAC slot is rendered as
        correct-sig+auth / wrong-sig+auth / strip(xpac) — the NI decoy diversity.
  MTE:  model-level (this box has no MTE HW): each slot renders as ADDG(delta)/NOP, deltas are
        deterministic, and speculative slots take diverse deltas across variants.

Usage: verify_seal.py <config.yml> <pac|mte|both> <n_cases>
"""
import os, sys, random, re, collections
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(_ROOT); sys.path.insert(0, _ROOT)

CFG = sys.argv[1] if len(sys.argv) > 1 else "config_pac.yml"
KIND = sys.argv[2] if len(sys.argv) > 2 else "pac"
NCASES = int(sys.argv[3]) if len(sys.argv) > 3 else 10
NVARIANTS = 40

from src.config import CONF
CONF.load(CFG)
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_executor import Aarch64NonInterferenceExecutor
from src.aarch64.aarch64_kernel import PacKeys
from src.aarch64.aarch64_relocations import apply_relocations
from src.aarch64.aarch64_disasm import disassemble_instruction
from src.aarch64.seal.pac import _AUTH_TO_PAC, _AUTH_TO_XPAC, _read_reg
from src import factory

isa = InstructionSet("base.json", CONF.instruction_categories)
gen = Aarch64RandomGenerator(isa, random.randrange(1 << 32))
ex = Aarch64NonInterferenceExecutor(gen)
PAC_KEYS = ex._pac_keys   # the executor's campaign keys — the seal signs under these
igen = factory.get_input_generator(random.randrange(1 << 32))
LE = ex.local_executor
os.makedirs("/tmp/_verify_seal", exist_ok=True)


def find_sealable():
    for _ in range(300):
        try:
            tc = gen.create_test_case("/tmp/_verify_seal/t.asm", disable_assembler=True)
        except Exception:
            continue
        ex.load_test_case(tc)
        pac = getattr(ex._sealed, "_pac", [])
        mte = getattr(ex._sealed, "_mte", [])
        if (KIND in ("pac", "both") and pac) and (KIND != "both" or mte):
            return tc
        if KIND == "mte" and mte:
            return tc
    return None


def read_ptr_ctx(s, cer, layout):
    xpac = next(i for i in s.slot_insts if i.name.lower() in ("xpaci", "xpacd"))
    xpac_off, base = layout.instruction_address[xpac], cer[0].cpu.pc
    vreg = s.committed_inst.operands[0].value
    creg = s.committed_inst.operands[1].value if len(s.committed_inst.operands) > 1 else None
    best = None
    for ite in cer:
        if ite.cpu.pc - base != xpac_off:
            continue
        ptr = _read_reg(ite.cpu, vreg)
        cval = _read_reg(ite.cpu, creg) if creg is not None else 0
        if best is None or ite.metadata.speculation_nesting == 0:
            best = (ptr, cval, ite.metadata.speculation_nesting)
    return best


def addg_delta(enc):
    d = disassemble_instruction(enc, 0) or ""
    if d.startswith("nop"):
        return 0
    m = re.findall(r"#(\d+)\b", d)   # addg x1, x1, #0, #5 -> last imm is the tag delta
    return int(m[-1]) if m else None


pac_ok = pac_tot = 0
spec_tally = collections.Counter()
mte_tally = collections.Counter()
spec_slots_seen = 0
mte_slots = mte_spec_seen = 0
ran_ok = ran_fail = 0

for case in range(NCASES):
    tc = find_sealable()
    if tc is None:
        print(f"[{KIND}] no sealable test case for this config"); break
    inp = igen.generate(1)[0]
    resolved = ex._resolve(inp)
    entries = {id(e.sealing): e for e in resolved._entries}

    # ---- PAC per-slot HW correctness ----
    if KIND in ("pac", "both"):
        cer = list(ex._seal_trace(ex._sealed._tc, inp))
        for s in ex._sealed._pac:
            e = entries.get(id(s))
            rc = read_ptr_ctx(s, cer, ex._sealed._layout)
            if e is None or e.value is None or rc is None:
                continue
            ptr, ctx, _ = rc
            auth_mn = s.committed_inst.name.lower()
            signed = LE.pac_sign(ptr, ctx, _AUTH_TO_PAC[auth_mn], PAC_KEYS)
            pac_tot += 1
            pac_ok += (((signed >> 48) & 0xFFFF) == e.value) and \
                (LE.pac_auth(signed, ctx, auth_mn, PAC_KEYS) == ptr)

        # ---- speculative NI-options distribution over many decoy variants ----
        for s in ex._sealed._pac:
            e = entries.get(id(s))
            if e is None or not e.speculative or not e.alts:
                continue
            offs = resolved._offsets.get(id(s))
            if not offs or len(offs) < 2:
                continue
            spec_slots_seen += 1
            for v in range(NVARIANTS):
                relocs = {r.offset: r.value for r in resolved.decoy(random.Random(v * 2654435761))}
                sig, second = None, "auth"
                for off in offs:                       # find the MOVK + the auth/xpac by mnemonic
                    d = disassemble_instruction(relocs.get(off, 0), 0) or ""
                    mn0 = d.split(" ")[0].lower()
                    if mn0 == "movk":
                        m = re.search(r"#(0x[0-9a-f]+|\d+)", d)  # first imm = value ('lsl #48' is 2nd)
                        sig = int(m.group(1), 0) if m else sig
                    elif mn0.startswith("xpac"):
                        second = "strip"
                which = "correct" if sig == e.value else ("wrong" if sig in e.alts else "other")
                spec_tally[f"{which}+{second}"] += 1

    # ---- MTE structure + speculative decoy diversity (model-level; no MTE HW on this box) ----
    if KIND in ("mte", "both"):
        for s in ex._sealed._mte:
            e = entries.get(id(s))
            offs = resolved._offsets.get(id(s))
            if offs is None:
                continue
            mte_slots += 1
            if e is None or e.value is None or not e.speculative or not e.alts:
                continue
            mte_spec_seen += 1
            correct = e.value % 16
            alts = {a % 16 for a in e.alts}
            for v in range(NVARIANTS):
                relocs = {r.offset: r.value for r in resolved.decoy(random.Random(v * 2654435761))}
                d = addg_delta(relocs.get(offs[0], 0))
                cls = "correct" if d == correct else ("wrong" if d in alts else "other")
                mte_tally[f"{cls}{'(nop)' if d == 0 else ''}"] += 1

    # ---- run enacted genuine on the executor (PAC only; MTE needs MTE HW) ----
    if KIND == "pac":
        LE.discard_all_inputs()
        try:
            ex.trace_test_case([ex.as_executor_input(inp)], 1)
            ran_ok += 1
        except Exception:
            ran_fail += 1

print(f"\n==== {KIND.upper()}  config={CFG}  cases={NCASES} ====")
if KIND in ("pac", "both"):
    print(f"PAC slots HW-correct: {pac_ok}/{pac_tot}")
    print(f"speculative PAC slots sampled: {spec_slots_seen}  ({NVARIANTS} variants each)")
    for key in ("correct+auth", "correct+strip", "wrong+auth", "wrong+strip", "other+auth", "other+strip"):
        if spec_tally.get(key):
            print(f"    {key:16s}: {spec_tally[key]}")
    have = lambda k: any(x.startswith(k) for x in spec_tally if spec_tally[x])
    print(f"  options present -> correct-sig:{have('correct')}  wrong-sig:{have('wrong')}  "
          f"strip:{any('strip' in x for x in spec_tally if spec_tally[x])}")
if KIND == "pac":
    print(f"enacted ran FPAC-free: {ran_ok}/{ran_ok + ran_fail}")
if KIND in ("mte", "both"):
    print(f"MTE retag slots: {mte_slots}  (tag-check enforcement NOT verifiable — no MTE HW)")
    print(f"speculative MTE slots sampled: {mte_spec_seen}  ({NVARIANTS} variants each)")
    for key in sorted(mte_tally):
        print(f"    {key:16s}: {mte_tally[key]}")
    mhave = lambda k: any(x.startswith(k) for x in mte_tally if mte_tally[x])
    print(f"  options present -> correct-delta:{mhave('correct')}  wrong-delta:{mhave('wrong')}")
