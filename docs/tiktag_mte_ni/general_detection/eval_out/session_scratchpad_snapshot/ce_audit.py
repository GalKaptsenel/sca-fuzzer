#!/usr/bin/env python3
"""ADVERSARIAL audit: CE-only (NO device). Enumerate every CE memory access for the genuine variant of
violation-261023-021502 input #0, with nesting / pc-offset / base-reg / EA / granule / L1D set.
Answers: is set 52 touched by CE? is set 46? at what nesting is granule 442 accessed? which seal gates it?
Mirrors accounting.py's device-free path (no load_test_case / trace_test_case / read_base_addresses)."""
import sys, glob
REPO = "/home/gal_k_1_1998/revizor/sca-fuzzer"
sys.path.insert(0, REPO)
VD = "/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/collect/session_20261023_011712/violation-261023-021502"
CFG = VD + "/org-config.yaml"
SUS = 0

from src.config import CONF
CONF.load(CFG)
from src.fuzzer import NoninterferenceFuzzer
from src.aarch64.aarch64_executor_input_encoder import deserialize
from src.aarch64.aarch64_relocations import apply_relocations
from src.aarch64.aarch64_trace import _cache_sets
from src.aarch64.aarch64_contract_executor import ExecutionClause
from src.aarch64.aarch64_mte import MTE_GRANULE, MTE_INITIAL_DEFAULT_TAG

BASE = 0xffffff8864da1000  # page-aligned; identical set-numbering to any page-aligned base
MASK = (1 << 56) - 1
ADDG_OP = 0x91800000

fz = NoninterferenceFuzzer("base.json", REPO); fz.initialize_modules()
ex = fz.executor
ex._sandbox_base = BASE
sealed = ex._sealer.seal(fz.asm_parser.parse_file(f"{VD}/generated.asm"))
blob = open(glob.glob(f"{VD}/input_{SUS:04d}.*")[0], "rb").read()
inp = deserialize(blob).input_
res = sealed.resolve(inp)
code = apply_relocations(res.object_code, res.genuine())

# MteSealing eligibility (speculative, has alts) -> eligible value regs
elig = {r.sealing.value_reg for r in res._entries
        if r.sealing.__class__.__name__ == "MteSealing" and getattr(r, "speculative", False) and getattr(r, "alts", None)}
print(f"eligible speculative MteSealing value regs: {sorted(elig)}")

# all MteSealing entries + their offsets/regs/speculative flag
print("\n== MteSealing entries (offset, value_reg, speculative, has_alts) ==")
for r in res._entries:
    if r.sealing.__class__.__name__ == "MteSealing":
        off = getattr(r, "offset", None)
        offs = f"0x{off:x}" if isinstance(off, int) else str(off)
        print(f"  off={offs} vreg=x{r.sealing.value_reg} spec={getattr(r,'speculative',None)} alts={getattr(r,'alts',None)}")

mte_tags = ex._mte_tags_for(inp)
print(f"\nmte_tags len={len(mte_tags)} ; tag@granule442={mte_tags[442] if len(mte_tags)>442 else 'OOR'} ; default={MTE_INITIAL_DEFAULT_TAG}")

cer = ex._contract_executor.run(ex._make_ce_execution(
    code, inp, BASE, CONF.model_max_nesting, CONF.model_max_spec_window,
    ExecutionClause.COND | ExecutionClause.BPAS, mte_tags=mte_tags,
    pac_keys=ex._input_pac_keys(inp), pac_profile=ex._pac_profile_value))
cb = cer[0].cpu.pc

def decode_addg_offsets():
    offs = {}
    for o in range(0, len(code), 4):
        w = int.from_bytes(code[o:o+4], "little")
        if (w & 0xFFC00000) == ADDG_OP:
            offs[o] = (w & 0x1f, (w >> 10) & 0xf)  # Xd, tag delta
    return offs
addg_offs = decode_addg_offsets()

# per access: (nesting, off, basereg, EA, granule, sets)
accs = []
arch_sets, spec_sets = {}, {}
gran_nesting = {}   # granule -> set of nestings
set_nesting = {}    # set -> set of nestings
for ite in cer:
    nest = int(ite.metadata.speculation_nesting)
    off = ite.cpu.pc - cb
    if not (0 <= off < len(code)):
        continue
    base_reg = (int.from_bytes(code[off:off+4], "little") >> 5) & 0x1f
    xbase = ite.cpu.gpr[29]
    for ma in ite.metadata.accesses():
        ea = ma.effective_address
        g_abs = ((ea & MASK) & ~(MTE_GRANULE-1))
        g_base = ((xbase & MASK) & ~(MTE_GRANULE-1))
        gidx = (g_abs - g_base) // MTE_GRANULE
        sets = _cache_sets(ma, xbase)
        accs.append((nest, off, base_reg, ea, gidx, tuple(sets), ma.is_write))
        for s in sets:
            (spec_sets if nest else arch_sets).setdefault(s, []).append((off, base_reg, gidx, nest))
            set_nesting.setdefault(s, set()).add(nest)
        gran_nesting.setdefault(gidx, set()).add(nest)

arch_s = set(arch_sets); spec_only = set(spec_sets) - arch_s
print(f"\nCE arch demand sets (nesting0): {sorted(arch_s)}")
print(f"CE spec-only sets             : {sorted(spec_only)}")
print(f"set 46 touched? {'YES' if 46 in set_nesting else 'NO'}  nestings={sorted(set_nesting.get(46,[]))}")
print(f"set 52 touched? {'YES' if 52 in set_nesting else 'NO'}  nestings={sorted(set_nesting.get(52,[]))}")
print(f"granule 442 nestings: {sorted(gran_nesting.get(442, []))}  (0 present => architectural)")

# which accesses hit set 46 / set 52, and which touch granule 442
print("\n== accesses hitting set 46 ==")
for (nest, off, br, ea, gidx, sets, w) in accs:
    if 46 in sets:
        print(f"  nest={nest} off=0x{off:x} base=x{br} gran={gidx} ea=0x{ea & MASK:x} sets={sets} {'W' if w else 'R'}")
print("== accesses hitting set 52 ==")
any52 = False
for (nest, off, br, ea, gidx, sets, w) in accs:
    if 52 in sets:
        any52 = True
        print(f"  nest={nest} off=0x{off:x} base=x{br} gran={gidx} ea=0x{ea & MASK:x} sets={sets} {'W' if w else 'R'}")
if not any52:
    print("  (none — set 52 never accessed by the CE under COND|BPAS max nesting)")
print("== accesses touching granule 442 ==")
for (nest, off, br, ea, gidx, sets, w) in accs:
    if gidx == 442:
        # find nearest preceding ADDG with same Xd == gating seal
        gate = max((o for o in addg_offs if o <= off and addg_offs[o][0] == br), default=None)
        gate_s = f"0x{gate:x}(tagdelta={addg_offs[gate][1]})" if gate is not None else "NONE"
        print(f"  nest={nest} off=0x{off:x} base=x{br} gran=442 ea=0x{ea & MASK:x} sets={sets} {'W' if w else 'R'} gate_ADDG={gate_s}")

# what access does seal 0x118 gate?
print("\n== access gated by ADDG @0x118 (if present) ==")
if 0x118 in addg_offs:
    xd118, d118 = addg_offs[0x118]
    print(f"  ADDG@0x118: Xd=x{xd118} tagdelta={d118}")
    # first access with base==xd118 at off>0x118 (before any later ADDG on same reg)
    later = sorted(o for o in addg_offs if o > 0x118 and addg_offs[o][0] == xd118)
    nextseal = later[0] if later else len(code)
    for (nest, off, br, ea, gidx, sets, w) in accs:
        if br == xd118 and 0x118 < off <= nextseal:
            print(f"  -> nest={nest} off=0x{off:x} base=x{br} gran={gidx} ea=0x{ea & MASK:x} sets={sets} {'W' if w else 'R'}")
else:
    print("  no ADDG at 0x118 in genuine code")
print("\n== all ADDG offsets (off: Xd,tagdelta) ==")
for o in sorted(addg_offs):
    print(f"  0x{o:x}: x{addg_offs[o][0]} d={addg_offs[o][1]}")
