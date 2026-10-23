#!/usr/bin/env python3
"""v2: fix _cache_sets generator exhaustion; full arch/spec-only sets; map every seal->gated access
+nesting; parse report.txt htraces for #0 vs #8. CE-only, no device."""
import sys, glob, re
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
BASE = 0xffffff8864da1000
MASK = (1 << 56) - 1
ADDG_OP = 0x91800000

fz = NoninterferenceFuzzer("base.json", REPO); fz.initialize_modules()
ex = fz.executor; ex._sandbox_base = BASE
sealed = ex._sealer.seal(fz.asm_parser.parse_file(f"{VD}/generated.asm"))
inp = deserialize(open(glob.glob(f"{VD}/input_{SUS:04d}.*")[0], "rb").read()).input_
res = sealed.resolve(inp)
code = apply_relocations(res.object_code, res.genuine())
mte_tags = ex._mte_tags_for(inp)

# map speculative MteSealing -> reloc offset (the decoyed seal)
gen = res.genuine()
print("== relocations (offset, value word, is ADDG, Xd, tagdelta) ==")
addg_offs = {}
for o in range(0, len(code), 4):
    w = int.from_bytes(code[o:o+4], "little")
    if (w & 0xFFC00000) == ADDG_OP:
        addg_offs[o] = (w & 0x1f, (w >> 10) & 0xf)

cer = ex._contract_executor.run(ex._make_ce_execution(
    code, inp, BASE, CONF.model_max_nesting, CONF.model_max_spec_window,
    ExecutionClause.COND | ExecutionClause.BPAS, mte_tags=mte_tags,
    pac_keys=ex._input_pac_keys(inp), pac_profile=ex._pac_profile_value))
cb = cer[0].cpu.pc
accs = []
set_nest = {}
gran_nest = {}
for ite in cer:
    nest = int(ite.metadata.speculation_nesting)
    off = ite.cpu.pc - cb
    if not (0 <= off < len(code)): continue
    br = (int.from_bytes(code[off:off+4], "little") >> 5) & 0x1f
    xbase = ite.cpu.gpr[29]
    for ma in ite.metadata.accesses():
        ea = ma.effective_address
        gidx = (((ea & MASK) & ~(MTE_GRANULE-1)) - ((xbase & MASK) & ~(MTE_GRANULE-1))) // MTE_GRANULE
        sets = list(_cache_sets(ma, xbase))   # MATERIALIZE (was a generator)
        accs.append((nest, off, br, ea & MASK, gidx, tuple(sorted(set(sets))), ma.is_write))
        for s in set(sets):
            set_nest.setdefault(s, set()).add(nest)
        gran_nest.setdefault(gidx, set()).add(nest)

arch_s = sorted(s for s,n in set_nest.items() if 0 in n)
spec_only = sorted(s for s,n in set_nest.items() if 0 not in n)
both = sorted(s for s,n in set_nest.items() if 0 in n and len(n) > 1)
print(f"\nCE arch demand sets (any nesting0): {arch_s}")
print(f"CE spec-only sets (never nesting0): {spec_only}")
print(f"CE arch+spec sets (both)          : {both}")
print(f"set46 nestings={sorted(set_nest.get(46,[]))}  set52 nestings={sorted(set_nest.get(52,[]))}")
print(f"granule442 nestings={sorted(gran_nest.get(442,[]))}")

def prefetch_parent(s, pool):
    best=None
    for D in pool:
        k=(s-D)%64
        if 1<=k<=16 and (best is None or k<best[1]): best=(D,k)
    return best
print(f"\nprefetch parent of 52 among arch sets: {prefetch_parent(52, arch_s)}  (46 -> +6)")

# seal -> gated access nesting
print("\n== each ADDG seal -> gated access (base==Xd, first use after seal) + nesting ==")
for o in sorted(addg_offs):
    xd, d = addg_offs[o]
    later = sorted(x for x in addg_offs if x>o and addg_offs[x][0]==xd)
    nxt = later[0] if later else len(code)
    hits = [(nest,off,gidx,sets,w) for (nest,off,br,ea,gidx,sets,w) in accs if br==xd and o<off<=nxt]
    nestset = sorted(set(n for (n,_,_,_,_) in hits))
    grans = sorted(set(g for (_,_,g,_,_) in hits))
    setz = sorted(set(x for (_,_,_,ss,_) in hits for x in ss))
    print(f"  seal@0x{o:x} x{xd} d={d}: nestings={nestset} granules={grans} sets={setz}")

# which seal is the decoyed (speculative) one? -> the offset that differs across enacted bins
en = sorted(glob.glob(f"{VD}/enacted_test*.bin"))
b0 = open(en[0],"rb").read()
diff=set()
for e in en[1:]:
    o=open(e,"rb").read()
    for i in range(0,min(len(b0),len(o)),4):
        if b0[i:i+4]!=o[i:i+4]: diff.add(i)
print(f"\nDECOYED seal offset(s) (enacted-bin diff): {sorted(hex(x) for x in diff)}")

# report.txt htraces
lines = open(f"{VD}/report.txt").read().splitlines()
def freq(n):
    cur=None; blk=[]
    for ln in lines:
        m=re.match(r"Input #(\d+)$", ln.strip())
        if m: cur=int(m.group(1)); continue
        if cur==n:
            m2=re.match(r"^([\.\^]{8,})\s*\[(\d+)\]", ln.strip())
            if m2: blk.append((int(m2.group(2)), m2.group(1)))
            elif ln.startswith("* Contract"): cur=None
    tot=sum(c for c,_ in blk) or 1; f={}
    for c,pat in blk:
        for i,ch in enumerate(pat):
            if ch=="^": f[63-i]=f.get(63-i,0)+c
    return {k:v/tot for k,v in f.items()}, tot
f0,t0=freq(0); f8,t8=freq(8)
print(f"\nreport.txt samples: #0={t0} #8={t8}")
div=sorted(s for s in range(64) if abs(f0.get(s,0)-f8.get(s,0))>=0.08)
print(f"HW diverging sets |d|>=0.08 (#0 vs #8): {div}")
for s in sorted(set([6,7,8,32,46,52]+div)):
    print(f"  set {s:2d}: #0={f0.get(s,0):.3f} #8={f8.get(s,0):.3f} |d|={abs(f0.get(s,0)-f8.get(s,0)):.3f}  CE_nest={sorted(set_nest.get(s,[])) or '-'}")
