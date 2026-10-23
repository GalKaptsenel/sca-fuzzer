#!/usr/bin/env python3
"""Seal-combination matrix in ONE device session. For each subset of seals {A=0xf8,B=0x10c,D=0x120}, set
those gates to a MISMATCH pointer tag (uimm4=9 -> ptag8) and the rest genuine (7), read the full 64-set
residency at the fixed tail slot (position-controlled). Prints resident sets and the leak block (excluding
the genuine arch/constant floor)."""
import glob, os, sys
ADDG=0x91800000
def resid(ht): 
    raw=ht.raw; return [0.0]*64 if not raw else [sum((int(r)>>s)&1 for r in raw)/len(raw) for s in range(64)]
cfg,vd,sus=os.path.abspath(sys.argv[1]),os.path.abspath(sys.argv[2]),int(sys.argv[3])
reps=int(sys.argv[sys.argv.index("--reps")+1]) if "--reps" in sys.argv else 1500
P=14
repo="/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0,repo)
from src.config import CONF; CONF.load(cfg)
from src import factory
from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
wd="/tmp/combo_wd"; os.makedirs(wd,exist_ok=True); os.chdir(wd)
fz=factory.get_fuzzer(os.path.join(repo,"base.json"),wd,f"{vd}/generated.asm",""); fz.initialize_modules()
ex=fz.executor; ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
def load(i): return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0],"rb").read()).input_
inp=load(sus); g=ex.genuine_variant(inp)
Reloc=type(g.code_reloc[0]); relocs=list(g.code_reloc)
gates={'A':0xf8,'B':0x10c,'D':0x120}
kof={n:next(k for k,r in enumerate(relocs) if (r.value&0xFFC00000)==ADDG and r.offset==o) for n,o in gates.items()}
prefix=[ex.genuine_variant(load(i)) for i in range(P+6) if i!=sus][:P]
def dput(v,d): return (v&~(0xF<<10))|((d&0xF)<<10)
def variant(mismatched):
    plan=list(relocs)
    for n in mismatched:
        k=kof[n]; plan[k]=Reloc(relocs[k].offset,dput(relocs[k].value,9),relocs[k].rtype)  # ptag8 mismatch
    return ExecutorInput(g.input_,code_reloc=tuple(plan),mte_tags=g.mte_tags,pac_keys=g.pac_keys,bpu_training=g.bpu_training,env_plan=g.env_plan)
def tail(v):
    hts,_=ex.trace_test_case(prefix+[v],reps); return resid(hts[-1])
rg=tail(g)
floor=set(s for s in range(64) if rg[s]>0.5)
print(f"{os.path.basename(vd)} combos reps={reps} prefix={len(prefix)}")
print(f"genuine(all-match) resident>0.5: {sorted(floor)}")
combos=[['A'],['B'],['D'],['A','B'],['A','D'],['B','D'],['A','B','D']]
for c in combos:
    r=tail(variant(c))
    lit=[s for s in range(64) if r[s]>0.5]
    leak=[s for s in lit if s not in floor]
    run=f"[{min(leak)}..{max(leak)}]" if leak else "[]"
    probes={s:round(r[s],2) for s in (1,4,5,9,11,12,47)}
    print(f"  {'+'.join(c):6s} lit={lit}")
    print(f"         leak(minus floor)={leak} run={run} n={len(leak)}  probes{probes}")
