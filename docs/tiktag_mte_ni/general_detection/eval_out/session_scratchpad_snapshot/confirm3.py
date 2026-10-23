#!/usr/bin/env python3
"""3-repeat stability confirmation of the present-on-mismatch contrast per seal, position-controlled.
For each seal (gate, probe set): measure the probe set residency at genuine (uimm4=7, match) vs mismatch
(uimm4=9), 3 independent repetitions each (fresh trace batches)."""
import glob, os, sys
ADDG=0x91800000
def res1(ht,s):
    raw=ht.raw; return sum((int(r)>>s)&1 for r in raw)/len(raw) if raw else 0.0
cfg,vd,sus=os.path.abspath(sys.argv[1]),os.path.abspath(sys.argv[2]),int(sys.argv[3])
reps=int(sys.argv[sys.argv.index("--reps")+1]) if "--reps" in sys.argv else 1500
P=14
repo="/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0,repo)
from src.config import CONF; CONF.load(cfg)
from src import factory
from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
wd="/tmp/confirm3_wd"; os.makedirs(wd,exist_ok=True); os.chdir(wd)
fz=factory.get_fuzzer(os.path.join(repo,"base.json"),wd,f"{vd}/generated.asm",""); fz.initialize_modules()
ex=fz.executor; ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
def load(i): return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0],"rb").read()).input_
inp=load(sus); g=ex.genuine_variant(inp)
Reloc=type(g.code_reloc[0]); relocs=list(g.code_reloc)
prefix=[ex.genuine_variant(load(i)) for i in range(P+6) if i!=sus][:P]
def dput(v,d): return (v&~(0xF<<10))|((d&0xF)<<10)
def variant(gate,d):
    k=next(j for j,r in enumerate(relocs) if (r.value&0xFFC00000)==ADDG and r.offset==gate)
    plan=list(relocs); plan[k]=Reloc(relocs[k].offset,dput(relocs[k].value,d),relocs[k].rtype)
    return ExecutorInput(g.input_,code_reloc=tuple(plan),mte_tags=g.mte_tags,pac_keys=g.pac_keys,bpu_training=g.bpu_training,env_plan=g.env_plan)
def meas(v,s):
    hts,_=ex.trace_test_case(prefix+[v],reps); return res1(hts[-1],s)
print(f"3-repeat confirm reps={reps} prefix={len(prefix)} (match=uimm4 7, mismatch=uimm4 9)")
for name,gate,s in [("A",0xf8,1),("B",0x10c,4),("D",0x120,9)]:
    gm=[meas(variant(gate,7),s) for _ in range(3)]
    mm=[meas(variant(gate,9),s) for _ in range(3)]
    print(f"  seal {name} set{s}: match={['%.2f'%x for x in gm]} (mean {sum(gm)/3:.2f}) | mismatch={['%.2f'%x for x in mm]} (mean {sum(mm)/3:.2f})")
