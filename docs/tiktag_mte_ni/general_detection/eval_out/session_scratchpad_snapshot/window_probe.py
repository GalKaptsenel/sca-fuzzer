#!/usr/bin/env python3
"""Window isolation by per-granule memory retag. Pointers all GENUINE (tag6). Retag ONE memory granule to
a mismatch (9) at a time; any speculative access that READS that granule mismatches and fires its stream.
Maps which window's access reads which granule:
  g0   = A(W2)&B -> set1 ; g189 = A(W1) -> set47 ; g287 = D(W1) -> set7 ; g305 = D(W2) -> set12+ ; g255 = arch store (confound)."""
import glob, os, sys
def resid(ht):
    raw=ht.raw; return [0.0]*64 if not raw else [sum((int(r)>>s)&1 for r in raw)/len(raw) for s in range(64)]
cfg,vd,sus=os.path.abspath(sys.argv[1]),os.path.abspath(sys.argv[2]),int(sys.argv[3])
reps=int(sys.argv[sys.argv.index("--reps")+1]) if "--reps" in sys.argv else 1500
P=14
repo="/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0,repo)
from src.config import CONF; CONF.load(cfg)
from src import factory
from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
wd="/tmp/winprobe_wd"; os.makedirs(wd,exist_ok=True); os.chdir(wd)
fz=factory.get_fuzzer(os.path.join(repo,"base.json"),wd,f"{vd}/generated.asm",""); fz.initialize_modules()
ex=fz.executor; ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
def load(i): return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0],"rb").read()).input_
inp=load(sus); g=ex.genuine_variant(inp)
prefix=[ex.genuine_variant(load(i)) for i in range(P+6) if i!=sus][:P]
def ei(mte): return ExecutorInput(g.input_,code_reloc=g.code_reloc,mte_tags=mte,pac_keys=g.pac_keys,bpu_training=g.bpu_training,env_plan=g.env_plan)
def tail(v):
    hts,_=ex.trace_test_case(prefix+[v],reps); return resid(hts[-1])
rg=tail(g); floor=set(s for s in range(64) if rg[s]>0.5)
print(f"window_probe reps={reps} floor={sorted(floor)}")
for gran,label in [(0,"g0 A(W2)&B->s1"),(189,"g189 A(W1)->s47"),(287,"g287 D(W1)->s7"),(305,"g305 D(W2)->s12+"),(255,"g255 archstore confound")]:
    mte=list(g.mte_tags); mte[gran]=9
    r=tail(ei(mte)); lit=[s for s in range(64) if r[s]>0.5]; leak=[s for s in lit if s not in floor]
    pr={s:round(r[s],2) for s in (1,7,12,47)}
    print(f"  retag {label:24s}: leak={leak} run=[{min(leak) if leak else '-'}..{max(leak) if leak else '-'}] probes{pr}")
