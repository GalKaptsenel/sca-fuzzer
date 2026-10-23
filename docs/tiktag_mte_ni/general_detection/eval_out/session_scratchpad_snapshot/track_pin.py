#!/usr/bin/env python3
"""Clean single-seal granule tracking: sweep one gate's pointer delta 0..15 for several memory tags T of
a granule, while PINNING other named gates to uimm4=0 (pointer tag 15 = TCMA-unchecked => their check is
OFF, removing the shared-granule confound). Position-controlled (fixed prefix, tail slot).
Usage: track_pin.py <cfg> <vd> <sus> <tset> <granule> <gate 0xNN> [--pin 0xMM[,0xKK]] [--tags a,b,c] [--reps N] [--prefix P]
"""
import glob, os, sys
ADDG_OP=0x91800000
def res1(ht,s):
    raw=ht.raw; return sum((int(r)>>s)&1 for r in raw)/len(raw) if raw else 0.0
def main():
    cfg,vd=os.path.abspath(sys.argv[1]),os.path.abspath(sys.argv[2])
    sus,tset,gran=int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
    gate=int(sys.argv[6],16)
    pins=[int(x,16) for x in sys.argv[sys.argv.index("--pin")+1].split(",")] if "--pin" in sys.argv else []
    tags=[int(x) for x in sys.argv[sys.argv.index("--tags")+1].split(",")] if "--tags" in sys.argv else [3,6,9,12]
    reps=int(sys.argv[sys.argv.index("--reps")+1]) if "--reps" in sys.argv else 2000
    P=int(sys.argv[sys.argv.index("--prefix")+1]) if "--prefix" in sys.argv else 14
    repo="/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0,repo)
    from src.config import CONF; CONF.load(cfg)
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize, ExecutorInput
    wd="/tmp/track_pin_wd"; os.makedirs(wd,exist_ok=True); os.chdir(wd)
    fz=factory.get_fuzzer(os.path.join(repo,"base.json"),wd,f"{vd}/generated.asm",""); fz.initialize_modules()
    ex=fz.executor; ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
    def load(i): return deserialize(open(glob.glob(f"{vd}/input_{i:04d}.*")[0],"rb").read()).input_
    inp=load(sus); g=ex.genuine_variant(inp)
    Reloc=type(g.code_reloc[0]); relocs=list(g.code_reloc)
    cell_tag=int(g.mte_tags[gran])
    kgate=next(k for k,r in enumerate(relocs) if (r.value&0xFFC00000)==ADDG_OP and r.offset==gate)
    C=(relocs[kgate].value>>10)&0xF
    kpins=[next(k for k,r in enumerate(relocs) if (r.value&0xFFC00000)==ADDG_OP and r.offset==p) for p in pins]
    prefix=[ex.genuine_variant(load(i)) for i in range(P+6) if i!=sus][:P]
    def dput(v,d): return (v&~(0xF<<10))|((d&0xF)<<10)
    def ei(d,mte):
        plan=list(relocs); plan[kgate]=Reloc(relocs[kgate].offset,dput(relocs[kgate].value,d),relocs[kgate].rtype)
        for kp in kpins:
            plan[kp]=Reloc(relocs[kp].offset,dput(relocs[kp].value,0),relocs[kp].rtype)  # ptag15 unchecked
        return ExecutorInput(g.input_,code_reloc=tuple(plan),mte_tags=mte,pac_keys=g.pac_keys,bpu_training=g.bpu_training,env_plan=g.env_plan)
    def tset_res(d,mte):
        hts,_=ex.trace_test_case(prefix+[ei(d,mte)],reps); return res1(hts[-1],tset)
    print(f"{os.path.basename(vd)} sus#{sus} set={tset} granule={gran} cell_tag={cell_tag} gate=0x{gate:x} C={C} pins={[hex(p) for p in pins]} reps={reps} prefix={len(prefix)}")
    print("rows=memtag T; cols=pointer delta 0..15; genuine match delta=(C+T-cell_tag)%16")
    for T in tags:
        mte=list(g.mte_tags); mte[gran]=T
        row={d:tset_res(d,mte) for d in range(16)}
        base_tag=(cell_tag-C)%16
        checked=[d for d in range(16) if (base_tag+d)%16 not in (0,15)]
        dip=min(checked,key=lambda d:row[d])
        print(f"  T={T:2d} match_delta={(C+T-cell_tag)%16:2d} dip={dip:2d}  "+" ".join(f"{d}:{row[d]:.2f}" for d in range(16)))
if __name__=="__main__": main()
