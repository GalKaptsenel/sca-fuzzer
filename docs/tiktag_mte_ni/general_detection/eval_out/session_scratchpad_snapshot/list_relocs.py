import glob, os, sys
repo="/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0, repo)
from src.config import CONF
CFG=sys.argv[1]; VD=sys.argv[2]; SUS=int(sys.argv[3])
CONF.load(CFG)
from src import factory
from src.aarch64.aarch64_executor_input_encoder import deserialize
ADDG=0x91800000; SUBG=0xD1800000
wd="/tmp/list_relocs_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
fz=factory.get_fuzzer(os.path.join(repo,"base.json"), wd, f"{VD}/generated.asm",""); fz.initialize_modules()
ex=fz.executor; ex.load_test_case(fz.asm_parser.parse_file(f"{VD}/generated.asm"))
def load(i): return deserialize(open(glob.glob(f"{VD}/input_{i:04d}.*")[0],"rb").read()).input_
inp=load(SUS)
g=ex.genuine_variant(inp)
print(f"=== genuine code_reloc entries ({len(g.code_reloc)}): ADDG/SUBG only ===")
for k,r in enumerate(g.code_reloc):
    v=r.value
    if (v&0xFFC00000)==ADDG or (v&0xFFC00000)==SUBG:
        nm="ADDG" if (v&0xFFC00000)==ADDG else "SUBG"
        u6=(v>>16)&0x3F; u4=(v>>10)&0xF; Rn=(v>>5)&0x1F; Rd=v&0x1F
        print(f"  k={k:2d} off=0x{r.offset:03x} {nm} X{Rd},X{Rn},#{u6*16},#{u4}  rtype={r.rtype} value=0x{v:08x}")
# resolution: which seals are speculative/decoyable
rsv=ex._resolve(inp)
print("\n=== resolved seal entries (speculative/alts) ===")
for e in rsv._entries:
    offs=rsv._offsets.get(id(e.sealing))
    offs_s=sorted(hex(o) for o in offs) if offs else None
    print(f"  speculative={e.speculative} has_alts={bool(e.alts)} offsets={offs_s}")
# decoy offsets
dec=set()
for r in rsv._entries:
    if r.speculative and r.alts:
        o=rsv._offsets.get(id(r.sealing))
        if o: dec.update(o)
print(f"\ndecoy(speculative+alts) offsets = {sorted(hex(o) for o in dec)}")
print(f"mte_tags default sample: gran0={g.mte_tags[0]} gran255={g.mte_tags[255]} gran305={g.mte_tags[305]}  (len={len(g.mte_tags)})")
