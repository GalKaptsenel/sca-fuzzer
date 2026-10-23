import glob, os, sys
REPO="/home/gal_k_1_1998/revizor/sca-fuzzer"; sys.path.insert(0, REPO)
cfg=os.path.abspath(sys.argv[1]); vd=os.path.abspath(sys.argv[2]); sus=int(sys.argv[3])
variant=sys.argv[4] if len(sys.argv)>4 else "genuine"
from src.config import CONF; CONF.load(cfg)
from src import factory
from src.aarch64.aarch64_executor_input_encoder import deserialize
from src.aarch64.aarch64_relocations import apply_relocations
from src.aarch64.aarch64_contract_executor import ExecutionClause
from src.aarch64.aarch64_mte import MTE_GRANULE
wd="/tmp/ce_trace_wd"; os.makedirs(wd,exist_ok=True); os.chdir(wd)
# disasm map
dis={}
for ln in open(f"{SCR_DIS}/disasm_0001.txt".replace("$SCR_DIS",os.environ.get("SCR_DIS","")) ) if False else []:
    pass
fz=factory.get_fuzzer(os.path.join(REPO,"base.json"),wd,f"{vd}/generated.asm",""); fz.initialize_modules()
ex=fz.executor; ex.load_test_case(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
inp=deserialize(open(glob.glob(f"{vd}/input_{sus:04d}.*")[0],"rb").read()).input_
if ex._sandbox_base is None:
    ex._sandbox_base,_=ex.read_base_addresses()
g=ex.genuine_variant(inp)
if variant=="genuine":
    V=g
else:
    sv=dict(ex.variants_for_input(inp))
    from src.aarch64.aarch64_executor import NIVariant
    if variant=="noncanon": V=ex.noncanon_variant(inp)
    else: V=sv[next(k for k in sv if str(k)==variant)]
rsv=ex._resolve(inp)
code=apply_relocations(rsv.object_code, rsv.genuine() if variant=="genuine" else rsv.genuine())
# use the sealed resolve for addresses (tag doesn't change addr/set)
sealed=ex._sealer.seal(fz.asm_parser.parse_file(f"{vd}/generated.asm"))
rsv2=sealed.resolve(inp)
code=apply_relocations(rsv2.object_code, rsv2.genuine())
cer=ex._contract_executor.run(ex._make_ce_execution(
    code, inp, ex._sandbox_base, CONF.model_max_nesting, CONF.model_max_spec_window,
    ExecutionClause.COND|ExecutionClause.BPAS, mte_tags=ex._mte_tags_for(inp),
    pac_keys=ex._input_pac_keys(inp), pac_profile=ex._pac_profile_value))
cb=cer[0].cpu.pc; MASK=(1<<56)-1
print("entry GPR x0..x7:", [hex(int(cer[0].cpu.gpr[i])) for i in range(8)])
print("x29(base):", hex(int(cer[0].cpu.gpr[29])), " sandbox_base:", hex(ex._sandbox_base))
print("meta fields:", [a for a in dir(cer[0].metadata) if not a.startswith('_')])
print(f"\n{'idx':>3} {'off':>5} {'nest':>4}  access(ld/st ea set gran)")
base=int(cer[0].cpu.gpr[29])
for i,ite in enumerate(cer):
    nest=int(ite.metadata.speculation_nesting); off=ite.cpu.pc-cb
    accs=list(ite.metadata.accesses())
    s=""
    for ma in accs:
        ea=ma.effective_address & MASK; cset=(ea>>6)&63
        gran=((ea & ~(MTE_GRANULE-1))-((base&MASK)&~(MTE_GRANULE-1)))//MTE_GRANULE
        isst=getattr(ma,'is_store',getattr(ma,'store',None))
        s+=f" [{'ST' if isst else 'LD'} ea=0x{ea:x} set={cset} g={gran}]"
    print(f"{i:>3} 0x{off:<3x} {nest:>4} {s}")
