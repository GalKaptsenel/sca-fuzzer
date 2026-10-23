import re, sys
path = sys.argv[1]
lines = open(path).read().splitlines()
# Blocks: a disasm line "  [NNN] 0xADDR: disasm", then register lines "      xNN: 0x..", "SP","PC","NZCV","Encoding","Extra data size",
# then "    InstrMetadata:" with "Index:", "Speculation nesting:", "Has memory access:"
blocks=[]
i=0
cur=None
disasm_re=re.compile(r'^\s*\[(\d+)\]\s+(0x[0-9a-f]+):\s+(.*)$')
reg_re=re.compile(r'^\s+(x\d+|SP|PC)\s*:\s*(0x[0-9a-f]+)$')
for ln in lines:
    m=disasm_re.match(ln)
    if m:
        if cur: blocks.append(cur)
        cur={'idx':int(m.group(1)),'addr':m.group(2),'disasm':m.group(3).strip(),'regs':{},'nesting':None,'hasmem':None,'mindex':None}
        continue
    if cur is None: continue
    rm=reg_re.match(ln)
    if rm: cur['regs'][rm.group(1)]=int(rm.group(2),16)
    if 'Speculation nesting:' in ln: cur['nesting']=int(ln.split(':')[1])
    if 'Has memory access:' in ln: cur['hasmem']=int(ln.split(':')[1])
    if ln.strip().startswith('Index:'): cur['mindex']=int(ln.split(':')[1])
if cur: blocks.append(cur)

MASK=(1<<56)-1
def setof(ea): return (ea>>6)&63
def granule(ea,base): return (((ea&MASK)&~15)-((base&MASK)&~15))//16

# x29 is sandbox base
print(f"total blocks: {len(blocks)}")
# Find the ldp at 0x...4124 occurrences
print("\n=== ldp w2,w1,[x2] @0x4124 occurrences (base=x29) ===")
for b in blocks:
    if b['addr'].endswith('4124'):
        x2=b['regs'].get('x2'); x29=b['regs'].get('x29')
        if x2 is None or x29 is None:
            print(f" trace#{b['idx']} nesting={b['nesting']} regs-missing"); continue
        ea=x2&MASK
        print(f" trace#{b['idx']:>3} mindex={b['mindex']} nesting={b['nesting']} hasmem={b['hasmem']} x2={hex(x2)} ea_off={ea-(x29&MASK)} gran={granule(x2,x29)} set={setof(ea)}")

# All memory-access instructions: nesting, set, granule
print("\n=== all memory accesses: distribution of (nesting,set) ===")
from collections import defaultdict
seen=defaultdict(int)
acc=[]
for b in blocks:
    if b['hasmem']:
        x29=b['regs'].get('x29')
        # best-effort EA: many accesses use a base reg; we can't always know which. Report per-instruction base guess.
        acc.append(b)
# Which sets are touched architecturally (nesting 0) vs spec (nesting>0)? Use the CE's own numbers instead;
# here just print count of mem-access instrs by nesting
byn=defaultdict(int)
for b in acc: byn[b['nesting']]+=1
print(" mem-access instr count by nesting:", dict(byn))
# nesting range
ns=[b['nesting'] for b in blocks if b['nesting'] is not None]
print(" nesting values present:", sorted(set(ns)), " max:", max(ns))
# architectural path length
arch=[b for b in blocks if b['nesting']==0]
print(f" architectural (nesting0) instr count: {len(arch)}; last arch idx={arch[-1]['idx']} disasm={arch[-1]['disasm']}")
