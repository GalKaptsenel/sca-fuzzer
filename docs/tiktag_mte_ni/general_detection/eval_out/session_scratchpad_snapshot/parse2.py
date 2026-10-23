import re, sys
from collections import defaultdict
path=sys.argv[1]
txt=open(path).read().splitlines()
MASK=(1<<56)-1
# Split into detailed CPU-state blocks: each ends with InstrMetadata (Index/nesting/hasmem).
blocks=[]
cur={'regs':{}}
for ln in txt:
    rm=re.match(r'^\s+(x\d+|SP|PC)\s*:\s*(0x[0-9a-f]+)\s*$',ln)
    if rm: cur['regs'][rm.group(1)]=int(rm.group(2),16); continue
    em=re.match(r'^\s+Encoding:\s+(0x[0-9a-f]+)\s+\((.*)\)\s*$',ln)
    if em: cur['enc']=em.group(1); cur['disasm']=em.group(2); continue
    if ln.strip().startswith('Index:'): cur['index']=int(ln.split(':')[1]); continue
    if 'Speculation nesting:' in ln: cur['nesting']=int(ln.split(':')[1]); continue
    if 'Has memory access:' in ln:
        cur['hasmem']=int(ln.split(':')[1]); blocks.append(cur); cur={'regs':{}}; continue
print(f"detailed blocks: {len(blocks)}")
# nesting profile over index order
prof=[(b.get('index'),b.get('nesting')) for b in blocks]
# compress runs
runs=[]
for idx,n in prof:
    if runs and runs[-1][2]==n: runs[-1][1]=idx
    else: runs.append([idx,idx,n])
print("nesting runs (start_idx,end_idx,nesting):")
for s,e,n in runs: print(f"   idx {s:>3}..{e:>3} nesting={n}")
# ldp @ ...4124
print("\n=== ldp @0x4124 ===")
for b in blocks:
    if b.get('regs',{}).get('PC',0)&0xffff==0x4124 or b.get('disasm','').startswith('ldp w2, w1, [x2]'):
        x2=b['regs'].get('x2'); x29=b['regs'].get('x29')
        pc=b['regs'].get('PC')
        if x2 and x29:
            ea=x2&MASK; off=ea-(x29&MASK); gran=((ea&~15)-((x29&MASK)&~15))//16; st=(ea>>6)&63
            print(f" index={b.get('index')} nesting={b.get('nesting')} PC={hex(pc)} x2={hex(x2)} off={off} gran={gran} set={st}")
# Build effective-address per memory-access by decoding base reg from disasm (simple [xN] or [xN, ...])
print("\n=== memory accesses: (index,nesting,disasm,set,granule) ===")
basere=re.compile(r'\[(x\d+|sp)')
hot_by_nest=defaultdict(set)
for b in blocks:
    if not b.get('hasmem'): continue
    d=b.get('disasm',''); m=basere.search(d)
    x29=b['regs'].get('x29')
    if not m or x29 is None:
        print(f" index={b.get('index')} nesting={b.get('nesting')} {d!r}  <base?>"); continue
    br=m.group(1); base=b['regs'].get(br)
    if base is None:
        print(f" index={b.get('index')} nesting={b.get('nesting')} {d!r} base {br} missing"); continue
    ea=base&MASK; off=ea-(x29&MASK); gran=((ea&~15)-((x29&MASK)&~15))//16; st=(ea>>6)&63
    inmain = 0<=off<4096; infaulty = 4096<=off<8192
    region = 'main' if inmain else ('FAULTY' if infaulty else 'OUT')
    hot_by_nest[b.get('nesting')].add(st)
    print(f" index={b.get('index'):>3} nest={b.get('nesting')} set={st:>2} gran={gran:>4} off={off:>5} {region:>6} base={br}  {d}")
print("\nsets touched arch(nest0):", sorted(hot_by_nest.get(0,set())))
for n in sorted(k for k in hot_by_nest if k>0):
    print(f"sets touched spec(nest{n}):", sorted(hot_by_nest[n]))
