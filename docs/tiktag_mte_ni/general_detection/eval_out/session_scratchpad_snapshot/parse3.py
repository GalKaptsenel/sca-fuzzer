import re, sys
from collections import defaultdict
path=sys.argv[1]
txt=open(path).read().splitlines()
MASK=(1<<56)-1
blocks=[]; cur={'regs':{}}
for ln in txt:
    rm=re.match(r'^\s+(x\d\d|SP|PC)\s*:\s*(0x[0-9a-f]+)\s*$',ln)
    if rm: cur['regs'][rm.group(1)]=int(rm.group(2),16); continue
    em=re.match(r'^\s+Encoding:\s+(0x[0-9a-f]+)\s+\((.*)\)\s*$',ln)
    if em: cur['disasm']=em.group(2); continue
    if ln.strip().startswith('Index:'): cur['index']=int(ln.split(':')[1]); continue
    if 'Speculation nesting:' in ln: cur['nesting']=int(ln.split(':')[1]); continue
    if 'Has memory access:' in ln:
        cur['hasmem']=int(ln.split(':')[1]); blocks.append(cur); cur={'regs':{}}; continue
def rk(name):  # x2 -> x02
    n=name[1:]; return 'x'+n.zfill(2)
def info(base,x29):
    ea=base&MASK; off=ea-(x29&MASK); gran=((ea&~15)-((x29&MASK)&~15))//16; st=(ea>>6)&63
    region='main' if 0<=off<4096 else ('FAULTY' if 4096<=off<8192 else 'OUT')
    return off,gran,st,region
basere=re.compile(r'\[(x\d+|sp)')
print("=== memory accesses (index,nesting,set,granule,off,region,disasm) ===")
hot=defaultdict(set)
for b in blocks:
    if not b.get('hasmem'): continue
    d=b.get('disasm',''); m=basere.search(d); x29=b['regs'].get('x29')
    if not m: print(f" idx={b['index']} nest={b['nesting']} {d!r} <no-base>"); continue
    base=b['regs'].get(rk(m.group(1)))
    if base is None: print(f" idx={b['index']} nest={b['nesting']} {d!r} base {m.group(1)} missing"); continue
    off,gran,st,region=info(base,x29)
    hot[b['nesting']].add(st)
    flag=' <== ldp@0x120' if '4124' in '%x'%b['regs'].get('PC',0) else ''
    print(f" idx={b['index']:>3} nest={b['nesting']} set={st:>2} gran={gran:>4} off={off:>5} {region:>6} {d}{flag}")
print("\narch(nest0) sets:", sorted(hot.get(0,[])))
for n in sorted(k for k in hot if k>0): print(f"spec(nest{n}) sets:", sorted(hot[n]))
# ldp specifically
print("\n=== ldp @0x4124 detail ===")
for b in blocks:
    if '4124' in '%x'%b['regs'].get('PC',0):
        x2=b['regs'].get('x02'); x29=b['regs'].get('x29')
        off,gran,st,region=info(x2,x29)
        print(f" idx={b['index']} nest={b['nesting']} x02={hex(x2)} off={off} gran={gran} set={st} region={region}")
