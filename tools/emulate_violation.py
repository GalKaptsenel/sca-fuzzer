#!/usr/bin/env python3
"""Independent by-hand emulator of a violation's sandboxed_test_case (no CE/htrace used).

    python3 tools/emulate_violation.py <violation_dir> <x29_hex> <idxA> <idxB>

x29 = &main_region = `cat /sys/executor/print_sandbox_base` (valid only while the module
stays loaded). Prints the architectural and speculative (mispredict-the-first-cond-branch)
line-by-line flows for each input, with per-access cache-set, and the speculative-only
(leaked) sets. Independent cross-check for tools/ce_always_mispredict.py.
"""
import numpy as np, os, re, sys
D=sys.argv[1]; X29=int(sys.argv[2],16); A=int(sys.argv[3]); B=int(sys.argv[4])
def _exists(p): return p if os.path.exists(p) else None
_tc = _exists(f"{D}/sandboxed_test_case.asm") or f"{D}/sandboxed_test_case"
def _inp(i):  # saved inputs are input_NNNN_nzcv_scheme.bin (old: input_NNNN.bin)
    base=f"{D}/input_{i:04d}"
    return next(base+s for s in ("_nzcv_scheme.bin",".bin") if os.path.exists(base+s))
M64=(1<<64)-1; M32=(1<<32)-1; u64=lambda v:v&M64; u32=lambda v:v&M32
prog=[]
for t in open(_tc):
    s=t.split("//")[0].strip()
    if not s or s.startswith(".section") or s.startswith(".macro"): continue
    if s.endswith(":") or not s.startswith("."): prog.append(s)
labels={s[:-1]:i for i,s in enumerate(prog) if s.endswith(":")}
isimm=lambda t: re.fullmatch(r'-?(0x[0-9a-fA-F]+|\d+)',t.strip().replace('#','')) is not None
immv=lambda t: int(t.strip().replace('#',''),0)
def reg(t): t=t.strip().rstrip(','); return ('w','x'+t[1:]) if t[0]=='w' else ('x',t)
def load_mem(fn):
    a=np.fromfile(fn,dtype=np.uint8); sb=bytearray(a[0:8192].tobytes())
    g=np.frombuffer(a[8192:8256].tobytes(),dtype=np.uint64); return sb,{f"x{i}":int(g[i]) for i in range(6)}
def emu(sb,gpr,force):
    R={f"x{i}":0 for i in range(31)}; R.update(gpr); R['x29']=X29; acc=[];log=[];pc=0;st=0
    g=lambda t:(u32(R[reg(t)[1]]) if reg(t)[0]=='w' else u64(R[reg(t)[1]]))
    def s_(t,v): w,n=reg(t); R[n]=u32(v) if w=='w' else u64(v)
    while pc<len(prog) and st<400:
        st+=1; ins=prog[pc]
        if ins.endswith(":"): pc+=1; continue
        op=ins.split()[0].upper(); rest=ins[len(op):].strip(); note=""
        if op=="NOP": pass
        elif op=="B":
            if rest.strip()==".test_case_exit": log.append((ins,"")); break
            log.append((ins,f"-> {rest.strip()}")); pc=labels[rest.strip()]; continue
        elif op in("TBNZ","TBZ","CBZ","CBNZ"):
            p=[x.strip() for x in rest.split(",")]; a=p[0]; tgt=p[-1]
            if op in("TBNZ","TBZ"): bit=immv(p[1]); cond=((g(a)>>bit)&1); cond = cond if op=="TBNZ" else (cond^1)
            else: cond=(g(a)!=0) if op=="CBNZ" else (g(a)==0)
            go=cond if force is None else force
            log.append((ins,f"cond={int(cond)} => {'TAKEN '+tgt if go else 'fall-through'}{'  [SPEC]' if force is not None else ''}"))
            pc=labels[tgt] if go else pc+1; continue
        elif op in("AND","ORR","EOR"):
            p=[x.strip() for x in rest.split(",")]; d,a=p[0],p[1]; sh=0;ty=None
            if p[-1].lower().startswith(("lsr","lsl","asr")): ty=p[-1].split()[0].lower(); sh=immv(p[-1].split()[1]); b=p[2]
            else: b=p[2]
            av=g(a); bv=immv(b) if isimm(b) else g(b); bv=bv>>sh if ty=="lsr" else (bv<<sh if ty=="lsl" else bv)
            s_(d,{'AND':av&bv,'ORR':av|bv,'EOR':av^bv}[op]); note=f"{reg(d)[1]}=0x{g(d):x}"
        elif op=="ADD":
            d,a,b=[x.strip() for x in rest.split(",")]; bv=R['x29'] if b=='x29' else (immv(b) if isimm(b) else g(b)); s_(d,g(a)+bv); note=f"{reg(d)[1]}=0x{g(d):x}"
        elif op in("SUB","SUBS","ADDS"):
            p=[x.strip() for x in rest.split(",")]; d,a,b=p[0],p[1],p[2]; lsl=immv(p[3].split('#')[1]) if len(p)>3 and 'lsl' in p[3] else 0
            bv=(immv(b)<<lsl) if isimm(b) else g(b); s_(d, g(a)+bv if op=="ADDS" else g(a)-bv); note=f"{reg(d)[1]}=0x{g(d):x}"
        elif op=="UDIV":
            d,a,b=[x.strip() for x in rest.split(",")]; bv=g(b); s_(d,g(a)//bv if bv else 0); note=f"{reg(d)[1]}=0x{g(d):x}"
        elif op in("LDR","STR"):
            rd=rest[:rest.index('[')].strip().rstrip(','); mem=rest[rest.index('['):]
            mm=re.match(r"\[([^\],]+)(?:,\s*([^\]]+))?\](!?)(?:,\s*(-?\d+))?",mem)
            base=mm.group(1).strip(); off=mm.group(2); bang=mm.group(3)=='!'; post=mm.group(4); bn=reg(base)[1]; disp=0
            if off: off=off.strip(); disp=immv(off) if isimm(off) else g(off)
            if bang: R[bn]=u64(R[bn]+disp); addr=R[bn]
            else: addr=u64(R[bn]+disp)
            w=4 if reg(rd)[0]=='w' else 8
            if op=="LDR": s_(rd,int.from_bytes(sb[addr-X29:addr-X29+w],'little') if 0<=addr-X29<=8192-w else 0)
            o=addr-X29; acc.append(((o//64)%64)); 
            if post is not None: R[bn]=u64(R[bn]+int(post))
            note=f"off={o:4d} set {(o//64)%64:2d} ({'main ' if 0<=o<4096 else 'fault'})"+(f" -> {reg(rd)[1]}=0x{g(rd):x}" if op=='LDR' else f" <- 0x{g(rd):x}")
        elif op in("CSINV","CSNEG"):
            d,a=[x.strip() for x in rest.split(",")][:2]; s_(d,g(a)); note=f"{reg(d)[1]}=0x{g(d):x}"
        log.append((ins,note)); pc+=1
    return log,acc
for idx in (A,B):
    sb,gp=load_mem(_inp(idx))
    la,aa=emu(bytearray(sb),dict(gp),None); tk=any("TAKEN" in n for _,n in la)
    ls,asp=emu(bytearray(sb),dict(gp),not tk)
    print(f"\n================= INPUT #{idx} =================")
    print("  regs:", " ".join(f"x{i}=0x{gp[f'x{i}']:x}" for i in range(6)))
    print(" --- ARCHITECTURAL (= ctrace) ---")
    for i,(ins,n) in enumerate(la): print(f"  {i+1:2d}. {ins:30s}{'  ; '+n if n else ''}")
    print("   arch sets:", aa)
    print(" --- SPECULATIVE (force opposite of arch) ---")
    for i,(ins,n) in enumerate(ls): print(f"  {i+1:2d}. {ins:30s}{'  ; '+n if n else ''}")
    print("   spec sets:", asp, " spec-only:", sorted(set(asp)-set(aa)))
