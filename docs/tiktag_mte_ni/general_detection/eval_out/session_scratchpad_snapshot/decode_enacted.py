import struct
D="/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/CONTESTED/fuzzer_072421"
def words(fn):
    b=open(fn,'rb').read()
    return [struct.unpack('<I',b[i:i+4])[0] for i in range(0,len(b),4)]
def kind(w):
    if (w & 0xFFC00000)==0x91800000: return "ADDG"
    if (w & 0xFFC00000)==0xD1800000: return "SUBG"
    return None
base=words(f"{D}/enacted_test0001.bin")
print(f"suspect 0001: {len(base)} words ({len(base)*4} bytes)")
print("\n-- ADDG/SUBG (tag-arith) in suspect 0001 --")
for i,w in enumerate(base):
    k=kind(w)
    if k:
        off=i*4
        uimm6=(w>>16)&0x3F; uimm4=(w>>10)&0xF; Rn=(w>>5)&0x1F; Rd=w&0x1F
        print(f"  off=0x{off:x} idx={i} {k} X{Rd},X{Rn},#{uimm6*16},#{uimm4}  (tagoff_uimm4={uimm4})  word=0x{w:08x}")
for dec in ["0009","0017","0025"]:
    dw=words(f"{D}/enacted_test{dec}.bin")
    print(f"\n-- diff enacted_test{dec} vs 0001 --")
    n=min(len(dw),len(base))
    for i in range(n):
        if dw[i]!=base[i]:
            off=i*4
            k0=kind(base[i]); k1=kind(dw[i])
            u4_0=(base[i]>>10)&0xF; u4_1=(dw[i]>>10)&0xF
            print(f"  off=0x{off:x} idx={i}: 0x{base[i]:08x}({k0},u4={u4_0}) -> 0x{dw[i]:08x}({k1},u4={u4_1})")
    if len(dw)!=len(base):
        print(f"  (length differs: {len(dw)} vs {len(base)})")
