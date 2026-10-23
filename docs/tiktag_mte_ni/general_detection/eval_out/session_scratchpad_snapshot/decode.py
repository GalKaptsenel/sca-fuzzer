import struct
D="/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/VERIFIED_GENUINE/fuzzer_072421"
bins={n:open(f"{D}/enacted_test{n}.bin","rb").read() for n in ["0001","0009","0017","0025"]}
for n,b in bins.items(): print(n,"len",len(b))
def words(b): return [struct.unpack_from("<I",b,i)[0] for i in range(0,len(b),4)]
W={n:words(b) for n,b in bins.items()}
base=W["0001"]
print("\n=== per-word diffs vs baseline 0001 (word index = instr #, byte off = idx*4) ===")
for n in ["0009","0017","0025"]:
    diffs=[(i, base[i], W[n][i]) for i in range(len(base)) if base[i]!=W[n][i]]
    print(f" 0001 vs {n}: "+", ".join(f"instr{i}(off0x{i*4:x}): {a:#010x}->{c:#010x}" for i,a,c in diffs))
# Decode ADDG/SUBG fields for the differing words. ADDG: sf=1 op=0b100110 -> top bits 1001 0001 10 ...
# ADDG encoding: 1 0 0 1 0 0 0 1 1 0 | uimm6(16:16? ) ... Actually ADDG: 
#  bits31-22: 1001000110 ; imm6 at 21:16 (<<4 offset); op3 15:14=00 ; uimm4 tag at 13:10 ; Rn 9:5 ; Rd 4:0
def decode_addg(w):
    top=(w>>22)&0x3ff
    if top==0b1001000110: kind="ADDG"
    elif top==0b1101000110: kind="SUBG"
    else: return None
    imm6=(w>>16)&0x3f; uimm4=(w>>10)&0xf; Rn=(w>>5)&0x1f; Rd=w&0x1f
    return f"{kind} x{Rd}, x{Rn}, #{imm6*16}, #{uimm4}(tag)"
print("\n=== decode differing words (both baseline & variant) ===")
allidx=set()
for n in ["0009","0017","0025"]:
    for i in range(len(base)):
        if base[i]!=W[n][i]: allidx.add(i)
for i in sorted(allidx):
    print(f" instr{i} off0x{i*4:x}:")
    for n in ["0001","0009","0017","0025"]:
        w=W[n][i]; dec=decode_addg(w)
        print(f"    {n}: {w:#010x}  {dec if dec else '(not ADDG/SUBG)'}")
