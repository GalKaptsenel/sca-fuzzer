# By-hand emulation of the seal A->D chain from input1 seed bytes + asm, no device, no CE.
base=0  # relative; granule = (val & 0x1ff0)//16
x4e=0x6bd2; x2e=0xc1d8
# granule memory (low 32-bit word at granule g) from input1 dump:
memword={189:0x000091fa, 0:0x0000f31d}
def mask(v): return v & 0x1ff0
print("SEAL A LDR[x4] @0xfc:")
# W1: x4 = entry 0x6bd2 (bb_0.4 skipped, so EOR w4 did NOT run)
a1=mask(x4e); print(f"  W1: x4&0x1ff0=0x{a1:x} -> granule {a1//16} set {((a1>>6)&63)}")
# W2: bb_0.4 ran -> 0x48 eor w4,w0,w0 => x4=0
a2=mask(0);  print(f"  W2: x4=0 (EOR w4,w0,w0 in bb_0.4) -> granule {a2//16} set {((a2>>6)&63)}")
print("SEAL D LDP[x2] @0x124 (x2 = w2 loaded by LDR[x4] at 0xfc):")
w2_w1=memword[189]; d1=mask(w2_w1); print(f"  W1: w2=mem[g189]=0x{w2_w1:x}; &0x1ff0=0x{d1:x} -> granule {d1//16} set {((d1>>6)&63)}")
w2_w2=memword[0];   d2=mask(w2_w2); print(f"  W2: w2=mem[g0]=0x{w2_w2:x}; &0x1ff0=0x{d2:x} -> granule {d2//16} set {((d2>>6)&63)}")
