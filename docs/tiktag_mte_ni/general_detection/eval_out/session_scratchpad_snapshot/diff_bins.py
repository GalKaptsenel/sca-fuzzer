#!/usr/bin/env python3
"""Word-by-word diff of enacted test binaries + decode ADDG tag fields. Pure local file read."""
import struct, sys, os
VD = sys.argv[1]
def load(n):
    with open(f"{VD}/enacted_test{n:04d}.bin","rb") as f: b=f.read()
    return b
def words(b):
    return [struct.unpack_from("<I", b, i)[0] for i in range(0, len(b)-len(b)%4, 4)]
def decode_addg_tag(w):
    # ADDG: sf=1 ... opcode 100011 0 ; tag offset uimm4 at bits [13:10], uimm6 at [21:16]
    if (w >> 22) == 0x246:  # 0b1001000110 top10 => ADDG (0x91800000 base)
        return (w>>10)&0xf, (w>>16)&0x3f
    return None
base = load(0)
bw = words(base)
print(f"enacted_test0000.bin: {len(base)} bytes, {len(bw)} words")
for n in (8,16,24):
    ow = words(load(n))
    print(f"\n=== diff 0000 vs {n:04d} ===")
    ndiff=0
    for i,(a,b) in enumerate(zip(bw,ow)):
        if a!=b:
            ndiff+=1
            ta=decode_addg_tag(a); tb=decode_addg_tag(b)
            note=""
            if ta and tb: note=f"  ADDG tagoff {ta[0]}->{tb[0]} (uimm6={ta[1]})"
            print(f"  word[{i}] off=0x{i*4:x}: 0x{a:08x} -> 0x{b:08x}{note}")
    print(f"  total differing words: {ndiff}")
# Also list ALL ADDG words in baseline with offsets
print("\n=== all ADDG words in enacted_test0000 ===")
for i,w in enumerate(bw):
    t=decode_addg_tag(w)
    if t is not None:
        print(f"  off=0x{i*4:x} word[{i}]=0x{w:08x}  ADDG tagoff={t[0]} uimm6={t[1]}")
