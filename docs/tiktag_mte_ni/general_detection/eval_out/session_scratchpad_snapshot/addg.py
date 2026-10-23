import struct
base=open('enacted_test0001.bin','rb').read()
def word(b,off): return struct.unpack('<I',b[off:off+4])[0]
def decode(w):
    op=(w>>22)&0x3ff  # ADDG: sf=1,op=0... check bits
    # ADDG (imm) encoding: 1 0 0 1 0 0 0 1 1 0 uimm6(21:16) (0)(0) uimm4(13:10) Xn(9:5) Xd(4:0)
    top=(w>>23)&0x1ff
    uimm6=(w>>16)&0x3f
    uimm4=(w>>10)&0xf
    xn=(w>>5)&0x1f; xd=w&0x1f
    is_addg = (w>>22)==(0x91800000>>22) or ((w & 0xFFC0C000)==0x91800000)
    return dict(top=hex(top),uimm6=uimm6,uimm4_tag=uimm4,xn=xn,xd=xd,is_addg=((w&0xFFE0C000)==0x91800000), raw=hex(w))
for off,pc in [(0xF8,'0xF8 seal-x4(LDR)'),(0x10C,'0x10C seal-x1(STR)'),(0x114,'0x114 retag-x1'),(0x120,'0x120 seal-x2(LDP GATE)')]:
    wb=word(base,off)
    print(f"\nPC {pc}: baseline word={hex(wb)} -> {decode(wb)}")
    for f in ['enacted_test0009.bin','enacted_test0017.bin','enacted_test0025.bin']:
        d=open(f,'rb').read(); wd=word(d,off)
        if wd!=wb:
            print(f"   {f}: {hex(wd)} tag {decode(wb)['uimm4_tag']}->{decode(wd)['uimm4_tag']}  (only-tag-differs={ (wb & ~0x3C00)==(wd & ~0x3C00) })")
