CE + by-hand seal x window x granule x set (input1 genuine, base=0xffffff8869761000):
  A @0xf8 x4 LDR[x4]@0xfc : W1 g189/set47  ; W2 g0/set0   (W2: EOR w4,w0,w0 in bb_0.4 zeroes x4)
  B @0x10c+0x114 x1 STR[x1]@0x110 : W1 g0/set0 ; W2 g0/set0
  D @0x120 x2 LDP[x2]@0x124 : W1 g287/set7 ; W2 g305/set12  (x2=w2 from LDR[x4]; 189->0x91fa->g287 ; g0->0xf31d->g305)
CE arch sets {3,7,21,23,63}; spec-only {0,12,47}. HW leak block {1,4,5,6,8,9,10,11} OUTSIDE both.
genuine ptr-tag delta=7 all seals (base_tag=15, (15+7)%16=6=memtag). ADDG uimm4=bits[13:10].
