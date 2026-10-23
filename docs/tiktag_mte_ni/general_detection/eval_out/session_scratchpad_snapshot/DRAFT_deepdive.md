# DEEPDIVE checkpoint (draft)

## Static (enacted_test0001.bin, 76 instr, no header)
Sealer pattern per access: `and xN,#0x1ff0 ; add xN,x29 ; [adjust] ; addg xN,#uimm4 ; <access>`; x29=sandbox base.
BBs: 0.0(0x00) 0.1(0x08) 0.2(0x10-0x34 STR w5,[x2,x1]) 0.3(0x38 CSINV; 0x3c CBNZ w3; 0x40 b.nv->0.7)
 0.4(0x44-0x70 STP,SUBG x1,STZG) 0.5(0x74) 0.6(0x78-0xc0 STR x5,STR w3,STZ2G,STR w5,0xbc TBZ x0#58,0xc0 b.al exit)
 0.7(0xc4-0xec CINV,LDG x5,[x1],LDG x1,[x3],UDIV,EOR) 0.8(0xf0-0x124 A+LDR,B+STR,D+LDP) exit(0x128)
Windows: W1=CBNZ@0x3c mispredict not-taken -> bb0.7/0.8 before bb0.4/0.6 stores. W2=TBZ@0xbc taken-spec -> bb0.7/0.8 after.
Seals (genuine uimm4=7 => ptag (15+7)%16=6=memtag):
 A @0xf8 addg x4 -> LDR w2,[x4]@0xfc
 B @0x10c(+0x114 restore) addg x1 -> STR x2,[x1]@0x110
 D @0x120 addg x2 -> LDP w2,w1,[x2]@0x124
4 enacted differ ONLY in uimm4 at 0xf8/0x10c/0x114/0x120 (cmp confirms):
 0001 A7 B(7,0) D7 = all-match baseline
 0009 A13 B(2,5) D7
 0017 A4  B(7,0) D4
 0025 A7  B(14,9) D7

## CE + by-hand (input1 genuine, base 0xffffff8869761000) — AGREE EXACTLY
 seal | W1 (CBNZ fallthru)      | W2 (TBZ taken)
 A    | g189 set47              | g0 set0   (W2: EOR w4,w0,w0 in bb0.4 zeroes x4)
 B    | g0 set0                 | g0 set0
 D    | g287 set7               | g305 set12  (x2=w2 from LDR[x4]; g189->0x91fa->g287 ; g0->0xf31d->g305)
CE arch sets {3,7,21,23,63}; spec-only {0,12,47}. HW leak block OUTSIDE both => genuine.
ADDG uimm4=bits[13:10]. pin uimm4=0 => ptag15 => TCMA-unchecked (check OFF).

## HW per-seal 16-delta sweeps (isolated; prefix=14 reps=2000; memtag6 u6=0)
 A @0xf8: probe set1; MATCH uimm4=7(ptag6); present-on-mismatch; ptag15 UNCHK, ptag0 CHECKED; W1 sets44-47 ~dark.
 B @0x10c: probes set1 AND set4; MATCH uimm4=7; present-on-mismatch; ptag15 UNCHK.
 D @0x120: probes set5,9,11 (+set12 anchor always lit); MATCH uimm4=7; present-on-mismatch; wide [2..63] n~28.
Shared: match delta 7, polarity present-on-mismatch, only ptag15 unchecked. A&B overlap set1 (shared granule0).

## HW granule localization
 A (set1,gate0xf8): only granule 0 retag lifts d7 suppression (0.00->0.87). g189/188/190/4/1 do NOT. => A checks granule 0 (W2). d10 stays lit due to B confound (shared g0).

## Tracking (match delta moves as (C+T-cell_tag)%16)
 D: DONE prior (TRACK2D set9 gran305): T3->d4,T6->d7,T9->d10,T12->d13 all track; gran287 does NOT (W1 inactive); gran255 confound tag-indep.
 A: track_pin running (B pinned off)
 B: pending (set4, A pinned off)

## Pending
 anchor_sweep A/B/D (stream anchored to access?); allsets combos A/B/D/AB/AD/BD/ABD; windows W1 vs W2 independence.
