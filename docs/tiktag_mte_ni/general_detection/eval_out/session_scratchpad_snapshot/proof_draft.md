# Mechanism proof — violation fuzzer_072421 (DRAFT DATA)

Device: Pixel 8, Cortex-X3 core 8 (MIDR 0x411fd4e0), module md5 5c8dfec, F+R, view_rotation ON.
Protocol: fixed 14-input prefix of genuine variants, suspect #1 at the tail slot, only the suspect
seal's ADDG uimm4 changed. pfc[0]=L2D_CACHE_REFILL, pfc[1]=MEM_ACCESS (demand), pfc[2]=BR_MIS_PRED_RETIRED.
memtag 6, genuine uimm4 7, base_ptr_tag 15, ptag(u4)=(15+u4)%16. MATCH at ptag6 (u4 7), TCMA1-unchecked
at ptag15 (u4 0). Seals A@0xf8 (x4), B@0x10c (x1), D@0x120 (x2).

## Proof 1a — PMU differential (3 reps x 2000)

Seal D @0x120, probes 1,3,4,9,12,20,28,40:
u4=7 MATCH : narrow n=9, set3=1.0 set12=1.0 rest cold; L2D 8.7, MEM 12.0, BRMIS 5.0, excess -3.3
u4=0 UNCHK : same as MATCH (L2D 8.5, MEM 12.0, BRMIS 5.0)
u4=8 MISMAT: wide n=28 [2..63], set3=0.0 set4/9/12/20=1.0 set28~0.7; L2D 91.5, MEM 54273, BRMIS 104, excess -54182

Seal B @0x10c mismatch: n=17, L2D 86, MEM 54107, BRMIS 102.
Seal A @0xf8 mismatch: n=17, L2D 83, MEM 54549, BRMIS 108.

MATCH MEM_ACCESS=12 equals the test case's architectural memory-op count -> counter mapping validated and
the bb_0.7/0.8 block does not execute on a match. On a mismatch MEM_ACCESS (demand) rises ~4500x while
L2D_CACHE_REFILL rises ~10x, and the ~54k is nearly constant across the three seals and both block widths.
A hardware prefetch stream raises refills without raising demand accesses. The measured pattern is the
opposite: a large demand/speculative access burst, most of it L1-resident (few L2 refills). PMU event 0x13
alone does not establish "prefetcher"; it establishes a speculative access burst coincident with the block.

## Proof 2 — set 3 direction + cause

(i) 16-delta seal-D sweep (3 reps x 1500), set 3:
set3 resident (1.0) ONLY at ptag6 (u4 7, match) and ptag15 (u4 0, TCMA1-unchecked). set3 evicted (0.0) at
the 14 genuine mismatches. set3 eviction coincides 1:1 with wide block (n=28) and MEM~54k. At the two
resident deltas the footprint is n=9 and MEM=12. Direction: resident at match, evicted at mismatch.

(ii) granule_scan (retag one granule to T=9, read set 3 at ptag6 vs ptag9):
granule 305 (seal-D LDP target): set3 0.00 at ptag6, 1.00 at ptag9 -> set-3 state moves WITH seal-D's
checked granule. granule 15 (set-3's own STP target): set3 stays resident, no seal-D-like move. So set 3 is
gated by seal-D's tag-check on granule 305, not by its own granule. set 3 is collateral.

## Proof 3 — window dependence (REFUTED as posed)

view_rotation OFF (enable_pre_run_flush=0, confirmed on device), seal D:
u4=7 MATCH : n=9, MEM 12, BRMIS 0.7
u4=8 MISMAT: n=28, MEM 54k, BRMIS ~80  (UNCHANGED from window ON)
view_rotation OFF + per-input BPU mistraining stripped: mismatch still n=28, MEM 54k.
notrain-only (view_rotation ON): mismatch still n=28, MEM 54k.

Toggling view_rotation does not remove the divergence. The bb_0.7/0.8 window is present on both match and
mismatch (match MEM=12 shows bb_0.7/0.8 never retire). The trigger is the tag mismatch inside the
speculative window, not the view rotation. The only window-knob effect is match-case BR_MIS_PRED 5.0 -> 0.7.

## Proof 1b — prefetcher disable (status)
X3 data prefetcher control = CPUACTLR6_EL1 (S3_0_C15_C8_1) bit 41 (NOT CPUECTLR_EL1). [status pending]
