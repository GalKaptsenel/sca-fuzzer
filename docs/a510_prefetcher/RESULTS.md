# Cortex-A510 L1 prefetcher — investigation results (Pixel 8, physical)

Scripts + full environment setup: `REPRODUCE.md` (same directory). Originally written to
`violations_fr/violation-260716-181940/`, which is gitignored — this is the tracked copy.

Physical Pixel 8 (shiba/Tensor G3), core 0 = Cortex-A510 (in-order little). F+R, 300 reps/point.
Violation `violation-260716-181940`, input #9. HTrace.raw bit b == cache set b (identity).

## Method
The victim's only prefetch-triggering access is `ldp w3, w5, [x3, #0x2c]`. Its accessed sandbox offset is
`x3 & 0x1fff` (main region 0x000-0xfff, faulty region 0x1000-0x1fff). We force that offset to an arbitrary
value V by byte-patching the sandbox mask instruction (dyn idx 19, `and x3,x3,#0x1fff`) to `movz x3,#V`
in the assembled test case, then measure input #9 on the pinned A510. Everything else (registers, the first
load, all memory) is #9's; only the ldp's demand address changes. `background` sets [15, 23, 31] are constant
across all V (the first load `ldr`@set23 and its ±8 A510 artifacts) and are excluded from the "prefetched" column.

## Key findings
- The cached (prefetched) block **tracks the demand address**: as the demand set increments, the prefetched
  set(s) shift by the same amount (fixed Δset) — the defining fingerprint of a hardware prefetcher.
- **Region-gated**: faulty-region demands stream; main-region demands do not (see the two tables).
- #9's ldp lands in **faulty** (0x1002) -> streams into sets 2-6; #28's lands in **main** (0x0) -> nothing.
  That page difference (abstracted away by the cache-set contract) is the entire violation.

## Faulty region — demand sets 0..23 (V = 0x1000 + set*0x40)
| V (offset) | region | demand set | lit sets (>0.5, ex. bg 15/23/31) | prefetched Δset from demand |
|---|---|---|---|---|
| 0x1000 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |
| 0x1040 | faulty | 1 | [17, 25, 33, 41, 49] | [16, 24, 32, 40, 48] |
| 0x1080 | faulty | 2 | [18, 26, 34, 42, 50] | [16, 24, 32, 40, 48] |
| 0x10c0 | faulty | 3 | [19, 27, 35, 43, 51] | [16, 24, 32, 40, 48] |
| 0x1100 | faulty | 4 | [20, 28, 36, 44, 52] | [16, 24, 32, 40, 48] |
| 0x1140 | faulty | 5 | [21, 29, 37, 45, 53] | [16, 24, 32, 40, 48] |
| 0x1180 | faulty | 6 | [22, 30, 38, 46, 54] | [16, 24, 32, 40, 48] |
| 0x11c0 | faulty | 7 | [39, 47, 55] | [32, 40, 48] |
| 0x1200 | faulty | 8 | [] | [] |
| 0x1240 | faulty | 9 | [] | [] |
| 0x1280 | faulty | 10 | [] | [] |
| 0x12c0 | faulty | 11 | [] | [] |
| 0x1300 | faulty | 12 | [] | [] |
| 0x1340 | faulty | 13 | [] | [] |
| 0x1380 | faulty | 14 | [] | [] |
| 0x13c0 | faulty | 15 | [] | [] |
| 0x1400 | faulty | 16 | [] | [] |
| 0x1440 | faulty | 17 | [] | [] |
| 0x1480 | faulty | 18 | [] | [] |
| 0x14c0 | faulty | 19 | [] | [] |
| 0x1500 | faulty | 20 | [] | [] |
| 0x1540 | faulty | 21 | [] | [] |
| 0x1580 | faulty | 22 | [] | [] |
| 0x15c0 | faulty | 23 | [] | [] |

## Faulty region — low-bit variants near #9 (demand set 0)
| V (offset) | region | demand set | lit sets (>0.5, ex. bg 15/23/31) | prefetched Δset from demand |
|---|---|---|---|---|
| 0x1000 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |
| 0x1001 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |
| 0x1002 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |
| 0x1004 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |
| 0x1008 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |
| 0x1010 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |
| 0x1020 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |
| 0x1030 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |

## Main region — demand sets 0..15 (V = set*0x40)  [no streaming]
| V (offset) | region | demand set | lit sets (>0.5, ex. bg 15/23/31) | prefetched Δset from demand |
|---|---|---|---|---|
| 0x0000 | main | 0 | [] | [] |
| 0x0040 | main | 1 | [] | [] |
| 0x0080 | main | 2 | [] | [] |
| 0x00c0 | main | 3 | [] | [] |
| 0x0100 | main | 4 | [] | [] |
| 0x0140 | main | 5 | [] | [] |
| 0x0180 | main | 6 | [] | [] |
| 0x01c0 | main | 7 | [] | [] |
| 0x0200 | main | 8 | [] | [] |
| 0x0240 | main | 9 | [] | [] |
| 0x0280 | main | 10 | [] | [] |
| 0x02c0 | main | 11 | [] | [] |
| 0x0300 | main | 12 | [] | [] |
| 0x0340 | main | 13 | [] | [] |
| 0x0380 | main | 14 | [] | [] |
| 0x03c0 | main | 15 | [] | [] |

## #9 vs #28 exact addresses
| V (offset) | region | demand set | lit sets (>0.5, ex. bg 15/23/31) | prefetched Δset from demand |
|---|---|---|---|---|
| 0x1002 | faulty | 0 | [2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] |
| 0x0000 | main | 0 | [] | [] |

---

## Trigger instruction — ldp vs ldr vs str (F+R, 300 reps, #9 on A510)
Replaced the `ldp` (dyn idx 22) in place with a single load/store at the same address:

| instruction at the trigger site | block 2-6 |
|---|---|
| `ldp w3, w5, [x3, #0x2c]` (baseline) | 0.97 |
| `ldr w3, [x3, #0x2c]` (single 32-bit load) | 0.94 |
| `ldr x3, [x3, #0x28]` (single 64-bit load) | 0.93 |
| `str w5, [x3, #0x2c]` (a **store**) | 0.95 |

=> The pair nature is irrelevant, and it is not even load-specific: **any demand access (load or store) to that
faulty address triggers the prefetch.**

## Page-table attributes — main vs faulty region (`/sys/executor/pte`)
```
main   ...191000  PTE=0x00680008e2191707  AttrIdx=1 AP=0 SH=3 AF=1 nG=0 DBM=1 Cont=0 PXN=1 UXN=1
faulty ...192000  PTE=0x00680008e2192707  AttrIdx=1 AP=0 SH=3 AF=1 nG=0 DBM=1 Cont=0 PXN=1 UXN=1
xor(main^faulty) = 0x0000000000003000     # only the physical frame (bits 12-13) — i.e. just the address
```
The two pages are **attribute-identical** (same Normal-cacheable memory type, shareability, permissions, AF, XN).
So "faulty streams, main doesn't" is **positional, not a page-attribute difference.**

## Per-attribute PTE-bit flip on the faulty page (F+R, 300 reps, #9 on A510)
Each bit XORed into the faulty page's leaf PTE only during the measured run (save/restore + TLBI):

| flipped attribute | block 2-6 |
|---|---|
| baseline (no flip) | 0.97 |
| SH (bits 8,9) | 0.92 |
| nG (bit 11) | 0.96 |
| DBM (bit 51) | 0.94 |
| Cont (bit 52) | 0.97 |
| PXN (bit 53) | 0.94 |
| UXN (bit 54) | 0.97 |
| PBHA (bits 59-62) | 0.95 |

=> **No safe attribute bit gates the prefetcher** (AttrIdx/AF/AP not tested — they would fault the input-load write).
Consistent with the identical-PTE finding: the main-vs-faulty behavior is positional/contention, not attributes.

## Interpretation
- The cached block **tracks the demand address** (Δset shifts +1 with the demand for sets 1-6) ⇒ a **hardware
  prefetcher**, not an architectural artifact.
- Fixed footprint: demand set 0 → sets 2-6; demand sets 1-6 → demand+{16,24,32,40,48} sets (= +1KB, +1.5KB, +2KB,
  +2.5KB, +3KB; +512B stride) ⇒ an **offset/spatial prefetcher** (matches the A510's documented L2 offset prefetcher).
- Fires only for a demand in the **first 512 B (sets 0-7)** of the faulty page; demand sets ≥8 and **all** main-region
  demands show nothing.
- A510-only (OoO A715/X3 clean); address-driven (value-swap); memory-content-independent (zero-except-reads);
  triggered by any single load/store; unaffected by DSB/DMB/ISB and by every safe PTE-attribute flip.
- Likely the **main** page shows nothing because the measurement harness churns it (evicting its prefetched lines),
  while the **faulty** page is quiet enough for the prefetched lines to survive to the F+R probe — positional, not
  architectural. #9's `ldp` landed in the faulty page's trigger zone; #28's landed in main. That address difference,
  abstracted away by the cache-set contract, is the entire "violation."

---

## Address-bit sensitivity — flip one bit of the accessed offset (F+R, 300 reps, A510)
Base demand offset V0 = 0x1080 (faulty page, cache set 2 -> prefetch 18,26,34,42,50, stride 8). Each row
flips a single bit of V (via the `movz x3,#V` patch); nothing else changes. Background sets {15,23,31} and
the demand set itself are excluded from "prefetched".

| flipped bit | V | region/set | prefetched sets | stride | note |
|---|---|---|---|---|---|
| — (base) | 0x1080 | faulty set2 | 18,26,34,42,50 | 8 | 5 lines |
| bit0..bit5 (0x01–0x20) | 0x1081..0x10a0 | faulty set2 | 18,26,34,42,50 | 8 | no change (within-line) |
| bit6 (0x40) | 0x10c0 | faulty set3 | 19,27,35,43,51 | 8 | prefetch shifts +1 |
| bit7 (0x80) | 0x1000 | faulty set0 | 2,3,4,5,6 | 1 | set-0 contiguous pattern |
| bit8 (0x100) | 0x1180 | faulty set6 | 22,30,38,46,54 | 8 | prefetch shifts +4 |
| bit9 (0x200) | 0x1280 | faulty set10 | (none) | — | OFF — demand set ≥8 |
| bit10 (0x400) | 0x1480 | faulty set18 | (none) | — | OFF — demand set ≥8 |
| bit11 (0x800) | 0x1880 | faulty set34 | (none) | — | OFF — demand set ≥8 |
| bit12 (0x1000) | 0x0080 | main set2 | (none) | — | OFF — main page |

### Address-bit roles
- bits [5:0] — within a 64-byte line: **irrelevant** (prefetcher is line-granular).
- bits [8:6] — cache-set bits inside the trigger zone: **steer** the prefetch; the constellation tracks the
  demand set 1-for-1 (set0 = contiguous 2-6; sets 1-7 = demand+{16,24,32,40,48}).
- bits [11:9] — **gate** the trigger zone: any set -> demand cache set ≥8 -> prefetch OFF.
- bit 12 — **gates the page**: faulty=on, main=off.
- Trigger condition: page==faulty AND offset[11:9]==0 (first 8 lines); then offset[8:6] picks the demand line.

### NOTE on the two reboots
The reboots were NOT from address flips — they were from PTE-attribute flips that fault: writing CPUECTLR_EL1
(firmware-trapped) and setting PTE AP[1] (EL0-accessible -> PAN fault when the kernel accesses the page). Address
(offset) flips stay within the sandbox [0,0x1fff] and never fault.

---

## Address-bit flip from multiple bases (F+R, 200 reps, A510) — generalization
Same single-address-bit flip repeated from three bases. Roles hold everywhere.

**base 0x1002 (faulty set0, prefetch 2-6):** bits0-5 no change; bit6->set1; bit7->set2 (18,26,34,42,50);
bit8->set4 (20,28,36,44,52); bits9-11 OFF (set 8/16/32); bit12 OFF (main).

**base 0x1140 (faulty set5, prefetch 21,29,37,45,53):** bits0-5 no change; bit6->set4; bit7->set7 (39,47,55);
bit8->set1; bits9-11 OFF (set 13/21/37); bit12 OFF (main).

**base 0x1300 (faulty set12, OUT of zone, prefetch NONE):** bits0-8 stay OFF (demand set stays ≥8), EXCEPT
**bit9 flip -> 0x1100 = set4 -> prefetch ON (20,28,36,44,52)**. bits10,11 OFF; bit12 OFF. => bits[11:9] gate the
zone in BOTH directions (set them=off, clear them=on).

### Complete generalized address map (demand offset, 13 bits; page = bit12)
| offset bits | role | behavior |
|---|---|---|
| [5:0] | within a 64B line | irrelevant (line-granular) |
| [8:6] | demand line in the zone | steer — prefetch tracks the demand set 1-for-1 |
| [11:9] | trigger-zone gate | must be 000 (set<8); any bit set -> prefetch OFF |
| [12] | page select | faulty=on, main=off |

Constellation by demand set (consistent across bases): set0 -> contiguous 2-6; sets2-6 -> demand+{16,24,32,40,48}
(+8 stride); set7 -> truncated (page runs out); set≥8 -> none.

---

## Multi-bit address combinations & all four sandbox pages (F+R, 200 reps, A510)

### Multiple address bits set at once (faulty base 0x1000)
- Steer bits [8:6] compose as place value -> select demand set (0-7); prefetch tracks. b6->set1, b7->set2,
  b6+7->set3, b8->set4, b6+8->set5, b7+8->set6, b6+7+8->set7.
- Gate bits [11:9] dominate: adding ANY -> OFF regardless of steer (b6+7+8+b9 -> set15 OFF).
- Within-line bits [5:0] are inert UNLESS the 8-byte access straddles a line boundary, then BOTH lines fire:
  V=0x10bf (set2->3 straddle) -> {18,19,26,27,34,35,42,43,50,51}; V=0x11ff (set7->8) -> set7 prefetch + set8 line.
- Page bit gates: main+b7 OFF, faulty+b7 ON.

### All four sandbox pages, demand sets 0..8 (x29 = main base)
| page | offset from x29 | prefetch |
|---|---|---|
| eviction (behind main) | -0x1000 | none, except a chaotic 24-set smear at set4 (P+P prime-buffer artifact) |
| main | 0 | none (any set) |
| faulty | +0x1000 | sets 0-7 stream (as mapped above); set≥8 none |
| overflow | +0x2000 | none (any set) |

Main-page bit-flip from 0x0080: every bit OFF except bit12 (-> 0x1080 faulty) -> ON. Confirms the page bit gates.

### Refined conclusion — it is the FAULTY page specifically
The prefetch is NOT explained by "quiet vs harness-churned page" (overflow is equally quiet yet silent) nor by
page attributes (main/faulty PTEs are attribute-identical). Only the page **+1 forward of x29/main** prefetches.
Since the harness always accesses the main page (input load) before the test case, a demand one page ahead
(faulty) matches a **forward next-page / +1-page-stride** prefetch trigger; same-page (main), +2 (overflow), and
-1 (eviction) do not. #9's ldp happened to land in that +1 page (faulty); #28's landed in main -> no trigger.

---

## Absolute-page vs relative-offset — shift x29 inside the test case (F+R, 200 reps, A510)
Prepended `add x29,x29,#0x1000` (restored with `sub` before measurement_end) and removed the first load, so
only the `ldp` accesses memory. Then swept its offset V.

| x29 base | ldp V | absolute page hit | prefetch |
|---|---|---|---|
| main (no shift) | 0x0080 | main s2 | none |
| main (no shift) | 0x1080 | faulty s2 | 18,26,34,42,50 |
| faulty (x29+=0x1000) | 0x0080 | faulty s2 | 18,26,34,42,50 |
| faulty (x29+=0x1000) | 0x1080 | overflow s2 | none |

Shifting x29 up one page makes the prefetch fire at V=0x0080 (not 0x1080) — it MOVES with the base to keep the
demand on the **absolute faulty page**. => the trigger is locked to the absolute faulty-page address, not the
offset value and not a "2nd-page-of-sandbox" role. (Pitfall: modifying x29 in the body WITHOUT restoring it before
measurement_end corrupts the harness's post-body base -> panic/reboot; restore it.)

### Open question
Why that one page (faulty = x29+0x1000) and not main/overflow/eviction, given identical PTE attributes? The
sandbox is vmalloc-backed (virtually contiguous, physically scattered), so the faulty page's *physical* frame may
have an alignment the others lack, or the harness's fixed access pattern trains the prefetcher for that page
specifically. Not resolved.

---

## ~~MAJOR CORRECTION: the violation is an F+R HARNESS ARTIFACT~~ — RETRACTED 2026-07-17
**This section was WRONG. Kept for the record; superseded by the two sections below.**

It claimed #9 and #28 become identical once flush+reload walks all four pages, hence "FALSE POSITIVE of the
incomplete F+R flush/reload". That rested on ONE run of an early all-pages probe. The model behind it — "the
reload's forward-most region is the one that gets primed" — has since been refuted twice on hardware:
  - a backward 2-page walk was predicted to clean/flip the trace; it flooded instead.
  - prepending lower_overflow was predicted to prime main; it did not.
The controlled re-measurement below contradicts the claim directly.

## 4-region harness — the asymmetry SURVIVES (2026-07-17, 300 reps, A510)
`flush()`/`reload()` extended to walk lower_overflow -> main -> faulty -> upper_overflow, recording htrace bits
ONLY for main+faulty (templates_jit.c, TEMP(lmfu)). This puts main and faulty in the SAME structural position:
neither is forward-most, each has a neighbour on both sides. If walk position were the mechanism, the
difference had to collapse. It did not.

| harness | #9 (ldp->faulty) block2-6 | #28 (ldp->main) block2-6 | #28 lit sets |
|---|---|---|---|
| 2-page (main->faulty), original      | lit  | **0.00** fully dark | [15,23,31] |
| 3-page (lower->main->faulty)         | 0.93 | **0.67** lit        | [0,2,3,4,5,6,15,23,31] |
| 4-region (lower->main->faulty->upper)| 0.89 | **0.48** below thr. | [0,15,23,31] |

- **#9 is robust** (0.89-0.93 everywhere): a faulty-page demand reliably streams.
- **#28 is marginal** (0.00 / 0.67 / 0.48): main-page prefetch sits AT the detection threshold and the harness
  pushes it around. Main is not inert — it is weakly and unstably prefetched.
- => the main-vs-faulty asymmetry is NOT a walk-position artifact. Why an identical access streams in faulty
  but only marginally in main remains **UNEXPLAINED** (PTE attributes are byte-identical, page-relative geometry
  identical). Untested lead: `get_stack_base_address()` = main_region+4096, so the TC's stack traffic writes into
  main's high offsets but never into faulty — asymmetric traffic hitting only main.

## Input-driven probe + prefetched-line knockout (2026-07-17, 100 reps, A510)
Scripts: `scripts/fast_sweep.py` (batched: one super-batch = one round-trip), `scripts/knockout.py`,
`scripts/mkreif.py`. A hand-assembled TC is injected via the `_sandboxed_cache` seam because `dc civac` is not
in the fuzzer ISA. Minimal victim, x1 supplied ONLY by the .reif input:

    and x1, x1, #0x1fff ; add x0, x29, x1 ; ldr x3, [x0] ; dc civac, x0 ; dsb ish ; isb

The demand line is flushed by the TC itself, so every remaining lit set is prefetcher-filled. For x1=0x1000 the
code touches EXACTLY ONE line (faulty+0x000, set 0) and five untouched lines light up (sets 2-6).

### EXP1 — with vs without the demand-line flush (same inputs; bg sets 15/23/31 excluded)
| x1 | demand set | WITHOUT flush (demand+prefetch) | WITH flush (prefetch only) | delta |
|---|---|---|---|---|
| 0x1000 | 0 | [0, 2, 3, 4, 5, 6]         | [2, 3, 4, 5, 6]         | -[0] |
| 0x1001 | 0 | [0, 2, 3, 4, 5, 6]         | [2, 3, 4, 5, 6]         | -[0] |
| 0x1020 | 0 | [0, 2, 3, 4, 5, 6]         | [2, 3, 4, 5, 6]         | -[0] |
| 0x1040 | 1 | [1, 17, 25, 33, 49]        | [13, 17, 25]            | -[1,33,49] **UNSTABLE** |
| 0x1080 | 2 | [2, 18, 26, 34, 42, 50]    | [18, 26, 34, 42, 50]    | -[2] |
| 0x10c0 | 3 | [3, 19, 27, 35, 43, 51]    | [19, 27, 35, 43, 51]    | -[3] |
| 0x1100 | 4 | [4, 20, 28, 36, 44, 52]    | [20, 28, 36, 44, 52]    | -[4] |

- The flush removes **exactly the demand set and nothing else** (6/7 points) => demand line and prefetched lines
  are cleanly separable; F+R reads real residency.
- **bits[5:0] of x1 are irrelevant** (0x1000/0x1001/0x1020 identical — same 64B line); **from bit 6 up the
  footprint moves** => two inputs differing only in low address bits produce different cache footprints.
- **demand set 0 is special**: prefetch is +2..+6 (stride 1, skipping +1). Every other demand set gives
  +16,+24,+32,+40,+48 (stride 8). #9's access lands exactly at set 0. Unexplained.
- `x1=0x1040` (demand set 1) is **UNSTABLE** across runs ([13] / [13,17,25] / [17,25,33,41,49]) — do not trust it.

### *** EXP3 — BATCH NEIGHBOURS CONTAMINATE THE RESULT (2026-07-17, 100 reps) — supersedes EXP1-main ***
`scripts/isolate.py`. The same address, measured ALONE in its own super-batch vs GROUPED with other inputs:

| x1 | page | demand set | alone (3 trials) | grouped (4 inputs) |
|---|---|---|---|---|
| 0x0000 | main   | 0 | [2,3,4,5,6] x3 | **[]** |
| 0x1000 | faulty | 0 | [2,3,4,5,6] x3 | [2,3,4,5,6] |
| 0x0080 | main   | 2 | [18,26,34,42,50] / [18,26] / [18,26,34,42,50] | **[18,26,34,42,50]** |
| 0x1080 | faulty | 2 | [18,26,34,42,50] x3 | [18,26,34,42,50] |

**Measured alone, MAIN and FAULTY behave IDENTICALLY.** Main set 2 streams the exact faulty footprint, though
the 7-input batch in EXP1-main reported NO prefetch for it; main set 0 fires reliably alone but goes dark when
grouped. Faulty is stable in these trials; main is the one that flips.

=> "faulty streams, main does not" is **NOT a page property**. It is contamination from **batch neighbours**:
the prefetcher's training state carries ACROSS inputs within a super-batch, and every main measurement happened
to sit in different company than every faulty measurement.

**This confirms the original day-one hypothesis** (#9 vs #28 differ "due to their predecessors"; priming may have
compared the wrong pairs) — which had been dismissed in favour of walk-position and page-attribute models that
hardware later refuted. It also puts the same confound on the 4-region #9=0.89/#28=0.48 run above: both came
from ONE 29-input batch, so #9 and #28 had different predecessors.

Caveats: main set 2 gave a partial [18,26] in trial 2, so some instability is real and not purely batch-order;
and WHICH neighbour property matters (position, page, address distance) is not yet established.

**Decisive next test:** measure #9 and #28 each ALONE in its own batch with identical predecessors. If they
then agree, the violation is a false positive and the mechanism is predecessor/batch context.

### EXP1-main — the SAME probe with the demand in the MAIN page (100 reps) — SUPERSEDED by EXP3
**The "main does not prefetch mid-page" conclusion below is an artifact of batch composition — see EXP3.**

| x1 | demand set | WITHOUT flush | WITH flush (prefetch only) | faulty equivalent |
|---|---|---|---|---|
| 0x0000 | 0 | [0, 2, 3, 4, 5, 6] | [2, 3, 4, 5, 6] | same as faulty |
| 0x0001 | 0 | [0]                | []              | **no prefetch** |
| 0x0020 | 0 | [0]                | [2, 3, 4, 5, 6] | **self-contradictory across batches** |
| 0x0040 | 1 | [1]                | []              | faulty: [17, 25] |
| 0x0080 | 2 | [2]                | []              | faulty: [18, 26, 34, 42, 50] |
| 0x00c0 | 3 | [3]                | []              | faulty: [19, 27, 35, 43, 51] |
| 0x0100 | 4 | [4]                | []              | faulty: [20, 28, 36, 44, 52] |

**The asymmetry reproduces on a 6-instruction test case** — no gadget, no speculation, no boosting, no
violation involved. Mid-page demands (sets 1-4) prefetch NOTHING in main while the identical page offsets in
faulty stream 5 lines each. This is a property of the two PAGES, not of the violation's test case.

Main's set-0 prefetch is **unstable**: 0x0000 fires, 0x0001 does not, and 0x0020 disagreed with itself between
the unflushed and flushed batches of a single run (same cache line as 0x0000). Same marginality as #28
(0.00/0.67/0.48) => #28 "dark" is NOT the harness hiding a prefetch; main genuinely does not prefetch reliably.

### EXP2 — targeted knockout of prefetched lines (demand x1=0x1000, prefetches sets 2-6)
| flushed by the TC | prefetched sets still lit |
|---|---|
| none                                  | [2, 3, 4, 5, 6] |
| faulty+0x1080 (set 2)                 | [3, 4, 5, 6] |
| faulty+0x1080,0x10c0 (sets 2,3)       | [4, 5, 6] |
| faulty+0x1080,0x10c0,0x1100 (sets 2,3,4) | [5, 6] |

Exactly the flushed sets go dark, one at a time, nothing else moves. **The sets are genuine prefetcher-filled
cache residency — not an F+R walk artifact.** This also resolves the main-vs-faulty page ambiguity of the
htrace (main and faulty share the same 64 sets, OR'd): flushing the **faulty** address darkens the set, so the
resident lines are in faulty.

### Status of the violation
The "harness artifact / false positive" verdict is **RETRACTED and UNRESOLVED**. What is established: the lit
sets are real prefetches; the footprint is demand-address-driven; the main-vs-faulty asymmetry survives a
position-neutral harness. What is NOT established: why main and faulty differ, and therefore whether
violation-260716-181940 is a genuine leak or a prefetcher-geometry artifact.

(The TEMP(lmfu) flush/reload extension is EXPERIMENTAL — revert before normal fuzzing.)
