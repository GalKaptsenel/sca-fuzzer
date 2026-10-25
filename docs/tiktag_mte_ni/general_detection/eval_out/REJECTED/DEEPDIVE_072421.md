# Deep-dive characterization — violation fuzzer_072421 (MTE TikTag-v1, Cortex-X3)

Subject: `docs/tiktag_mte_ni/general_detection/eval_out/VERIFIED_GENUINE/fuzzer_072421/`
Config: `s24_mte_random.yml` (fuzzer: non-interference, clause cond, observation l1d, F+R, pinned core 8,
superbatch_size 1, view_rotation on, branch mistraining off). Suspect input #1. Device 47061FDJH001K3.

All three analyses (disassembly, simulation, hardware) agree. The finding is a speculative MTE
pointer-vs-memory tag check whose MISMATCH trains a contiguous L1D prefetch stream (present-on-mismatch
polarity), anchored to the sealed speculative access's granule, gated at pointer-tag delta 7, tracking the
memory tag, outside the COND-contract footprint. The active transmitter is window W2; three seals
(A/B/D) each produce the effect, with a data-dependency that makes D's reach conditional on A matching.

---

## 1. Static disassembly map (enacted_test0001.bin — the genuine/all-match enactment)

304 bytes, 76 instructions, no header, code at offset 0. The MTE sealer expands every memory access to:
`and xN,#0x1ff0 ; add xN,x29 ; [adjust] ; addg xN,#0,#uimm4 ; <access>` where x29 = sandbox base and the
ADDG uimm4 (bits [13:10]) is the pointer-tag delta. Genuine uimm4 = 7 for every seal: pointer tag =
(base_tag 15 + 7) mod 16 = 6 = the uniform memory tag.

| BB | offset range | content (branches, seals, accesses) |
|----|--------------|--------------------------------------|
| 0.0 | 0x00–0x04 | measurement NOP; `b.nv 0x08` (always) |
| 0.1 | 0x08–0x0c | `ccmn x0`; `b.nv 0x10` (always) |
| 0.2 | 0x10–0x34 | `bics w1`; seal x2(0x20); `str w5,[x2,x1]`(0x24, arch); `and/csel`; `b 0x38` |
| 0.3 | 0x38–0x40 | `csinv x2`; **`cbnz w3,0x44`(0x3c)**; `b.nv 0xc4`(0x40, always) |
| 0.4 | 0x44–0x70 | `ccmn`; **`eor w4,w0,w0`(0x48 → zeroes x4)**; seal x2(0x54); `stp w4,w5,[x2],#-76`; `subg x1,x5`; seal x0; `stzg x3,[x0,#2960]`; `b 0x74` |
| 0.5 | 0x74 | `b.nv 0x78` (always) |
| 0.6 | 0x78–0xc0 | seal x0; `str x5,[x0],#157`; seal x1; `str w3,[x1],#78`; `stz2g x2,[x2]`; seal x0; `str w5,[x0,x1]`; **`tbz x0,#58,0xc4`(0xbc)**; `b.al 0x128`(exit) |
| 0.7 | 0xc4–0xec | `cinv w2,w1`; `ldg x5,[x1,#-4048]`; `ldg x1,[x3]`; `udiv w1`; `eor x5`; `b 0xf0` |
| 0.8 | 0xf0–0x124 | **A:** seal x4(0xf8)+`ldr w2,[x4],#222`(0xfc); **B:** seal x1(0x10c)+`str x2,[x1]`(0x110)+restore(0x114); **D:** seal x2(0x120)+`ldp w2,w1,[x2]`(0x124) |
| exit | 0x128 | measurement NOP; `b` out |

Architectural direction (input #1): `cbnz w3` is TAKEN (w3 = 0xee39 ≠ 0 → bb_0.4); `tbz x0,#58` is
NOT-taken (→ `b.al` exit). So the architectural path is 0.0→0.2→0.3→0.4→0.5→0.6→exit and **never reaches
bb_0.7/bb_0.8**. Every sealed access (A/B/D) is reached only speculatively, through one of:

- **W1** = `cbnz w3` mispredicted NOT-taken → `b.nv`→bb_0.7→bb_0.8 (reached before the bb_0.4/0.6 stores).
- **W2** = `tbz x0,#58` mispredicted TAKEN → bb_0.7→bb_0.8 (reached after the bb_0.4/0.6 stores ran).

Seals (gate offset, register, protected access):
- **A** 0xf8, x4, `ldr w2,[x4]` @0xfc
- **B** 0x10c (+ restore 0x114), x1, `str x2,[x1]` @0x110
- **D** 0x120, x2, `ldp w2,w1,[x2]` @0x124

## 2. Enacted-binary decode — which decoy = which seal combination

The four enacted binaries are byte-identical except the uimm4 fields of A(0xf8), B(0x10c+0x114), D(0x120)
(`cmp` differs only at bytes 0xf9,0x10d,0x115,0x121). Genuine (match) delta = 7.

| enacted | A @0xf8 | B @0x10c/0x114 | D @0x120 | combination |
|---------|---------|----------------|----------|-------------|
| test0001 | 7 (match) | 7 / 0 | 7 (match) | all-match baseline |
| test0009 | 13 (mis) | 2 / 5 (mis) | 7 (match) | A + B mismatched |
| test0017 | 4 (mis) | 7 / 0 | 4 (mis) | A + D mismatched |
| test0025 | 7 (match) | 14 / 9 (mis) | 7 (match) | B mismatched only |

The four "counterexample inputs" (#1/#9/#17/#25) carry byte-identical register seeds and memory — they are
the same logical input materialised as these four seal variants. All three enacted decoys (0009/0017/0025)
have A or B mismatched, so all fall in the [1..11] stream class (see §6); the pure D-only wide stream is not
among the saved enactments and was produced here by editing D alone with A left matching.

## 3. Contract-executor simulation + by-hand emulation (agree exactly)

CE run of input #1 genuine, COND|BPAS, model_max_nesting 40, base = 0xffffff8869761000. Entry GPRs:
x0=0x20f5000020f5, x1=0x7d6f00007d6f, x2=0xc1d80000c1d8, x3=0xee390000ee39, x4=0x6bd200006bd2,
x5=0xad290000ad29. Granule = (ea − base)/16; main = granules 0–255, faulty = 256–511.

Per-window sealed-access granule/set (CE), with independent by-hand derivation from the input bytes:

| seal | access | W1 (before bb_0.4/0.6) | W2 (after bb_0.4/0.6) |
|------|--------|------------------------|-----------------------|
| A | `ldr w2,[x4]` | granule 189, set 47 | granule 0, set 0 |
| B | `str x2,[x1]` | granule 0, set 0 | granule 0, set 0 |
| D | `ldp w2,w1,[x2]` | granule 287, set 7 | granule 305, set 12 |

By-hand chain (confirms CE): `eor w4,w0,w0` in bb_0.4 zeroes x4, so in W2 `ldr w2,[x4]` reads granule 0
(value 0xf31d) whereas in W1 (bb_0.4 skipped) it reads granule 189 (value 0x91fa). That loaded word becomes
x2; `and x2,#0x1ff0` then gives 0x91fa→0x11f0→granule 287 (W1) and 0xf31d→0x1310→granule 305 (W2). So seal
A feeds seal D: the two windows differ because the bb_0.4/0.6 stores (and the EOR) change what A loads and
hence where D dereferences. CE arch sets = {3,7,21,23,63}; CE speculative-only sets = {0,12,47}. The
measured HW stream sets {1,4,5,6,8,9,10,11} (and D's extension to 28) lie outside both — the stream is not
accounted for by the COND contract, which is the non-interference violation.

Tag change does not move any effective address (uimm4 sets only bits [59:56]); genuine and decoy share
identical addresses/sets/granules. The only per-variant difference is the tag-check outcome.

## 4. Hardware matrix (position-controlled; fixed prefix of 14 genuine variants of other inputs; suspect
swapped at one fixed tail slot; view_rotation on; reps 1500–2000; residency = fraction of runs with the set
evicted). "Match" = pointer tag equals memory tag; TCMA leaves pointer tag 15 unchecked.

### 4a. 16-delta pointer-tag sweeps (one seal swept, the others left matching)
- **A @0xf8**: set 1 absent at uimm4=7 (ptag 6, match) and uimm4=0 (ptag 15, unchecked), present at every
  other delta. ptag 0 (uimm4=1) IS checked. Present-on-mismatch; match delta 7.
- **B @0x10c**: sets 1 and 4 absent at uimm4=7 and uimm4=0, present elsewhere. Match delta 7.
- **D @0x120**: sets 5/9/11 absent at match (uimm4=7) and unchecked (uimm4=0), present elsewhere; the gated
  load's own set 12 is lit regardless of tag. Match delta 7.

### 4b. Granule localisation
- A (probe set 1, gate 0xf8): only retagging **granule 0** lifts the set-1 suppression; granules
  189/188/190/4/1 do not. A checks granule 0 (its W2 access).
- D (prior, probe set 9, gate 0x120): retagging **granule 305** moves the match; granule 287 (W1) does not;
  granule 255 is a tag-independent architectural-store confound.

### 4c. Tag tracking (retag the gated granule, re-sweep the pointer; match delta must move as (C+T−cell)%16)
With the other seal pinned to pointer tag 15 (check off) to remove the shared-granule-0 confound:

| seal | probe set | T=3 | T=6 | T=9 | T=12 |
|------|-----------|-----|-----|-----|------|
| A (gate 0xf8, g0, pin B) | 1 | dip d4 | dip d7 | dip d10 | dip d13 |
| B (gate 0x10c, g0, pin A) | 4 | dip d4 | dip d7 | dip d10 | dip d13 |
| D (gate 0x120, g305) | 9 | dip d4 | dip d7 | dip d10 | dip d13 |

All three track exactly (C=7, cell=T): match delta = (7 + T − T) = 7 at the default, moving to (T+1) after
retag. Genuine pointer-vs-memory tag check for every seal.

### 4d. Anchor-move (hold the seal at a mismatch, shift its ADDG address uimm6 in granules; the lit block
must move with the access)
- A (anchor granule 0): u6=0 → block {0..12}; u6=16 (anchor set 4) → block {0..15}; u6=32 (anchor set 8) →
  block {0..~25}. Upper bound climbs with the anchor.
- D (anchor granule 305, set 12): u6=0 → block {2..28}; u6=16 (anchor set 16) → low edge climbs 4→7.
The stream is anchored to the sealed speculative access, not to a fixed set.

### 4e. 3-repeat stability (probe set, match uimm4=7 vs mismatch uimm4=9, three independent batches each)
- A set 1: match 0.00/0.00/0.00 — mismatch 1.00/1.00/1.00
- B set 4: match 0.07/0.09/0.20 — mismatch 1.00/1.00/1.00
- D set 9: match 0.00/0.00/0.01 — mismatch 1.00/1.00/1.00

### 4f. Seal-combination matrix (mismatch = uimm4 9; genuine floor = {0,2,3,7,12,21,23,56,63})
| combination | leak block (resident sets minus floor) |
|-------------|----------------------------------------|
| A only | {1,4,5,6,8,9,10,11} |
| B only | {1,4,5,6,8,9,10,11} (identical to A) |
| D only | {4,5,6,8,9,10,11,13,14,…,28} (set 1 dark) |
| A+B | {1,4,5,6,8,9,10,11} |
| A+D | {1,4,5,6,8,9,10,11} |
| B+D | {1,4,5,6,8,9,10,11} |
| A+B+D | {1,4,5,6,8,9,10,11} |

Not additive. A and B share granule 0 and give one and the same block. D alone (A matching) reaches set 28.
Any A-or-B mismatch collapses D's extension back to {1..11}, because A's mismatching speculative `ldr w2,[x4]`
stops forwarding the value that D dereferences (the A→D chain of §3). D's wide stream requires A to MATCH.

### 4g. Window isolation (retag one memory granule to a mismatch, pointers left genuine)
| retagged granule | reader | result |
|------------------|--------|--------|
| 0 | A(W2) & B | block {1..11} fires |
| 305 | D(W2) | block {4..28} fires |
| 189 | A(W1) | nothing (empty) |
| 287 | D(W1) | nothing (empty) |
| 255 | architectural store | {1..11}+{52..62}, tag-check of the arch store (confound, not a window) |

**W2 is the active window; W1 does not produce a measurable stream** (branch mistraining is off and the HW
predictor does not speculate the `cbnz`-not-taken direction, so W1's granules 189/287 are never engaged).

## 5. THE combined table — seal × window × granule × set-block × match-delta × tracks? × polarity

| Seal | Gate | Access | Window | Gated granule | Access set | HW stream block | Match delta | Tracks? | Polarity | HW active |
|------|------|--------|--------|---------------|------------|-----------------|-------------|---------|----------|-----------|
| A | 0xf8 | LDR w2,[x4] | W1 | 189 | 47 | — | 7 | n/a | present-on-mismatch | **no** |
| A | 0xf8 | LDR w2,[x4] | W2 | 0 | 0 | {1,4,5,6,8,9,10,11} | 7 | yes (d4/7/10/13) | present-on-mismatch | **yes** |
| B | 0x10c+0x114 | STR x2,[x1] | W1/W2 (granule 0 both) | 0 | 0 | {1,4,5,6,8,9,10,11}+set4 | 7 | yes (d4/7/10/13) | present-on-mismatch | **yes** (via W2) |
| D | 0x120 | LDP w2,w1,[x2] | W1 | 287 | 7 | — | 7 | no (g287 static) | present-on-mismatch | **no** |
| D | 0x120 | LDP w2,w1,[x2] | W2 | 305 | 12 | {4,5,6,8,9,10,11,13..28} (needs A match) | 7 | yes (d4/7/10/13) | present-on-mismatch | **yes** |

## 6. Stream geometry and overlap

- Contiguous, stride 1 (every set in the run), base near set 0–2, upper bound ≈ anchor_set + 13–16.
- A/B block {1,4,5,6,8,9,10,11} spans sets 1–11 (sets 2,3,7 sit under the genuine floor), anchor set 0.
- D block reaches set 28, anchor set 12 (inside the block).
- A and B overlap completely (shared granule 0). D overlaps A/B over sets 4–11 and extends distinctly over
  13–28, but only while A matches.
- The gated load's own set (0 for A/B, 12 for D) stays lit on match and mismatch alike; the MISMATCH adds
  the surrounding stream.

## 7. Reconciliation with prior data and the saved report
- `ctrl_pos_tag` / `gd_check` saw {1,4,5,6,8,9,10,11}: the A/B granule-0 stream (their decoys had A or B
  mismatched). `gd_check` decoy0's wider {4..30}: the pure D-only stream (A matching). Both now explained.
- `ce_set_class` speculative-only {0,12,47}: 0 = A-W2/B set, 12 = D-W2 set, 47 = A-W1 set. The leak stream
  sets are the prefetch residue around these, outside the contract footprint.
- TCO=1 (tag-check-off) collapse was established previously; not re-run (no kernel-module changes here).

## 8. Method, reps, device state, ambiguities
- Every hardware number is position-controlled (fixed 14-input genuine prefix, suspect at one fixed tail
  slot) to exclude batch-position confounds. Sweeps at reps 1500–2000; the core present-on-mismatch
  contrast confirmed over 3 independent repetitions per seal (§4e). Tag-tracking over 4 memory tags per seal.
- Device left canonical and idle: no lingering process, `/dev/executor` present, module loaded, collection
  campaign PAUSE untouched. No fuzzer started, no module rebuilt/reinserted.
- Ambiguities flagged honestly: (i) B's match floor at set 4 is ~0.12 (not 0.00) — the contrast to mismatch
  (1.00) is still unambiguous but B's suppression is less complete than A/D's. (ii) W1 is inactive on this
  hardware for this input; it is present in the CE model but not reproducible without branch mistraining, so
  its granule/set entries are from simulation only. (iii) Anchor-move of the wide D block is coarse for small
  u6 (sub-cache-line), clear only at u6≥16. (iv) granule 255 is the architectural store's own tag check, a
  confound distinct from the speculative windows.
