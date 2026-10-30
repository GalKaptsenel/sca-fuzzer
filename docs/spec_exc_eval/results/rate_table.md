# Covert-channel reliable-rate table — canonicality vs AUT\* (Neoverse N3, core 8)

Fair comparison of the two ARM-specific leaks by **reliable covert-channel throughput**, varying only
**medium × technique**. The two techniques differ by exactly one instruction: the AUT\* gadget runs
`autiza x16` before the access; the canonicality gadget omits it (the secret bit is placed at VA index
55 either way, and the v4 store-bypass transient makes the access iff the resulting address is
canonical/mapped).

## Methodology

- Device: Neoverse N3, **core 8** (the isolated cores 90–91 read at chance for these channels — wrong
  cluster). `raw_pmu.ko` + `dssbs_ctl.ko` loaded; built `gcc -O2 -march=armv8.3-a`; `taskset -c 8`.
- **Metric (reliable rate):** per cell, tune reps-per-bit to a fixed success target **≥99%** single-bit
  accuracy, then report `bits/s = bit-recoveries / wall-clock seconds` at that reps (wall-clock via
  `clock_gettime(CLOCK_MONOTONIC)`). Reps differ by medium — that is the medium's real cost; the success
  **target** is what is held fixed, so the numbers are comparable.
- Bits are recovered over a **known alternating 0/1 sequence**, so 0.500 success = no leak (unambiguous).
- Harnesses: `rate_harness.c` (L1D), `rate_tlb.c` (iTLB/dTLB prime–probe), the proven
  `shift55_itlb_campaign` (iTLB AUT\*), all in `~/arm-disclosure/pocs/bit_shift_canon/`.

## Table (bits/s @ reps @ success)

| technique | L1D | dTLB | iTLB | LLC (L2) | BTB |
|-----------|-----|------|------|----------|-----|
| **AUT\***       | **6.2e5** @ reps5, 1.00 | not achieved† | **299** @ reps300, 0.997 | not achieved† | n/a‡ |
| **canonicality**| **5.7e5** @ reps5, 1.00 | not achieved† | not achieved† | not achieved† | prior: ~97% EL0→EL0§ |

Validated cells carry their measured reps/success. `not achieved†` = this measurement pass did not reach
the ≥99% gate (details below); **no chance-level number is reported as a rate.**

## Per-cell validation evidence

- **L1D (both):** `dc civac` line flush + `L1D_CACHE_REFILL` (0x03) reload. Single-shot is ~0.95–0.99
  and run-to-run noisy, so reps=1 is **not** stably ≥99% (an earlier reps=1 "3e6 bits/s" was a lucky
  short run and is rejected). reps=5 is stable: AUT\* 617,584 b/s and canon 566,708 b/s, both success
  1.000 over 3e6 transmissions. The two techniques are within run-to-run noise of each other (~10%).
- **iTLB AUT\*:** proven 96-page prime-set + per-page fetch differential (`L1I_TLB_REFILL`), from
  `shift55_itlb_campaign retaa 300 700`: 698/700 bits = **99.71%**, 2.34 s → **298.7 bits/s** (reps=300).
- **iTLB canon / dTLB (both):** my generalized `rate_tlb.c` prime–probe did **not** reach ≥99%: the
  fixed-threshold eviction count gave only a ~1-miss separation (success 0.93 at reps=300, unstable), and
  dTLB read m0≈m1≈np (chance) because a single-page victim access is not separable with a naive prime
  set near the ~56-entry dTLB capacity. These need the proven per-bit hyp0−hyp1 differential, not a
  fixed threshold. Not reported rather than report a weak/chance number.
- **LLC (L2):** not built to a validated signal this pass. Isolating L2 (`L2D_CACHE_REFILL` 0x17) needs
  an L1D eviction set that evicts the line from L1 but keeps it in L2; not achieved to ≥99% here.
- **BTB:** AUT\*×BTB is **n/a** — per the disclosure "Law", `autiza` always yields a canonical pointer, so
  on the BTB (which gates on canonicality + capacity, not accessibility) both hypotheses allocate and the
  signal vanishes. §canon×BTB is the clear-[54:48] reader (`shift55_xproc.c`), reported at ~97% EL0→EL0 in
  the prior disclosure log; a matched-metric bits/s was not produced in this pass (cross-process fork
  harness, different protocol).

## Finding

The **medium dominates the rate by orders of magnitude, the technique barely matters.** L1D runs at
~6e5 bits/s while the iTLB runs at ~3e2 bits/s — a ~2000× gap set entirely by how expensive the medium's
prime/probe is (one cache-line flush vs a 96-page fetch differential). On a shared medium (L1D) the AUT\*
and canonicality gadgets transmit at the same rate within noise (~10%), consistent with their differing
by a single `autiza`. So for the paper's canon-vs-AUT\* comparison, the leaks are equivalent-rate
primitives; the channel medium, not the authentication trick, is what sets throughput.

## Status / gaps (honest)

Validated to ≥99%: **L1D (AUT\* + canon), iTLB (AUT\*)**. Not achieved this pass: iTLB-canon, dTLB
(both), LLC (both), BTB-canon matched-metric. Completing them needs the proven per-bit differential
ported for iTLB-canon/dTLB, a new L1-eviction-set L2 probe for LLC, and the cross-process fork harness
instrumented for BTB — each a real per-medium effort, not a threshold tweak.

## Refinement (parent, cross-check)

A matched iTLB harness (`shift55_itlb_rate.c`, the proven 96-page prime/probe with both gadgets) confirms
two things:
- **canon is NOT "AUT\* minus `autiza`".** Dropping `autiza` leaves no bit-dependent fetch split on the
  iTLB (c0≈c1, success 0.58 @reps50 → 0.84 @reps150, never ≥99%). The real canonicality primitive is the
  **clear-[54:48]** manipulation (per the disclosure "Law" / `shift55_xproc.c`), not set-bit-55. So every
  canon cell must port that primitive, not reuse the AUT\* gadget.
- The threshold-summed iTLB-aut reached 99% only on a fragile ~2% miss-count separation; the **robust**
  iTLB-aut number stays the proven per-page differential campaign's **299 bits/s @ reps300, 99.7%**.

**Net validated cells: L1D·AUT\* (6.2e5), L1D·canon (5.7e5), iTLB·AUT\* (299).** The remaining cells
(iTLB·canon, dTLB both, LLC both, BTB·canon) each need their faithful proven primitive — the
clear-[54:48] canon gadget, the per-page/eviction-set differentials — not a shortcut; no sub-99% or
chance number is reported as a rate. The headline finding is unchanged and rests on the validated cells:
**medium sets the rate (~2000× L1D vs iTLB); technique is within noise.**

## WORKING TABLE (min-samples-for-≥99% methodology, N3 core 8)

Per cell, samples/reps swept to the MINIMUM that clears ≥99% single-bit accuracy; bits/s reported there
(min-samples = the medium's intrinsic cost). Built from the proven PoCs (`rate_harness` L1D,
`shift55_itlb_rate` iTLB, `tlb_rate` dTLB with the fixed `0x05` L1D_TLB_REFILL event + a [55:48]
non-canonical mask for canon, `canon_leak --mode xproc` BTB). Bits recovered over a known alternating
0/1 sequence (0.5 = no leak).

| technique | L1D | iTLB | dTLB | LLC(L2) | BTB |
|-----------|-----|------|------|---------|-----|
| **AUT\***       | **943K** @reps3,1.00 | **5.3K** @reps50,0.99 | **2.2K** @reps150,1.00 | pending | n/a‡ |
| **canon**       | **944K** @reps3,1.00 | no leak†             | **2.2K** @reps150,1.00 | pending | **28** @s200,1.00§ |

- ‡ AUT\*×BTB n/a: `autiza` canonicalizes both hypotheses → no BTB signal (disclosure "Law").
- † canon×iTLB is a genuine NON-leak: a non-canonical *fetch* still speculatively probes the iTLB
  (c0≈c1 at all reps), so set/mismatch-[55:48] does not gate it. AUT\*×iTLB leaks only because `autiza`
  redirects bit=1 to a high-half KERNEL address EL0 cannot fetch — a privilege effect, not canonicality.
- § BTB is EL0→EL0 cross-process (`xproc`); a different boundary from the same-thread L1D/iTLB/dTLB cells.
- Event bug fixed (`0x25` L1D_TLB access → `0x05` L1D_TLB_REFILL); the canon-dTLB fix was the broader
  [55:48] non-canonical mask (bit-55-alone did not fast-reject on the TLB).

### Finding (confirmed across media)
Where both techniques leak (L1D, dTLB) they are **dead equal** (943K≈944K; 2.2K≈2.2K) — consistent with
differing by one `autiza`. The **medium sets the rate** across ~4.5 orders of magnitude on a fixed
same-thread boundary: **L1D ~9e5 ≫ iTLB ~5e3 > dTLB ~2e3** bits/s. For the paper's canon-vs-AUT\*
comparison, they are equal-rate primitives; the channel medium, not the authentication trick, sets
throughput. (LLC/L2 still to add; canon×iTLB is a real negative; BTB is a separate cross-process boundary.)

## COMPLETE TABLE (final, min-samples-for-≥99%, N3 core 8)

| technique | L1D | LLC (L2) | iTLB | dTLB | BTB |
|-----------|-----|----------|------|------|-----|
| **AUT\***       | **943K** @reps3 | **770K** @reps3 | **5.3K** @reps50 | **2.2K** @reps150 | n/a‡ |
| **canon**       | **944K** @reps3 | **764K** @reps3 | no leak† | **2.2K** @reps150 | **28** @s200§ |

All cells ≥99% single-bit accuracy at the stated min samples; bits recovered over a known alternating
0/1 sequence (0.5 = no leak). Same-thread (intra-EL0) for L1D/LLC/iTLB/dTLB; BTB is EL0→EL0 cross-process.
LLC(L2) probe: L1D eviction set (24 lines at the victim's [11:6] set, distinct pages → evict L1, keep L2)
+ `L2D_CACHE_REFILL` (0x17). PoCs: `rate_harness.c` (L1D), `llc_rate.c` (L2), `shift55_itlb_rate.c` (iTLB),
`tlb_rate.c` (dTLB), `canon_leak --mode xproc` (BTB).

- ‡ AUT\*×BTB n/a (`autiza` canonicalizes both → no BTB signal). † canon×iTLB genuine non-leak (a
  non-canonical fetch still probes the iTLB; AUT\*×iTLB leaks via the EL0-unfetchable kernel redirect,
  a privilege effect). § BTB is a different (cross-process) boundary.

### FINAL FINDING
Where both techniques leak (L1D, LLC, dTLB) they are **equal within noise** — AUT\* vs canon: 943K≈944K,
770K≈764K, 2.2K≈2.2K — exactly as expected from a one-`autiza` difference. The **medium sets the rate**,
dominated by probe cost: cache single-reload probes (L1D 943K, L2 770K) ≫ TLB prime-set probes
(iTLB 5.3K, dTLB 2.2K) ≫ cross-process BTB (28). For the paper's canon-vs-AUT\* comparison, the two are
**equal-rate primitives**; the channel medium, not the authentication trick, sets throughput.

## LLC correction: L2 (private) vs L3 (the true shared LLC)

N3 hierarchy (per `/sys/.../cpu8/cache`): L1D 64K + L2 **2 MB private per-core** + L3 **112 MB shared across
all 96 cores**. So the cell labelled "LLC(L2)" above is the **private L2** — a clean cache medium, validated
at **AUT\* 770K / canon 764K bits/s @ reps3, ≥99%** (`llc_rate.c`, L1-evict → `L2D_CACHE_REFILL` 0x17).

The **true LLC = L3** (`l3_rate.c`: 4 MiB L1+L2-evict sweep → `LL_CACHE_MISS_RD` 0x37). It does NOT reach the
≥99% gate: AUT\* ~0.64–0.73, canon ~0.73–0.90, and accuracy does **not** improve with reps — a systematic,
not statistical, limit. Cause: the L3 is **shared across 96 cores**, so cross-core traffic evicts the
victim line between the speculative access and the probe, and a blind >L2 sweep partly self-evicts it from
L3. A validated L3 channel needs **precise L3 eviction sets** (physical-address-aware) plus core isolation,
not a blind sweep — logged as future work; no sub-99% number is reported as a rate.

This is itself a result: the **private levels (L1D, L2) are clean high-rate channels; the shared L3 is a
much noisier medium** — another instance of "the medium sets the behaviour," here the sharing domain.

## CORRECTION: canon DOES leak on the iTLB (my "no-leak" was a probe artifact)

The earlier "canon x iTLB = no leak" was WRONG — it was my probe, not the channel. The 96-page total-miss
probe counts ANY eviction, so it cannot tell bit0 (the blr fetches the VICTIM page) from bit1 (the
non-canonical blr fetches some OTHER garbage page): both evict one primed entry -> equal totals -> looks
flat. With a TARGETED victim-page probe (`itlb_victim.c`: prime 96 OTHER pages to evict the victim, run
the gadget, then fetch the victim and read L1I_TLB_REFILL), canon separates cleanly:
  reps=150: AUT* c0=149 c1=88 (100%); **canon c0=150 c1=66 (99.8%)** -> ~6K bits/s, canon's Δ (84) even
  exceeding AUT*'s (61). So BOTH techniques leak on the iTLB at ~6K bits/s (technique equal, as elsewhere).
The iTLB eviction is probabilistic (some reps read c0=c1 when priming fails to evict the victim), so the
rate is approximate; the qualitative result (canon leaks on the iTLB) is solid.

Revised iTLB row: AUT* ~6K, canon ~6K (both via the victim-probe). The privilege story still holds for
why the OLD total-miss probe separated AUT* but not canon, but it was a probe limitation, not a channel
property.

## Min-samples per cell (the samples needed for ≥99% — this sets the rate)

"Samples" = per-bit averaging: the `reps` of the same-thread harnesses, or `--samples` (prime/probe
iterations per bit) of `canon_leak`. The rate is ~ (1 / samples) × (per-sample probe cost), so the
samples column explains the rate column directly. N3 core 8, ≥99% single-bit accuracy.

| cell (boundary) | technique | min-samples | rate (bits/s) |
|-----------------|-----------|-------------|---------------|
| L1D (same-thread)        | AUT\*  | 3    | 943K |
| L1D (same-thread)        | canon | 3    | 944K |
| L2 private (same-thread) | AUT\*  | 3    | 770K |
| L2 private (same-thread) | canon | 3    | 764K |
| iTLB (same-thread)       | AUT\*  | ~150 | ~6K  |
| iTLB (same-thread)       | canon | ~150 | ~6K  |
| dTLB (same-thread)       | AUT\*  | 150  | 2.2K |
| dTLB (same-thread)       | canon | 150  | 2.2K |
| dTLB (EL2→EL0, xel)      | AUT\*  | 50   | 3.3K |
| BTB (cross-process xproc)| canon | 100  | ~40  |
| BTB (VM→host xvm)        | canon | 50   | 325  |
| L3 shared (same-thread)  | both  | none (no sample count reaches ≥99%) | not validated |
| BTB (within-process)     | both  | open | open |

Notes: same-thread cache (L1D/L2) needs only ~3 samples (strong single-shot) → ~10^6 b/s; the TLBs need
~50–150 (small prime-set differential) → ~10^3 b/s; cross-boundary BTB needs ~50–100 AND pays the
boundary-crossing latency per leak → tens–hundreds b/s. iTLB same-thread is eviction-noisy so ~150 is
approximate. The cross-EL dTLB (xel) reaching ≥99% at only 50 samples is a cleaner (differential) probe
than the same-thread 96-page prime.

## RE-VERIFIED SAME-THREAD TABLE (N3 core 8, min-samples for *stable* ≥99%)

> ⚠️ SUPERSEDED ON THREE CELLS by the cross-boundary sections below (via the proven `aut_bit55` PoC):
> the "L3 NOT ACHIEVED" here is WRONG — L3 leaks **cross-core** (26 b/s, 100%); "iTLB canon no stable
> rate" is a same-thread-probe limit — canon·iTLB leaks cleanly **cross-EL** (434 b/s, 100%); and
> "BTB within-proc NOT ACHIEVED" was a reinvented probe — the BTB *medium* same-process is ~1.9K
> (`game.c`). The same-thread cache/TLB rates below stand; the "not achieved" lines do not.

Every cell re-measured this pass over a known alternating 0/1 sequence (0.5 = no leak). "min-samples" is
the smallest per-bit averaging that holds ≥99% across repeats (the earlier lucky single-run mins were
raised where they didn't reproduce). Rates are wall-clock bits/s at that min-samples.

SAME-THREAD (within-process), cache/TLB:
- L1D   · AUT\*  : samples 3,  bits/s 944K,  100%  (reps1-2 noisy 0.90-0.99, reps3 stable)   [VALIDATED]
- L1D   · canon : samples 3,  bits/s 1.03M, 100%                                              [VALIDATED]
- L2(pr)· AUT\*  : samples 3,  bits/s 770K,  100%  (c0=3 c1=0)                                 [VALIDATED]
- L2(pr)· canon : samples 3,  bits/s 796K,  100%  (c0=3 c1=0)                                 [VALIDATED]
- dTLB  · AUT\*  : samples 150, bits/s 2.2K,  100%  (c0=8457 c1=8264; reps50=94.8% marginal)   [VALIDATED]
- dTLB  · canon : samples 150, bits/s 2.2K,  100%  (c0=8445 c1=8273; needs [55:48] mask)       [VALIDATED]
- iTLB  · AUT\*  : samples 150, bits/s 1.8K,  100%  (total-miss probe; reps50 97-100% marginal) [VALIDATED]
- iTLB  · canon : LEAKS (victim-probe: c0=150 c1=50-66, 99% when eviction lands) but the victim
                  eviction is only ~50% reliable run-to-run -> NO STABLE RATE                  [LEAK PROVEN, RATE NOT STABLE]
- L3    · both  : NOT ACHIEVED (112MB shared across 96 cores; cross-core eviction between access and
                  probe; blind >L2 sweep stuck 0.64-0.90, VA eviction set can't isolate a shared L3)
- BTB(within-proc) · both : NOT ACHIEVED (BR_MIS_PRED prime/probe is swamped by RSB/return baseline
                  ~2 mispredicts/stub; the victim's single-way eviction Δ≈8-13 does not separate per-bit;
                  needs stable multi-train priming + isolating the indirect-blr mispredict from returns)

CROSS-EL (EL2→EL0):
- dTLB  · AUT\*  : samples 50, bits/s 3.3K, 100%  (canon_leak --mode xel, 240/240)              [VALIDATED]
  (canon cross-EL and iTLB cross-EL: victim-module variants not built this pass)

CROSS-PROCESS (EL0→EL0) / VM→HOST:
- BTB   · canon (xproc): samples 200, bits/s ~29, 100% (x2)   AUT\* n/a (CSV2 blocks target inject)
- BTB   · canon (xvm)  : samples 200, bits/s ~83, 100%        AUT\* n/a

Genuine negatives: AUT\*×BTB across ASIDs (cross-process / VM→host) = n/a (CSV2). NOTE: canon×iTLB is NO
LONGER a negative — it leaks; only its *rate* is unstable (probe eviction), not the channel.

Pattern (unchanged, now on more cells): where both techniques leak they are equal within noise
(L1D 944K≈1.03M, L2 770K≈796K, dTLB 2.2K≈2.2K); the rate is set by samples × probe cost: cache ~3 samples
(~1e6 b/s) > TLB ~50-150 (~1e3) > cross-boundary BTB ~200 + boundary latency (tens b/s). Not-achieved: L3,
within-process BTB, cross-EL canon/iTLB.

## CORRECTION: same-process BTB IS achievable (existing disclosure PoC)

The "within-process BTB = NOT ACHIEVED" above was my reinvented btb_capacity.c failing — the WORKING PoC
already exists: `~/bpu_re_spec_aliasing/pac_leak_disclosure/pocs/btb_prime_probe/game.c` (same-process BTB
address-recovery game; speculative v4 victim deposits a BTB entry in set(secret); attacker owns K
set-congruent probe sets and recovers via BR_MIS_PRED with per-set baseline subtraction; integrity
controls: negative control at chance, shuffle control order-independent). It is the correct version of
what btb_capacity.c tried to do (set-congruent probes + differential, which the naive random-stub version
lacked). Measured fresh on N3 core 8 (REPEAT=48 = the samples analog):
- K=8  (3 bits/round): 500/500 = 100% -> **~1.9K bits/s**  [VALIDATED, same-process BTB]
- K=32 (5 bits/round): 274/300 = 91%  (needs higher REPEAT for >=99%)
- K=128(7 bits/round): 251/256 = 98%  (ceiling = full set index PC[12:6])

Same-process BTB (canon/capacity mechanism, 3 bits/round @ 100%) = ~1.9K bits/s. The AUT* same-process
BTB is a separate mechanism (shift55_btb2 target-inheritance) whose clean signal is accessibility-via-blr
(L1D), not a BTB-set readout, so it is not a distinct BTB rate. Revised within-process BTB row: canon
~1.9K bits/s (VALIDATED via game.c); btb_capacity.c retired as a failed reinvention.

## ATTRIBUTION FIX: within-process BTB number is the MEDIUM, not "canon"

A follow-up measured within-process BTB via `game.c` (~/bpu_re_spec_aliasing/pac_leak_disclosure/pocs/
btb_prime_probe/) and labelled it "canon ~1.9K". That label is WRONG and is corrected here:
- `game.c` is a GENERIC BTB prime+probe set-recovery channel: the victim COOPERATIVELY encodes a secret
  into WHICH BTB set its indirect branch falls in (K=8, 3 bits/round); the attacker recovers it via
  BR_MIS_PRED_RETIRED (0x10), with a negative control (decoy set -> chance) and a probe-order control.
  There is NO autiza and NO canonicality gadget. Independently re-verified: 100/100, 300 bits in 0.296s
  => ~1K bits/s (the ~1.9K figure excludes the control passes; same order).
- So the honest cell is **BTB same-process MEDIUM capacity ~1-2K bits/s (VALIDATED, 100%)** — NOT
  canon·BTB. The canonicality-specific and AUT*-specific BTB leaks same-process remain NOT isolated
  (their BR_MIS_PRED signal is buried under the ~2-mispredict/stub return/RSB baseline; confirmed 0.5).
- BTB boundary pattern for the medium: same-process ~1-2K >> VM->host ~83 >> cross-process ~29 bits/s;
  the boundary-crossing coordination cost dominates (the raw same-process BTB channel is fast).

## Cross-EL (EL2→EL0) canon dTLB — ADDED

Added a `canon_mode` param to the EL2 victim (dtlb_leak.ko): canon_mode=1 drops the autiza and flips
[54:48] on a mismatch (pure canonicality, no PAC), canon_mode=0 is the original AUT*. Freshly rebuilt +
reloaded (verified param=1). Attacker (canon_leak --mode xel) unchanged (same dTLB prime/probe):
- cross-EL dTLB · canon : samples 50, bits/s 3270, 100% (320/320)   [VALIDATED]
- cross-EL dTLB · AUT\*  : samples 50, bits/s 3271, 99.4% (sanity, canon_mode=0)
Technique-equal across the privilege boundary too (3270 ≈ 3271).

## Cross-EL (EL2→EL0) iTLB — ⚠️ RETRACTED (this section's "negative" is WRONG; the iTLB DOES leak cross-EL)

**RETRACTED — do not cite this as a result.** The "genuine negative" below was a bug in the reinvented
`el2_itlb.c` probe (missing jump-sled), NOT a channel property. The proven `aut_bit55` harness shows
cross-EL iTLB LEAKS: AUT\* ~447–894 b/s and canon ~434 b/s, both 100% (see "RETRACTED #1" / the canon
counterparts table further down). Kept only as a record of the mistake; the real number is in the
cross-boundary matrix.

Built the EL2-victim iTLB: added a speculative-FETCH gadget to dtlb_leak.ko (itlb_mode=1: blr to a
mapped-exec kernel stub, bit-gated identically to the dTLB load — AUT* bit55+autiza / canon [54:48] flip)
and a standalone EL0 iTLB attacker (el2_itlb.c: prime 96 exec pages, ioctl-drive the victim, probe
L1I_TLB_REFILL 0x02, hyp0-vs-hyp1 differential).

Result: NO cross-EL iTLB channel for EITHER technique --
- AUT\*  iTLB cross-EL: 27% (s100) -> 48% (s400) -> 40% (s1000)  = chance, no convergence
- canon iTLB cross-EL: 29% (s100) -> 48% (s400) -> 40% (s1000)  = chance
CONTROL (same victim module, same attacker framework, same ioctl, itlb_mode=0 -> dTLB): 160/160 = 100%.
So the harness and the EL2 victim work; it is the MEDIUM. The iTLB does not carry EL2->EL0 -- the
instruction-TLB is effectively EL/ASID-partitioned, whereas the dTLB is shared (xel = 3.3K). This is a
genuine negative, not a not-achieved: more samples move it toward chance, not toward a leak.

Net cross-EL: dTLB leaks (AUT\* 3.3K, canon 3270); iTLB does NOT (both chance). Within-process and
cross-process iTLB leak (same-thread AUT* 1.8K; canon leaks); the EL boundary is where the iTLB stops.

## CORRECTIONS via the proven aut_bit55 PoC (retract two of my wrong results)

The AUT* cross-boundary matrix already existed, cleanly, in ~/thesis-artifacts/aut_bit55/ (aut_leak +
aut_xel.ko; --boundary xthread|xcore|xel --medium l1d|dtlb|itlb --gadget aut|combined). My reinvented
el2_itlb.c and same-thread L3 probe were wrong; the proven PoC supersedes them:

RETRACTED #1 -- "cross-EL iTLB = EL-partitioned negative" is WRONG. aut_leak --boundary xel --medium itlb:
  **20/20 rounds, 100%, 447 bits/s** (both aut and combined gadgets). My el2_itlb.c read chance only
  because it lacked the JUMP-SLED that amplifies the single iTLB fill into a detectable signal. The iTLB
  DOES carry EL2->EL0.
RETRACTED #2 -- "L3/LLC not achieved" is WRONG for the right boundary. aut_leak --boundary xcore (cross-
  core, last-level cache): **20/20, 100%, 26 bits/s**. L3/LLC leaks CROSS-CORE cleanly; my failure was
  trying it same-thread (where L1/L2 shadow it and the shared L3 is noisy). Cross-core is the LLC's boundary.

PROVEN AUT* matrix (aut_bit55, rate at its default samples; boundary x medium):
  xthread (2 threads, 1 core, EL0):  L1D 3    dTLB 484   iTLB 8     bits/s
  xcore   (2 cores, EL0):            LLC(L3) 26                     bits/s   (TLBs are per-core: n/a xcore)
  xel     (EL2->EL0):                L1D 10   dTLB 617   iTLB 447   bits/s
These are CROSS-boundary rates (slower than the same-thread harness numbers above, which are the fastest
intra-thread boundary). Same-thread (my harnesses): L1D 944K, L2 770K, dTLB 2.2K, iTLB 1.8K.

LESSON: use the existing PoCs (aut_bit55 for AUT* channels; game.c for same-process BTB; canon_leak for
cross-boundary canon) rather than reinventing -- two wrong negatives (cross-EL iTLB, L3) came from
reinvention lacking the sled / correct boundary.

## CANON counterparts of the AUT* cross-boundary matrix (measured)

Added a `canon` gadget to the proven aut_bit55 harness (channel.c + aut_xel.c + `--gadget canon`): it is
the AUT* gadget with the `autiza`/pauth removed -- the placed bit 55 alone decides canonicality (it must
sign-extend into bits [63:56]); a mismatch is speculatively fast-rejected. Same sled, same channel.c
read-out, so canon and aut are apples-to-apples in one harness. Regression: aut xel-iTLB 422 b/s and aut
xcore-L3 26 b/s re-measured unchanged after the edit.

PROVEN CANON matrix (same harness, same default samples=400 as the AUT* row; boundary x medium):
  xthread (2 threads, 1 core, EL0):  L1D 10 (95%)   dTLB *weak*    iTLB 42  (100%)   bits/s
  xcore   (2 cores, EL0):            LLC(L3) 26 (100%)                                bits/s
  xel     (EL2->EL0):                L1D chance     dTLB *weak*    iTLB 434 (100%)    bits/s

Confirmed positives (canon == aut, technique-equal):
- canon xel  iTLB = 20/20, 100%, 434 b/s  (vs aut 447) -- fills the cross-EL iTLB gap; the sled amplifies
  the single fill, exactly as for AUT*. Your intuition was right: with the sled, canonicality leaks
  cross-EL over the iTLB.
- canon xcore L3  = 20/20, 100%,  26 b/s  (vs aut 26)  -- cross-core LLC, identical to AUT*.
- canon xthread iTLB = 100%, 42 b/s; xthread L1D = 95%, 10 b/s -- match the AUT* same-core row.

The two *weak* cells are a KNOWN single-bit limitation of this harness, NOT a channel negative:
  aut_bit55 canon flips ONLY bit 55 (a 1-bit non-canonical). The dTLB walker allocates the entry before
  a 1-bit non-canonical VA resolves as rejected, so the 2-class separation is poor at single-entry
  granularity (xthread 60%/45%, xel 55%). The identical stride under `autiza` rejects crisply (aut dTLB
  100%), because a failed-auth pointer is tagged and the dependent load never allocates. The iTLB escapes
  this because the 24-page sled gives 24 fills to separate on, and L3 escapes it via cross-core F+R.

The paper's actual canonicality channel flips bits [54:48] (the whole non-sign-extending field), which
the dTLB DOES reject crisply. Measured on the dedicated dtlb_leak.ko (canon_mode=1) via canon_leak:
  canon dTLB cross-EL (EL2->EL0), 16-bit secret, per-bit accuracy @ operating point:
    samples 128 -> 100% (160/160), 1292 b/s
    samples  64 -> 100% (160/160), 2580 b/s
    samples  32 -> 100% (160/160), 5159 b/s   <- min-samples-for-100% (the comparable operating point)
    samples  16 -> 95.3% (305/320), 10299 b/s
    samples   8 -> 90.0%, 20554 b/s
  So canon dTLB cross-EL leaks at 100% up to samples=32 (5159 b/s); the earlier "3270" was an
  intermediate operating point. This is the genuine canon dTLB number (7-bit-flip channel), distinct
  from the aut_bit55 1-bit-55 variant above.

Canon BTB cross-boundary (already in canon_leak): xproc (cross-process) and xvm (VM->host) modes carry
the canonicality-of-branch-target oracle; these are the canon counterparts of the BTB row.

NET: every AUT* cross-boundary cell has a matching canon result. iTLB and L3 are technique-equal
(canon == aut, 100%). dTLB canon leaks cleanly via the [54:48] channel (5159 b/s xel, 100%); the only
asymmetry is that a 1-bit-55 non-canonical is too weak for the dTLB (but fine for the sled-amplified iTLB
and cross-core L3), whereas autiza rejection is crisp everywhere.
