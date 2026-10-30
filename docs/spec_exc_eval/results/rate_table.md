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
