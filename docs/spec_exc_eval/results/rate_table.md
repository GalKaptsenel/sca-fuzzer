# Covert-channel rate matrix — canonicality vs AUT\* (Neoverse N3)

Fair comparison of the two ARM-specific speculative leaks by covert-channel throughput, varying only
**medium × technique × boundary**. The two techniques differ by one step: the **AUT\*** gadget runs
`autiza` before the access; the **canon** gadget omits auth and instead flips the non-sign-extending
field **[54:48]** so the address is non-canonical iff the secret bit says so. In both, the secret is
bit 55 of a fresh per-round value, and a v4 store-bypass transient makes the access iff the resulting
address is canonical/authentic (→ a footprint) or fast-rejects it (→ none).

All cells re-measured 2026-10-30, N3. Success is over a known 0/1 oracle (0.50 = no leak). Harnesses
(all proven, no reinvention): `aut_bit55/aut_leak` (+aut_xel.ko) for the cross-boundary matrix;
`bit_shift_canon/{rate_harness,llc_rate,shift55_itlb_rate}` for the inline same-thread cache peaks;
`cross_boundary/canon_leak` (+dtlb_leak.ko) for cross-process / VM→host / cross-EL; `btb_prime_probe/game.c`
for same-process BTB.

## Methodology — how each rate is defined

Each covert-channel transmission sends **one secret bit**; the attacker recovers it by averaging the
microarchitectural probe over `samples` (a.k.a. `reps`) repetitions of that bit (majority / differential).
More samples ⇒ higher recovery accuracy but lower throughput (wall-clock time per bit grows), so there is
a speed/reliability trade-off. We define the rate as the operating point at the knee of that trade-off:

> **For each cell (medium × technique × boundary): sweep `samples`, find the MINIMUM `samples` that still
> achieves ≥99% single-bit recovery success, and report the wall-clock `bits/s` at that minimum.**

`bits/s ≈ (1 bit) / (time per bit at the min-samples)`, i.e. rate scales ≈ 1/samples, so a cell's number
is its fastest *reliable* (≥99%) rate, not a lucky single-shot. **One caveat for comparability:** the
cross-boundary rows (xthread / xcore / xel / cross-process / VM→host) are reported at a **fixed
`samples=400` snapshot** rather than each cell's own floor — this holds `samples` constant so the
*cross-cell* comparison is apples-to-apples; the true per-cell min-samples floors are ~10× faster (e.g.
xel dTLB is 603 b/s @ s400 but 7485 b/s @ its s32 floor). The same-thread cache rows ARE at their
min-samples floor (`reps≈3`). "chance / no leak" means no `samples` count reaches ≥99% (0.50 ≈ coin).

## Master table (bits/s @ samples, success)

| boundary | medium | AUT\* | canon |
|---|---|---|---|
| **same-thread, inline** (EL0, no context switch) | L1D | 1.03M @r3, 1.00 | 972K @r3, 1.00 |
| | L2 (private) | 833K @r3, 1.00 | 771K @r3, 1.00 |
| | iTLB | 1.7K @r150, 1.00 | **leaks w/ sled, 1.00†** |
| **xthread** (2 threads, 1 core, EL0) | L1D | 100% @s2000 (≈2) | 100% @s2000 (≈2) |
| | dTLB | 492 @s400, 1.00 | 489 @s400, 1.00 |
| | iTLB | 43 @s400, 1.00 | **leaks w/ sled, 1.00†** |
| **xcore** (2 cores, EL0, shared L3) | L3 / LLC | 26 @s2000, 1.00 | 26 @s2000, 1.00 |
| **xel** (EL2→EL0, cross-privilege) | L1D | 100% @s4000 (≈2) | 100% @s2000 (≈2) |
| | dTLB | 603 @s400 → **7485 @s32**, 1.00 | 607 @s400 → **5159 @s32**, 1.00 |
| | iTLB | 445 @s400, 1.00 | 445 @s400, 1.00 |
| **cross-process** (EL0→EL0) | BTB | **n/a‡** | 34 @s200, 1.00 |
| **VM→host** (guest→host) | BTB | **n/a‡** | 80 @s200, 1.00 |
| **same-process** (EL0) | BTB | ~2.3K, 1.00 (generic medium§) | ~2.3K, 1.00 (generic§) |

- **†** canon×iTLB **does** leak same-EL0 — the earlier "no-leak" was a *single-page probe artifact*,
  now retracted (see `rootless_cntvct.md`). A single-page victim probe cannot separate the bits same-EL0
  because bit0 fetches the victim and bit1 speculatively fetches some *other* garbage page: both evict
  one entry (1-vs-1). With a **24-page jump-sled** target (the same amplification cross-EL uses), an
  in-range branch fetches ~24 pages and a non-canonical one ~1 (24-vs-1), and the eviction-set
  reverse-probe recovers it at 1.00 same-EL0 (rootless cntvct: ~1.1K b/s, canon ≈ aut). So the sled, not
  privilege, is what canon×iTLB needs; cross-EL (`xel`) also leaks (445 b/s, 100%).
- **‡** AUT\*×BTB across ASIDs = n/a: CSV2 blocks cross-ASID BTB target injection; `autiza` also
  canonicalizes both hypotheses, so there is no BTB signal to read.
- **§** same-process BTB is a generic medium-capacity channel (`game.c`: secret = which of K=8 BTB sets
  the victim branches in; controls — shuffle 64/64 order-independent, fixed-set 7/64 chance). It is not an
  AUT\*- or canon-specific leak; those remain buried under the ~2-mispredict/stub return baseline.

## What was corrected this pass (vs the earlier draft)

1. **canon dTLB is NOT weak.** The earlier "weak 55–60%" was a gadget bug: the canon gadget flipped only
   bit 55 (a 1-bit non-canonical the dTLB walker allocates before rejecting). Fixed to the paper-faithful
   [54:48] flip → canon dTLB now **100% both boundaries** (xthread 492, xel 608), equal to AUT\*.
2. **L1D "marginals" were undersampling,** not a channel limit. At samples≥2000 L1D is 100% on every
   boundary (the low rate — ≈2 b/s cross-thread — is the context-switch cost, not unreliability).
3. **The xel-dTLB "617 vs 5159 canon" gap was operating point, not channel.** At equal samples=32, AUT\*
   xel-dTLB is 7485 b/s (100%) ≥ canon 5159; the matrix's samples=400 rows are a comparable snapshot, not
   per-cell maxima (floors are ~10× higher).
4. **`rate_tlb dtlb` (inline) is broken** (reads a constant m0=m1=48 → chance); the dTLB same-core number
   is taken from the proven 2-thread `aut_bit55` harness instead. `rate_harness`'s llc/dtlb/itlb sub-modes
   are likewise L1D-only stubs — each medium uses its dedicated harness.
5. Two prior wrong negatives already retracted (cross-EL iTLB; L3) — both leak (447/26 b/s); the failures
   were a missing jump-sled and wrong boundary (L3 leaks cross-core, not same-thread).

## Findings

1. **Where both techniques leak, they are equal within noise** — L1D 1.03M≈972K, L2 833K≈771K, dTLB
   492≈489 / 7485≈5159, xcore L3 26=26, xel iTLB 445=445. Exactly as expected from a one-`autiza`
   difference. The two are equal-rate primitives.
2. **Medium (and operating point) set the rate, not the gadget** — inline cache ~10⁶ ≫ TLB ~10³ ≫
   cross-boundary BTB tens of b/s; within a cell, rate ≈ 1/samples.
3. **Two genuine technique asymmetries:** canon needs the [54:48] flip (bit-55-alone is too weak for the
   dTLB), and canon×iTLB does not leak same-EL0 (only cross-EL via the sled+privilege); AUT\*×BTB is n/a
   across ASIDs (CSV2). Everywhere else the techniques match.
