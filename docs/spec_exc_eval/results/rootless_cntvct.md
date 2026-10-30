# Rootless reproduction of the canon / AUT\* channels via `cntvct_el0` (Neoverse N3)

Every same-thread (same-EL0) channel in the covert-channel matrix reproduced with **no root, no PMU, no
`raw_pmu.ko`** — the generic virtual timer `cntvct_el0` as the only clock. On this N3 `cntfrq_el0` =
1 GHz (1 ns/tick, ~10 ns back-to-back resolution), readable at EL0 by default, which is fine to resolve
a TLB walk / cache miss with light averaging. All harnesses run as an ordinary user with
`taskset -c 8`; the secret bit is recovered over a known 0/1 oracle.

## Results (same-EL0, core 8, 2026-10-30)

| medium | technique | decoder | reps/floor | success | rootless bits/s |
|---|---|---|---:|---:|---:|
| **L1D** | canon | F+R, random line, majority | 1 | 0.995 | **2.30 M** |
| L1D | aut | F+R, random line, majority | 1 | 0.995 | 2.27 M |
| **dTLB** | canon | victim-probe, stable-cal | 120 | 1.000 | **8 403** |
| dTLB | aut | victim-probe, stable-cal | 120 | 1.000 | 8 442 |
| **iTLB** | canon | 24-page sled, reverse-probe | 80 | 0.998 | **1 097** |
| iTLB | aut | 24-page sled, reverse-probe | 80 | 0.993 | 1 090 |
| **BTB** (generic, which-set) | — | set-recovery, dwell-timing | REPEAT 48 | 100/100 | ≈2K (3 bits/round) |

BTB controls: shuffled probe order 63/64 (not a positional artifact), fixed-set 7/64 (= chance, no
off-channel leak). The harnesses are `l1d_cntvct.c`, `tlb_victim_cntvct.c` (dtlb), `itlb_sled_cntvct.c`,
`btb_game_cntvct.c` in `~/arm-disclosure/pocs/bit_shift_canon/`.

## What each reproduction needed (and what it corrected)

1. **same-EL0 canon×iTLB LEAKS — correcting the earlier "no-leak" verdict.** `rate_table.md` recorded
   canon×iTLB as a "genuine same-EL0 negative". That was a **probe artifact**, not a channel property.
   A single-page victim probe cannot separate the bits same-EL0 because bit0 fetches the victim and bit1
   speculatively fetches *some other* garbage page — both evict one entry (1-vs-1). The paper-faithful
   fix is a **24-page jump-sled** target (`b .+4096` chain): an in-range speculative branch fetches a
   run of ~24 pages (evicting ~24 of a 128-page eviction set), a non-canonical target fetches ~1 —
   24-vs-1 separates cleanly. Reverse-probing the eviction set, timed with cntvct, recovers it at
   0.998 (canon) / 1.000 (aut). So canon×iTLB leaks same-EL0, exactly as cross-EL does; the sled is the
   requirement, not privilege.
2. **same-EL0 dTLB leaks rootless** via a single-page victim-probe (prime 256 pages to evict the victim,
   run the v4 gadget, time one victim re-access: resident=fast=bit0, evicted=slow=bit1). The per-shot
   signal is ~10× smaller than the cache's, so it needs reps≈120 and a stable one-time threshold
   calibration (a single noisy calibration + the baseline drifting over the run was the cause of the
   earlier instability; averaging the threshold over 256 rounds fixes it).
3. **L1D needs a randomized target line.** A fixed line reads chance despite a clean mean separation:
   the v4 window is long enough (cache flush + fence + speculative gadget) that the hardware stride
   prefetcher refills the fixed line before the timed reload (flush-only = 153 ticks, flush+v4 on a
   fixed line = 54). Picking a random line in a 4 MB buffer each shot the prefetcher cannot predict
   restores a crisp 37-vs-157-tick hit/miss split → 0.995 single-shot.
4. **within-process canon/aut BTB does not separate — confirmed, not a rootless limitation.** The
   rootless readout reads exactly 0.500, and so does the **root PMU** version (`btb_rate.c`, BR_MIS_PRED)
   — the within-process canon/aut BTB signal is buried under the ~2-mispredict / stub-return baseline
   (the canon BTB leak is inherently cross-ASID). The same-process BTB channel that *does* exist is the
   generic which-set recovery (`game.c`); its PMU mispredict count converts directly to a rootless
   **dwell-time** readout (time the W probe branches per set; the victim-polluted set mispredicts and so
   runs slower) and recovers 100/100 with the controls intact.

## Cross-EL and cross-process (contention channels): blocked by the timer resolution

The same-EL0 channels above all read a **resident-vs-flushed** signal: bit0 leaves the line/page in a
cache or TLB (fast), bit1 leaves it in **memory** (slow). On this N3 that is an easy split for cntvct:
measured **L1-hit = 35, L2-hit = 39, memory = 155 ticks**, and cntvct's granularity is ~10 ticks.

The **cross-EL (EL2→EL0)** and **cross-process** channels are different in kind — they are **contention**
channels. The attacker cannot name the victim's target (it lives in the kernel module / another address
space), so there is no victim-probe; the only readout is reverse-probing the attacker's own eviction set
to see that the victim displaced one entry. That displacement moves a line **L1→L2**, i.e. 35→39 ticks —
**below cntvct's resolution**. So the rootless cntvct readout reads chance (measured: xel centroids ≈ 0
for L1D/dTLB/iTLB; success 40–65% and not improving with samples). The root PMU reproduces these fine
because `L1D_CACHE_REFILL` counts the L1→L2 refill as a discrete event regardless of its ~4-tick cost.

This is a **hardware timer limit, not a channel or effort limit**: it is not fixable by more samples or a
better probe structure. The legitimate rootless path for the contention channels is a **counting-thread
timer** (a sibling thread spinning on a shared counter — the technique TikTag uses on the Pixel), whose
resolution is far finer than cntvct and can resolve L1↔L2. That is a separate subsystem, not built here.
Scaffold: `channel.c` has a `-DROOTLESS` build (cntvct threshold-count) in `~/arm-disclosure`
(`aut_bit55/`), which confirms the resolution wall above; it is WIP, not a working cross-EL result.

## Matrix gap status (so no cell is silently blank)

- **EL2→EL0 BTB** — N/A: CSV2 blocks cross-privilege BTB target injection (same reason AUT*×BTB is n/a
  cross-ASID). Not a measurement gap, a hardware negative.
- **xcore (2 cores) L1D / L2 / dTLB / iTLB / BTB** — N/A by topology: those structures are per-core;
  two cores share only the L3. That is why the xcore row carries L3 alone.
- **same-thread canon/AUT* BTB** — confirmed negative (0.500 on both rootless and root PMU); only the
  generic which-set BTB exists same-process (≈2K b/s).
- **cross-EL / cross-process L1D/dTLB/iTLB rootless** — unmeasured with cntvct; blocked by the timer
  resolution (above). Needs a counting-thread timer.

## Cross-validation against the root PMU (verify, don't trust the timer)

A PMU victim-probe with the exact `L1D_TLB_REFILL` event (0x05), same structure as the rootless dtlb
harness, reads **bit0 = 0 refills, bit1 = 30/30 refills** — a crisp binary 0-vs-30 at 100%. The rootless
cntvct victim-probe measures the same resident-vs-evicted effect as a noisy *analog* of that binary
event (hence it needs averaging), and returns the same verdict. Rootless == PMU.

## Why the rates differ by medium

Same shape as the PMU matrix: cache ≫ TLB ≫ sled/branch. L1D is a single flushed line (one fast reload,
reps=1 → ~2 M bits/s). dTLB is a tiny single-page walk-latency delta (needs averaging → ~8 K). The iTLB
sled trades rate for separability — 24 serialized page fetches per shot, reps≈80 → ~1.1 K. The generic
BTB is a which-of-8 recovery, 3 bits/round.

**canon ≈ aut everywhere (equal-rate primitives).** L1D 2.30M≈2.27M, dTLB 8403≈8442, iTLB 1097≈1090 —
exactly as expected from a one-instruction (`autiza`) difference. (An earlier draft reported iTLB canon
585 @ reps150 vs aut 1101 @ reps80; that asymmetry was a floor-finding artifact from a sparse sweep —
pinned across 4 trials each, both floor at reps≈60–80 with indistinguishable per-shot separation.)
