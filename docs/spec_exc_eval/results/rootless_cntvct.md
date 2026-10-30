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
| **iTLB** | canon | 24-page sled, reverse-probe | 150 | 0.998 | **585** |
| iTLB | aut | 24-page sled, reverse-probe | 80 | 1.000 | 1 101 |
| **BTB** (generic, which-set) | — | set-recovery, dwell-timing | REPEAT 48 | 100/100 | 3 bits/round |

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

## Cross-validation against the root PMU (verify, don't trust the timer)

A PMU victim-probe with the exact `L1D_TLB_REFILL` event (0x05), same structure as the rootless dtlb
harness, reads **bit0 = 0 refills, bit1 = 30/30 refills** — a crisp binary 0-vs-30 at 100%. The rootless
cntvct victim-probe measures the same resident-vs-evicted effect as a noisy *analog* of that binary
event (hence it needs averaging), and returns the same verdict. Rootless == PMU.

## Why the rates differ by medium

Same shape as the PMU matrix: cache ≫ TLB ≫ sled/branch. L1D is a single flushed line (one fast reload,
reps=1 → ~2 M bits/s). dTLB is a tiny single-page walk-latency delta (needs averaging → ~8 K). The iTLB
sled trades rate for separability — 24 serialized page fetches per shot and reps≈100-150 → ~0.6-1.1 K.
The generic BTB is a which-of-8 recovery, 3 bits/round.
