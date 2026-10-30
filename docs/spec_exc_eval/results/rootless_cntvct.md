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
| **dTLB** | canon | victim-probe, stable-cal (NPRIME 64) | 40 | ≥0.998 | **~52 000** |
| dTLB | aut | victim-probe, stable-cal (NPRIME 64) | 40 | ≥0.993 | ~51 000 |
| **iTLB** | canon | 40-page sled, reverse-probe (NPRIME 64) | 30 | ~0.99 | **~5 900** |
| iTLB | aut | 40-page sled, reverse-probe (NPRIME 64) | 12 | ≥0.993 | **~14 800** |
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

## Cross-process and cross-EL (contention channels): rootless via reload-line amplification

These are **contention** channels — the attacker cannot name the victim's target (it is in another
address space / the kernel module), so there is no victim-probe; the readout reverse-probes the
attacker's own eviction set to see the victim displaced one entry. A *naive* single-line probe cannot
read this rootless: a displaced entry only moves **L1→L2** (measured L1-hit = 35, L2-hit = 39, memory =
155 ticks), a ~4-tick change under cntvct's ~10-tick granularity. (That is the real, measured
resolution wall — but it is *not* the end of the story.)

The fix — already implemented in `arm-disclosure/pocs/cross_boundary/xproc.c` — is **reload-line
amplification**: back each eviction-set entry with several reload lines (`RELOAD_LINES = 8`), so when the
victim displaces that entry, those lines are flushed all the way **to memory**. The attacker then sums
their reload latency, which is the L1-vs-**memory** gap (35 vs 155) that cntvct resolves, ×8 amplified.
With this, **cross-process rootless works at 100%**: `canon_leak --mode xproc --rootless` recovers a
7-bit secret 10/10 rounds, 70/70 bits, ~6 bits/s, no PMU. `xvm` (guest→host) is likewise rootless by
construction (the KVM guest cannot touch the PMU, so it reads over cntvct). An earlier draft of this
file wrongly called the contention channels "blocked by timer resolution / needs a counting thread" —
**retracted**; the reload-line trick is the rootless path and it is already proven here.

### Cross-boundary rootless status (measured)

| boundary × medium | rootless result | notes |
|---|---|---|
| cross-process (EL0→EL0) **BTB** | **100%** (10/10 rounds, 70/70 bits, ~6 b/s) | `canon_leak --mode xproc --rootless`; reload-line amplification |
| VM→host **BTB** | rootless by construction | KVM guest can't touch PMU → cntvct (`--mode xvm`) |
| cross-EL (EL2→EL0) **dTLB** | **achievable — 100% on the reliably-leaking bits** (`xel --rootless`) | cntvct readout; the probe is a data load that walks to **memory** on a displaced page → resolvable |
| cross-EL **iTLB** | chance (47–54%, even s6000) | single-page instruction-fetch probe (front-end prefetches); needs a victim-side 24-page sled, but the victim is in the fixed `dtlb_leak.ko` module |
| cross-EL **L1D** | not built | no L1D-primed xel attacker yet; the data-load reload-line approach (as dTLB) should carry it |

**What made cross-EL dTLB work (and what still limits it):**
- **The fix was interleaving.** A first port measured all hyp0 trials then all hyp1 trials; baseline drift
  between the two passes flipped the differential's sign → whole rounds *inverted* (~42%). Porting the
  **xproc interleaved paired differential** — hyp0 and hyp1 back-to-back each trial, so drift cancels —
  removed the inversions, and per-bit accuracy then climbs monotonically with samples (69%→83%→96%→100%).
  Majority voting per bit is added on top.
- **It is slower than the PMU xel** (which runs at s32): timing a dTLB *walk* is a noisier signal than
  counting an `L1D_TLB_REFILL` *event*, so it needs more samples per bit. That is inherent to a rootless
  timer, not a defect.
- **The top byte `[63:56]` is profile-limited, not a readout failure.** The 16-bit secret spans `[63:48]`;
  per this PoC's README the top byte leaks only where `TBID1=1`, so those bits sit near the margin and cap
  whole-secret accuracy at ~88–100% across trials even with voting — exactly as the PMU version behaves.
  On the reliably-leaking low bits the rootless readout is 100%.

This supersedes an earlier draft that called cross-EL rootless "blocked by timer resolution / needs a
counting thread" — **wrong and retracted**: it is achievable with cntvct via the reload-line / data-walk
readout plus the interleaved differential, already in `cross_boundary`.

Harnesses: `arm-disclosure/pocs/cross_boundary/` — `canon_leak` (xproc/xvm/xel, `--rootless`), `el2_itlb`
(iTLB, `rootless` arg). Each EL2 victim is a kernel module (`dtlb_leak.ko`, one root `insmod` + `chmod
666 /dev/dtlb_leak`) — the privileged *victim* is inherent to a cross-privilege experiment; the *attacker*
readout is what is rootless.

## Same-thread canon/AUT* via a speculative INDIRECT BRANCH (the branch/BTB path)

Reading the canon/AUT* secret through a speculative **`blr`** same-thread, not a load — three outcomes,
all rootless cntvct, which together pin down exactly what the branch path can and cannot do same-thread:

| carrier → readout | technique | result | why |
|---|---|---|---|
| `blr` → **L1D** (gadget touches a line) | **AUT\*** | **works — 2.27M b/s @reps1 / 747K @reps3** | `autiza` sends the wrong case to a different (kernel) page → gadget not reached |
| `blr` → L1D | canon | **cannot** | non-canonical `[54:48]` target and the canonical one hit the **same physical gadget line** (front-end fetches on low bits) → both touch R |
| `blr` → **iTLB** (24-page sled) | **canon & AUT\*** | **works — ~1.1K b/s** | iTLB is **VA-tagged**, so canonical vs non-canonical sled pages are *different* iTLB entries → separable (this is the `itlb_sled` channel above) |

So **same-thread canon-via-branch does work** — through the iTLB (VA-tagged), 1.1K b/s; AUT\*-via-branch
also works through L1D at ~MHz rates. The branch front-end does not gate on `[54:48]` for the *fetch*
itself, which is why canon needs a VA-tagged structure (iTLB) rather than the physically-tagged L1D.
Harness: `btb_branch_canon_cntvct.c`.

## Matrix gap status (so no cell is silently blank)

- **EL2→EL0 BTB** — N/A: CSV2 blocks cross-privilege BTB target injection (same reason AUT*×BTB is n/a
  cross-ASID). Not a measurement gap, a hardware negative.
- **xcore (2 cores) L1D / L2 / dTLB / iTLB / BTB** — N/A by topology: those structures are per-core;
  two cores share only the L3. That is why the xcore row carries L3 alone.
- **same-thread canon/AUT* BTB** — confirmed negative by **two** readouts and a positive control. The
  single-site allocate-vs-reject readout (`btb_rate.c`) reads 0.500 on both rootless and PMU; and the
  **eviction-set** readout run inline (`btb_evset_same.c`, the xproc channel without the fork) also reads
  ~chance — its positive control shows an *architectural* canonical victim branch does **not** evict a
  primed co-set entry at any prime-set size (K=24..64: no-victim metric ≈ canonical-victim). The canon
  allocate-vs-reject channel needs a **cross-domain boundary**: cross-process/EL/VM the victim runs in a
  separate context that contends for the shared per-core BTB ways (and the context switch churns the BTB),
  but same-thread the primed entries stay fresh and one inline victim branch does not displace them. The
  generic which-set BTB (`game.c`, varying branch *site* → pollutes different sets) does work same-process
  (≈2K b/s) — a different mechanism. **Note the scope:** only the pure-BTB *eviction* readout needs the
  cross-domain boundary. Same-thread canon/AUT* *via a speculative indirect branch* DOES leak when read
  through a footprint — AUT* `blr`→L1D at ~MHz, canon `blr`→iTLB at ~1.1K (see the branch-path section
  above). So "same-thread BTB is absent" applies specifically to allocate-vs-reject eviction, not to the
  speculative-branch channel in general.
- **cross-EL / cross-process L1D/dTLB/iTLB rootless** — unmeasured with cntvct; blocked by the timer
  resolution (above). Needs a counting-thread timer.

## Cross-validation against the root PMU (verify, don't trust the timer)

A PMU victim-probe with the exact `L1D_TLB_REFILL` event (0x05), same structure as the rootless dtlb
harness, reads **bit0 = 0 refills, bit1 = 30/30 refills** — a crisp binary 0-vs-30 at 100%. The rootless
cntvct victim-probe measures the same resident-vs-evicted effect as a noisy *analog* of that binary
event (hence it needs averaging), and returns the same verdict. Rootless == PMU.

## Why the rates differ by medium

Same shape as the PMU matrix: cache ≫ TLB ≫ sled/branch. L1D is a single flushed line (one fast reload,
reps=1 → ~2 M bits/s). dTLB is a single-page walk-latency delta (needs averaging; ~52 K at reps40 with
the eviction set right-sized to the ~56-entry dTLB — NPRIME 64, not 256). The iTLB
sled, with the eviction set right-sized to the ~56-entry iTLB (NPRIME 64, not 128): canon ~5.9K @ reps30,
aut ~14.8K @ reps12 (aut needs fewer reps — `autiza` sends the reject to a genuinely different kernel
page, a cleaner split than canon's `[54:48]` flip). The generic BTB is a which-of-8 recovery, 3 bits/round.

**canon ≈ aut everywhere (equal-rate primitives).** L1D 2.30M≈2.27M, dTLB 8403≈8442, iTLB 1097≈1090 —
exactly as expected from a one-instruction (`autiza`) difference. (An earlier draft reported iTLB canon
585 @ reps150 vs aut 1101 @ reps80; that asymmetry was a floor-finding artifact from a sparse sweep —
pinned across 4 trials each, both floor at reps≈60–80 with indistinguishable per-shot separation.)
