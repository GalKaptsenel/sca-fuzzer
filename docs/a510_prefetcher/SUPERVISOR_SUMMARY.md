# Cortex-A510: a cache flush does not erase the footprint of the access it flushes

Physical Pixel 8 (Tensor G3). Revizor executor, Flush+Reload. All results below are from
**single-input measurements** (see "Methodology" — batching corrupts this experiment), 100 reps,
3 independent trials per point; every point is marked stable or unstable.
Raw log: `verify_claims.log`. Script: `verify_claims.py`. Full history: `RESULTS.md`.

## Headline

`DC CIVAC` evicts the cache line it names — but **not the lines the hardware prefetcher installed because of
that same access**. Those survive the flush untouched, and their position is a deterministic function of the
address that was accessed. So code that touches a secret-dependent address and then flushes that line,
believing it has scrubbed the trace, **still leaks which address it touched**. We show this with a
6-instruction victim containing a single memory access.

## Components involved

1. **The Cortex-A510 hardware data prefetcher.** A spatial/offset prefetcher. A single demand access installs
   5 additional lines at a fixed offset from it. Present only on the A510 (in-order "little", cpu0-3);
   absent on the out-of-order A715 (cpu4-7) and X3 (cpu8).
2. **`DC CIVAC`** — data cache clean & invalidate by VA to point of coherency. Architecturally: evicts *that*
   line. It carries no promise about state the access induced elsewhere — and that is the gap we exploit.
3. **Flush+Reload harness.** Flush the probed pages, run the victim, then time one load per cache set; a fast
   load means the line is resident. Produces a 64-bit htrace, one bit per L1D set.
4. **Sandbox layout.** Two adjacent 4 KB pages (`main`, `faulty`). Each is 64 lines, so **both alias onto the
   same 64 htrace sets** — a lit set alone does not reveal which page holds the line. Disambiguated by
   flushing one page's specific address (experiment E).
5. **The contract / cache-set abstraction.** Revizor models leakage at cache-set granularity; prefetcher-
   installed lines are outside that model, which is why they surface as "violations".
6. **The super-batch measurement path** (many inputs per round-trip) — a confound, see Methodology.

## Expected vs. observed behaviour of the flush

**Expected:** `DC CIVAC` on address A evicts A's line; a probe afterwards finds A absent.

**Observed:** A's line is absent — **and 5 lines that were never architecturally accessed are present**, at a
fixed offset from A, in the same page. Nothing in the victim touched them; only the prefetcher can have
installed them. Flushing A does not disturb them at all (experiment F). An attacker probing after the
victim's flush recovers the demand set from the survivors alone.

**The point: flushing is not an eraser.** It undoes the direct residency of the access, not the
microarchitectural state that access induced. Scrubbing built on `DC CIVAC` is insufficient against an
adversary who observes the whole cache rather than only the flushed line.

## Experiments

Victim (x1 comes from the input, so the demand address is input-driven; the victim flushes its own line):

    and x1, x1, #0x1fff ; add x0, x29, x1 ; ldr x3, [x0] ; dc civac, x0 ; dsb ish ; isb

### 1. A single access is enough — no training, no speculation
One load in a 6-instruction test case triggers the prefetch. No loop, no repetition, no warm-up, no branch
mistraining, no speculative window. (Contrast Spectre-style gadgets, which require a mistrained predictor.)

### 2. Any access type triggers it — including a store
Replacing the trigger instruction in place (F+R block strength):

| instruction at the trigger site | block 2-6 |
|---|---|
| `ldp w3, w5, [x3, #0x2c]` (baseline) | 0.97 |
| `ldr w3, [x3, #0x2c]` (single 32-bit load) | 0.94 |
| `ldr x3, [x3, #0x28]` (single 64-bit load) | 0.93 |
| `str w5, [x3, #0x2c]` (**a store**) | 0.95 |

Not pair-specific, not load-specific. Consistent with a demand-address prefetcher, not value forwarding.

### 3. A510-only (experiment A) — demand `faulty+0x000`
| core | prefetched sets | |
|---|---|---|
| cpu0 — **Cortex-A510** (in-order), MIDR 0x411fd461 | **[2, 3, 4, 5, 6]** | stable |
| cpu4 — Cortex-A715 (OoO), MIDR 0x411fd4d0 | [] | stable |
| cpu8 — Cortex-X3 (OoO), MIDR 0x411fd4e0 | [] | stable |

The effect exists only on the **in-order** core. An in-order core cannot produce this by speculative
execution, which is direct evidence for a prefetcher rather than a speculation artifact.

### 4. Address-driven, not data-dependent (experiment B)
Address held fixed at `faulty+0x000`; the entire contents of both sandbox pages varied:

| memory contents | prefetched sets | |
|---|---|---|
| all zeros | [2, 3, 4, 5, 6] | stable |
| all ones | [2, 3, 4, 5, 6] | stable |
| pattern 0x5a5a… | [2, 3, 4, 5, 6] | stable |
| value = a valid sandbox offset | [2, 3, 4, 5, 6] | stable |

The footprint is a function of the **address only**. Rules out a data-memory-dependent prefetcher (DMP):
even planting a pointer-like value at the accessed location changes nothing.

### 5. Pattern — trigger zone, strides and deltas (experiment C, faulty page, alone)
| demand set | prefetched sets | delta (sets) | in bytes | |
|---|---|---|---|---|
| 0 | 2, 3, 4, 5, 6 | +2…+6 | +128 B…+384 B, 64 B stride | stable |
| 1 | *(varies: [13] / [7,13] / [13,17,25])* | — | — | **UNSTABLE** |
| 2 | 18, 26, 34, 42, 50 | +16,+24,+32,+40,+48 | +1 KB…+3 KB, 512 B stride | stable |
| 3 | 19, 27, 35, 43, 51 | +16,+24,+32,+40,+48 | as above | stable |
| 4 | 20, 28, 36, 44, 52 | +16,+24,+32,+40,+48 | as above | stable |
| 5 | 21, 29, 37, 45, 53 | +16,+24,+32,+40,+48 | as above | stable |
| 6 | 22, 30, 38, 46, 54 | +16,+24,+32,+40,+48 | as above | stable |
| 7 | 39, 47, 55 *(+16,+24 land on background sets 23,31)* | +16,+24,+32,+40,+48 | as above | stable |
| **8 – 15** | **[]** — no prefetch at all | — | — | stable |

Two robust structural facts:
- **A trigger zone:** only a demand in the **first 512 B (sets 0-7)** of the page prefetches. Sets 8-15 give
  nothing, stably. This is a property of the *demand's position in the page*, not of target overflow — set 8's
  targets would still lie inside the page.
- **Five lines per trigger**, at +1 KB…+3 KB with a 512 B stride. Demand set 0 is the exception, using a
  tighter +128 B…+384 B / 64 B stride — **unexplained**.

### 6. It stays inside the demand's own page — NOT the next page (experiment E)
Demand `faulty+0x080` (set 2) → prefetches sets [18,26,34,42,50]. Set 18 could be `faulty+0x480` (same page)
or `main+0x480` (the other page) — they alias. Flushing one and not the other decides it:

| victim additionally flushes | prefetched sets still lit | |
|---|---|---|
| nothing | [18, 26, 34, 42, 50] | stable |
| **faulty+0x480** (set 18) | **[26, 34, 42, 50]** — set 18 darkened | stable |
| **main+0x480** (set 18) | [18, 26, 34, 42, 50] — unchanged | stable |

The prefetched lines live in **faulty, alongside the demand**: same page, forward within it. (Also the
conventional design — prefetchers do not cross a 4 KB boundary.)

### 7. Which address bits matter
- **bits [5:0]: no effect** — `0x1000`/`0x1001`/`0x1020` all give `[2,3,4,5,6]` (same 64 B line).
- **bits [11:6]: select the footprint** — `0x1080` → `[18,26,34,42,50]`, `0x10c0` → `[19,27,35,43,51]`.
- Two inputs differing **only in low address bits** therefore produce **different, distinguishable** cache
  footprints. This is the channel.

### 8. THE LEAK — the footprint survives the flush unchanged (experiment F)
The victim flushes its own demand line. Prefetched sets, with and without that `dc civac`:

| demand | without `dc civac` | with `dc civac` | |
|---|---|---|---|
| faulty+0x000 (set 0) | [2, 3, 4, 5, 6] | **[2, 3, 4, 5, 6]** | stable |
| faulty+0x080 (set 2) | [18, 26, 34, 42, 50] | **[18, 26, 34, 42, 50]** | stable |
| faulty+0x0c0 (set 3) | [19, 27, 35, 43, 51] | **[19, 27, 35, 43, 51]** | stable |

**The flush changes nothing about the prefetched lines.** It removes only the demand line itself (verified
separately: the with/without difference is exactly the demand set and nothing else). The surviving pattern is
a deterministic function of the demand set, so it still identifies the address the victim touched.

### 9. The survivors are genuine cache residency, not a measurement artifact (knockout)
Demand `0x1000`; the victim additionally flushes specific *prefetched* addresses:

| additionally flushed | prefetched sets still lit |
|---|---|
| none | [2, 3, 4, 5, 6] |
| faulty+0x080 (set 2) | [3, 4, 5, 6] |
| faulty+0x080, 0x0c0 (sets 2,3) | [4, 5, 6] |
| faulty+0x080, 0x0c0, 0x100 (sets 2,3,4) | [5, 6] |

Exactly the flushed sets go dark, one at a time; nothing else moves. F+R is reading real cache state.

## What the leak is, stated precisely

An attacker who probes the cache after the victim has flushed its secret-dependent line still learns
**bits [11:6] of that address** (i.e. which of the 64 cache sets was touched), because the prefetched
footprint encodes it and survives the flush. It does **not** reveal bits [5:0] — the granularity is the same
as the original access. The novelty is **survival of the flush**, not finer resolution. Applies to demands in
the **first 512 B of a page**; accesses at set ≥ 8 install no prefetch and leave no residue.

## Methodology: batch neighbours corrupt this measurement

The prefetcher's training state **carries across inputs measured in one super-batch**, so an input's htrace
depends on its predecessors. Measured alone, addresses that looked inert inside a batch prefetch normally.
Every number in this document is therefore from a single-input batch.

This resolved the violation the investigation started from (`violation-260716-181940`): measured alone, the
two counterexample inputs are **identical** (3/3 trials, same lit sets); inside the original 29-input batch
one collapses (0.34-0.37) while the other holds (0.89) — and that collapse *was* the violation. It is a
**false positive**. An earlier apparent asymmetry ("the `faulty` page prefetches, `main` does not") was the
same artifact: measured alone, **main and faulty behave identically** at every demand set (0-7).

Implication beyond this one violation: every htrace the fuzzer compares comes from a multi-input batch, and
boosting deliberately groups related inputs — precisely the condition that creates the confound. Re-measuring
(priming) does not catch it, because it reproduces the contamination.

## Confidence

- **Established, stable across 3 trials, single-input:** A510-only; content-independence; the trigger zone
  (sets 0-7 prefetch, 8-15 do not); the stride/delta patterns; same-page containment; flush-survival;
  the knockout; main ≡ faulty.
- **Known unstable:** demand set **1**, on *both* pages ([13] / [7,13] / [13,17,25]). The anomaly is set 1
  itself, not a page difference. Not understood.
- **Unexplained:** why demand set 0 uses a 64 B-stride pattern while sets 2-7 use 512 B; why the trigger zone
  stops at 512 B.
- **From batched runs, not re-verified alone** (directionally consistent but weaker): the instruction-type
  table (§2) and the PTE-attribute flips (in `RESULTS.md`).
