# Cortex-A510: one memory access deposits a cache footprint at five undocumented addresses

Physical Pixel 8 (Tensor G3). Revizor executor, Flush+Reload. All results below are from
**single-input measurements** (see "Methodology" — batching corrupts this experiment), 100 reps,
3 independent trials per point; every point is marked stable or unstable.
Raw log: `verify_claims.log`. Script: `verify_claims.py`. Full history: `RESULTS.md`.

## Headline

A single demand access on the Cortex-A510 installs **5 extra cache lines** at a fixed offset from it.
`DC CIVAC` on the accessed line removes that line and leaves all 5 untouched, so their position — a
deterministic function of the accessed address — still identifies what was touched after the scrub.

**Framing (important).** `DC CIVAC` is architecturally specified to invalidate **one line by VA**; it never
promised to touch the other 5 addresses. This is therefore **not** a flush-correctness bug and not an ARM
erratum. It is a **scrubbing-completeness** result: one access deposits a footprint at five addresses the
defender **does not know exist**, because ARM documents no offset value for these engines. The general
observation that flush-based scrubbing leaves prefetcher residue is already known (Ge et al., 2016) — the
contribution here is the concrete A510 mechanism and its undocumented parameters.

**What appears genuinely unreported** (see "Prior work"): (1) `prfm`, a software prefetch *hint*, triggers the
hardware prefetcher — contradicting ARM's own separate-accounting model; (2) the prefetch fires only in the
first 512 B of a page, a cutoff no documented design predicts; (3) the effect exists on the little in-order
A510 but not on the big out-of-order cores of the same SoC.

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

**Do not overclaim this.** A prefetched line is just a cache line: `DC CIVAC` works by address and carves out
no exception for how the line arrived. Our own knockout (§9) proves it — naming `faulty+0x080` evicts set 2
normally. So flush-survival is **architecturally unremarkable**: flushing A leaves lines at A+1KB because you
did not flush A+1KB. Confirmed by grep of the full TRM: of the five co-occurrences of "maintenance" and
"prefetch", four are PMU event definitions and the fifth is §8.5's stream-end bullet; ARM says nothing about
flushing prefetched lines because there is nothing to say. **The contribution is therefore NOT the flush — it
is that the defender cannot know which addresses to flush**: the offsets, the 512 B trigger zone, and
`prfm` firing the prefetcher are all undocumented.

**The point is scrubbing completeness, not a flush bug.** `DC CIVAC` does exactly what it is specified to do
— one line, by VA. The problem is that the defender would have to also flush 5 further addresses whose offsets
ARM does not document (and which we had to measure). Scrubbing built on flushing "the line I touched" is
therefore insufficient by construction, against an adversary who observes the whole cache.

## Experiments

Victim (x1 comes from the input, so the demand address is input-driven; the victim flushes its own line):

    and x1, x1, #0x1fff ; add x0, x29, x1 ; ldr x3, [x0] ; dc civac, x0 ; dsb ish ; isb

### 1. A single access is enough — no training  **[NOT ESTABLISHED — see caveat]**
The victim is one load in a 6-instruction test case: no loop, no branch mistraining, no speculative window.

**CAVEAT (do not state this as a finding).** The test case is executed **~100 reps back-to-back plus warmup
rounds, all at the SAME address**, so the prefetcher sees that address ~100+ times in quick succession. TRM
§8.5 says the engine *"looks for cache line fetches with regular or repetitive patterns"* — i.e. the repetition
across reps could be doing the training, and a one-access *test case* does not demonstrate a one-access
*trigger*. A true first-touch test (an address never accessed before, measured once) has NOT been done, and is
hard here: prefetcher state demonstrably persists across runs (see Methodology) and no CPUECTLR bit disables
the A510 L1 stride prefetcher to force a clean slate.

Partial defence, not decisive: demand set 8 never prefetches even over 100 identical reps (if repetition alone
built a stream, set 8 should eventually stream too — it does not, stably), so repetition cannot be the whole
mechanism and the trigger zone is real. The §2 `prfm` result also stands independent of the training story.

What IS established: no *branch* mistraining and no speculative window are needed (see the speculation test:
a load behind a branch that skips it leaves nothing, even with the branch trained the wrong way).

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

## Prior work (literature search, 2026-07-17)

- **A510 TRM §8.5** (101604_0102_21_en, p76) lists the conditions that end a prefetch stream, including
  verbatim: *"A data cache maintenance operation is committed."* This does **not** contradict our result —
  ending a *stream* is not invalidating *lines already fetched* — but it sharpens the framing two ways.
  (i) A defender reading the TRM would reasonably conclude `DC CIVAC` stops the prefetcher; it does, the stream
  ends, yet the 5 already-installed lines stay resident and those are what leak. (ii) Our `flush()` issues 256
  CMOs before **every** run, which per §8.5 should end all streams — yet contamination still crosses inputs. So
  the contamination is **learned state** (tables/offsets), not live streams. ARM never states that a CMO clears
  the prefetcher's learned tables, only that streams "end"; that gap is unspecified in every retrieved doc.
- **ARM erratum 2077160** (SDEN-1873361, Category C) is the closest vendor prior art: *"causing the cache clean
  and invalidate to not clean and invalidate the line brought in by the hardware prefetch."* **It does not
  explain our result**: it requires a change in cacheability (break-before-make) and is **fixed in r0p3**,
  whereas Tensor G3's A510 is **r1p1**. It shows ARM knows the failure *mode*, and that ours is a different one.
- **Ge et al. (2016)** already make the general claim that flush-based scrubbing leaves prefetcher residue →
  our observation 8 is a concrete instance, not a new class. **Michaud (BOP)** covers same-page containment.
  **Augury / GoFetch** are data-dependent (DMP) prefetchers — our experiment 4 rules that out, so those are
  ruled out rather than matched. **AfterImage / CVE-2023-33936** cover prefetcher state persisting across
  contexts (our batch contamination) adversarially.
- **`PRFM` is a clean documented negative.** Across the A510/A55/A53 TRMs, ARM nowhere states that PRFM trains
  or triggers the hardware prefetcher. Stronger: A53/A55 define `L1PCTL` as the outstanding-prefetch budget
  *"not counting the data prefetches generated by software load/PLD instructions"* — i.e. software prefetches
  are **explicitly accounted separately** from the hardware prefetcher. Our §2 result (a `prfm pldl1keep` hint
  installing 5 HW prefetches) cuts against ARM's own accounting model. **Best-supported novel element.**
- **A510 microarchitecture** (TRM): five data prefetch engines — L1 stride, L2 offset, L2 pattern, L3 stride,
  L3 offset. `L2OPFTRIG` defaults to *offset prefetcher triggered by the pattern prefetcher*; `L1SPFL2` resets
  to 0, so L1 stride prefetches into L1 and L3, **skipping L2**. Note: **the L1 stride prefetcher cannot be
  fully disabled via CPUECTLR_EL1 on the A510** (A53/A55 could) — so a "prefetcher off" control was never
  available to us, independent of that register being firmware-trapped on this part. The Software Optimization
  Guide is silent: "prefetcher" appears zero times.
- **Search caveats:** the SDEN retrieved is **v16.0 (Dec 2023)**; v17-19 are gated, so a newer erratum cannot be
  excluded. No erratum exists for page-boundary prefetch behaviour or prefetcher state. Web-search snippets were
  caught **fabricating vendor content** (a nonexistent A510 "L2 region prefetch" field; a nonexistent A55 "8KB
  granule"), so every vendor claim here was extracted from the actual PDFs.

### Novelty verdict
| obs | claim | status |
|---|---|---|
| §2 | `prfm` (SW hint) triggers the HW prefetcher | **unreported**, contradicts ARM's separate-accounting model |
| §5 | 512 B trigger zone (sets 0-7 only) | **unreported and unexplained** by any documented design |
| §3 | A510-only; absent on A715/X3 | **unreported**; inverts the "A510 borrows big-core prefetchers" lineage |
| §5 | same-page containment | known (Michaud BOP); the A510 offsets are not |
| §4 | address-driven, not data-dependent | not novel (rules out Augury/GoFetch) |
| §8 | flush-survival | general claim known (Ge et al. 2016); reframed as scrubbing-completeness |
| Meth. | cross-input prefetcher contamination | known adversarially (AfterImage, CVE-2023-33936) |

**Strongest novelty: §2 + §5 (trigger zone) + §3.** Not §8.

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
