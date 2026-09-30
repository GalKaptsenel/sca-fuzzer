# Cross-input priming — implementation handoff (AArch64 → x86-64)

**Goal.** Port **only** the cross-input priming step from the AArch64 Revizor fork to the
x86-64 Revizor, and see whether it surfaces new leaks there. Nothing else moves.

**Why it matters (the *what* and *why*).** Standard priming discards a suspected violation
unless the two inputs differ in the L1D trace of *their own* run. That misses a whole class
of genuine leaks — the ones whose only footprint is **outside** the cache the tool reads
(a BTB entry, branch history, the TLB, the PHT, the RSB, the prefetcher). Cross-input
priming certifies exactly those, without ever reading the full leftover: it lets one input's
leftover propagate into the cache of a *later* run and reads it there. The full argument,
proofs, and figures are in `priming_generalization.tex` / `.pdf` (repo root) — **read that
first**; it is the specification. This file is the engineering map from that spec to code.

---

## 1. The algorithm in one page

Notation (matches the proof):

- **Test case `T`**, **input `i`**, **run `run(T,i,μ)`** from microarch state `μ`.
- **Contract trace `ct(T,i)`**; two inputs are in the same **class** iff `ct` equal.
- **Boosting**: expand one input into many of its class (distinct inputs, same `ct`).
- **Lane** `L = (i_0, …, i_k)`: one input per class `0..k`. Two lanes `a,b` meet each
  class with different members that share `ct`.
- **Initialization sequence `IS`**: a fixed routine run **before each lane** that drives the
  microarch state to one reproducible value `μ_IS`. Batches are `IS, L^0, IS, L^1, …`. `IS`
  separates lanes; it is **not** placed between classes inside a lane (the intra-lane forward
  cascade is the leftover channel we exploit).
- **Probe = class `k`** (the last class). WLOG the detecting class is last: a later run cannot
  affect an earlier run's trace, so drop classes after the detecting one and relabel.

**Splice** `σ_{a→b}(t)` = a batch row taking classes `< t` from lane `a` and classes `≥ t`
from lane `b`. So `σ(k+1)=` all-`a`, `σ(0)=` all-`b`; neighbours `σ(t)` and `σ(t+1)` differ
in exactly one input (class `t`) and share everything else — including the whole suffix
(classes `> t`), i.e. **the same routine runs after class `t`** in both.

**Splice measurement** `r(t)` = run `IS, σ_{a→b}(t)` and read the L1D trace of its class-`k`
(probe) run.

**Localization.** Revizor flags a suspected violation when lanes `a,b` disagree at the probe:
`r(k+1) ≠ r(0)`. Sweep `t` from `k+1` down to `0`. A **boundary** is a `t` with
`r(t) ≠ r(t+1)`.
- **A boundary always exists** (the ends disagree; Prop. "A boundary exists").
- **Every boundary is a genuine violation** (Lemma "genuine"): the two neighbouring batches
  run identical histories except the one input at class `t`, then the same routine, yet the
  probe reads differently → `(i_t^a, i_t^b)` is a genuine violation. If additionally the pair
  is `H_L1D`-indistinguishable in its own run, the violation is **noncache** — the case
  standard priming cannot report.
- There may be **several boundaries** (up to `k+1`), each a distinct leaking class.
- `t = k` (probe column) ⇒ **self-dependent** leak (standard priming already catches it);
  `t < k` ⇒ **cross-input** leak (the new case). A caller wanting only cross-input findings
  keeps `t < k`.

**Two localizers** (both use the same splices/measurement, both sound):
- **Linear scan** — read `r(k+1), r(k), …` until the first change; returns `t_max` (largest
  boundary), and can enumerate all boundaries. Worst case `k+2` runs; optimal for `t_max`.
- **Binary search** — keep an interval `[ℓ,h]` with `r(ℓ)≠r(h)`, always narrow to a half that
  still disagrees. Returns *some* boundary in `⌈log₂(k+1)⌉` runs; optimal for one boundary.

**Both directions.** Which class reads the leftout back can be lane-specific, so run the
sweep both `a→b` and `b→a`; reporting a boundary from either is sound.

**Measurement note.** Each `r(t)` is a *repeated* measurement; two readings differ only if
they stay apart across repetitions (noise-robust comparison). The only cross-class channel in
a spliced batch is the forward microarch cascade (arch state is set before each run; each lane
starts from `μ_IS`).

---

## 2. Reference implementation (AArch64), file by file

The core is **architecture-independent** — it never models the channel; it only compares
opaque traces. It is injected with three seams: `measure`, `key`, `confirm`.

### `src/aarch64/cross_input.py` — the pure algorithm (port ~as-is)
- `splice(base, toggle, detecting, t)` → the spliced variant list.
- `linear_scan(r_key, lo, hi)` and `exponential_search(r_key, lo, hi)` → the two localizers
  as pure functions over an oracle `r_key: int → Key` (galloping+bisection is
  `exponential_search`).
- `class GeneralizedPrimingDetector(measure, key, confirm, *, reps, verify_reps, localizer)`:
  - `measure(batch, reps) → [trace per slot]` — **the only hardware seam**.
  - `key(trace) → hashable` — canonical/denoised form used *during* the search.
  - `confirm(a, b) → bool` — robust (noise-tolerant) "are these really different" used to
    re-verify a boundary at `verify_reps`.
  - `find_leaking_pair(base, toggle, detecting, …)` → localizes one boundary (memoizes each
    split; re-verifies the boundary at `verify_reps`).
  - `detect(lane_a, lane_b, exclude_self_dependence)` → scans all detecting positions.
- **This file has no AArch64 in it.** It should port to x86 essentially unchanged.

### `src/aarch64/boosted_lanes.py` — lane bookkeeping
- `lanes_of(boosted, n_orig)` → split the boosted input list into lanes (one member per class).
- `detecting_position_and_lanes(htrace_groups, n_orig)` → from a flagged violation, find the
  detecting class and the two diverging lanes.
- Depends on how the fuzzer lays out boosted inputs; re-derive for x86's layout.

### `src/aarch64/aarch64_fuzzer.py` — the seam to the fuzzer (`CrossInputPrimingMixin`)
- `_priming(violations, inputs)`: when `enable_cross_input_priming`, **replaces** standard
  priming; builds lanes, makes a detector, localizes the flagged violation's boundary in both
  directions, attaches a `CrossInputFinding` for the report. Otherwise falls back to
  `super()._priming` (standard priming).
- `_make_cross_input_detector(reps, verify_reps, localizer)`: wires the seams —
  `measure = lambda batch,n: executor.trace_test_case(batch,n)[0]`,
  `key = MergedBitmapAnalyser.merged_bitmap`, `confirm = ChiSquaredAnalyser` (robust).
- `_localize_boosted_violation(...)`: focused localization of one flagged violation.
- `_sample_size_sweep_needed() → not CONF.enable_cross_input_priming`: **optimization** —
  cross-input priming already re-verifies at the largest sample in one pass, so it opts out of
  the slow-path sample-size sweep (see §4).
- `_CROSS_INPUT_PRIMING_REGIME`: the executor regime applied around the search (see §3).
- `executor_regime(...)` / `regime_controllable(...)`: apply/restore sysfs regime; require a
  local HW executor.

### `src/fuzzer.py` — one generic hook (shared with x86 already)
- `_sample_size_sweep_needed()` (base returns `True`); the slow-path loop at ~line 421 is
  guarded by it. **This lives in the shared base fuzzer, so the x86 fuzzer already has it.**

### Config (`src/aarch64/aarch64_config.py`)
- `enable_cross_input_priming: bool = False` — master switch (regular fuzzing).
- `cross_input_priming_localizer: "any" | "optimal"` — `any` = galloping+bisection (one
  boundary), `optimal` = linear scan (`t_max`).
- `cross_input_leaks_only: bool = False` — keep only `t < k` (strictly cross-input) findings.

---

## 3. What is architecture-specific (the actual porting work)

The algorithm is portable; these seams are not:

1. **`measure` — batch measurement.** x86 needs an executor call that runs a batch of
   variants and returns a per-slot L1D htrace at `n` reps. In the AArch64 fork this is
   `executor.trace_test_case(batch, n)`. Find/confirm the x86 executor's batch API and adapt.
2. **`IS` — the initialization sequence.** The algorithm needs a fixed routine that drives
   microarch state to a reproducible `μ_IS` before each lane. On AArch64 this is realized by
   the executor regime (BPU/PHR reset once per trace, etc.). **On x86 you must decide what
   `IS` is** (e.g. a flush/warm-up sequence, or the executor's existing per-trace reset) and
   ensure lanes start from the same state. This is the single most important arch decision.
3. **Regime knobs.** `_CROSS_INPUT_PRIMING_REGIME` on AArch64 sets SSBS / pre-run-flush /
   view-rotation / PHR-flush / pinning so that (a) cross-input training survives within a
   trace and (b) the readout is stable. x86 has its own equivalents (or none) — map each knob
   to the x86 executor's sysfs, or drop the ones that don't apply. Keeping training alive
   between lanes (do **not** flush everything between classes) is the crucial property.
4. **`key` / `confirm`.** The denoised-bitmap key and chi-squared confirm are generic; they
   should port, but re-tune thresholds for x86 noise.
5. **Boosting / lane layout.** `lanes_of` / `detecting_position_and_lanes` depend on how the
   x86 fuzzer stores boosted inputs; re-derive.
6. **Executor input encoding.** The batch you hand `measure` must be x86 executor inputs.

---

## 4. Gotchas / lessons (learned on the AArch64 side)

- **Unpinned + no-flush is required** for the residue to survive between lanes, which makes
  each measurement noisier → needs more reps. This is inherent, not a bug.
- **Cost.** Localization dominates wall-time on the AArch64 fork: each probe re-runs the whole
  prefix batch. Budget for it. The linear scan is `O(k)` runs, binary search `O(log k)`.
- **`cross_input_leaks_only` pathology.** With `cross_input_leaks_only=True` and a
  *self-dependent* leak, priming can never accept and re-localizes every flagged violation —
  a large blow-up. Be aware when choosing the flag.
- **The sample-size sweep is redundant** for cross-input priming (it self-verifies at the
  largest sample in one pass); the `_sample_size_sweep_needed` opt-out removes it. Keep that.
- **Determinism contract.** The CE pass, HW pass, and priming must rebuild the *identical*
  variant set for a given input — field/variant selection is a pure function of
  `(input identity, salt)`. Preserve this when porting.

---

## 5. Suggested order of work on x86

1. Read `priming_generalization.pdf` (the spec).
2. Copy `cross_input.py` and `boosted_lanes.py` (adjust imports/layout only).
3. Decide and implement **`IS`** for x86 + the regime mapping.
4. Wire `_make_cross_input_detector` to the x86 executor's batch `measure`.
5. Add the three config knobs; gate the slow-path sweep via the existing
   `_sample_size_sweep_needed`.
6. Validate on a known x86 leak (or a BTB/PHT template) end to end; check a boundary is
   returned and is genuine.

**Pointers:** proof `priming_generalization.{tex,pdf}` (repo root); reference code
`src/aarch64/{cross_input,boosted_lanes,aarch64_fuzzer}.py`, `src/fuzzer.py`,
`src/aarch64/aarch64_config.py`.
