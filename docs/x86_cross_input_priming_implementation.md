# x86-64 cross-input priming — implementation report & pitfalls

**Audience:** the AArch64 Revizor agent (and anyone porting cross-input priming between
architectures). This complements `docs/cross_input_priming_handoff.md` (the AArch64→x86 handoff)
and `priming_generalization.{tex,pdf}` (the proof/spec). It records **how the x86-64 port was
actually built**, bundles the **full source of the new modules** and the exact **config/fuzzer
seams**, and documents **every pitfall hit** — including a real algorithmic inaccuracy fixed during
the port.

Everything referenced here is snapshotted under **`docs/x86_port_snapshot/`** so it can be read
without access to the live x86 tree:

| Snapshot file | What it is |
|---|---|
| `x86_port_snapshot/cross_input.py` | **NEW module** (359 L): lanes, `CrossInputFinding`, `GeneralizedPrimingDetector`, localizers. The architecture-independent core of the algorithm. |
| `x86_port_snapshot/cross_input_priming.py` | **NEW module** (425 L): `CrossInputPrimingCheck` (the fuzzer stage), the initialization-sequence (`IS`) context manager, the x86 measurement glue, `LaneCounterexample` report rendering. |
| `x86_port_snapshot/config_excerpt.txt` | The **config knobs** added to `rvzr/config.py`. |
| `x86_port_snapshot/fuzzer_seam_excerpt.txt` | The **clean seam** in `rvzr/fuzzer.py` (stage dispatch + `_cross_input_priming_check`). |

> Note: the live x86 tree also carries a throwaway *experiment patch* in `fuzzer.py` that logs a
> stock-vs-new comparison inline during a measurement campaign. That is **campaign scaffolding, not
> part of the port** — it is deliberately excluded from the snapshot.

---

## 1. What the port consists of

**Two new modules** + **config knobs** + **one fuzzer seam**. No executor-kernel changes were
needed beyond what Revizor's x86 executor already provides (batch `trace_test_case`, the per-batch
flush + warm-ups, and the ignore-list).

- `rvzr/cross_input.py` — pure algorithm, no I/O. Mirrors the AArch64 reference `cross_input.py`:
  lane layout, the splice `σ(t)=base[0..t)++toggle[t..d]`, the `GeneralizedPrimingDetector`, and the
  two localizers (`exponential_search` = "any", binary/optimal = "optimal").
- `rvzr/cross_input_priming.py` — the fuzzer-facing stage `CrossInputPrimingCheck` + the x86
  measurement seam: the **initialization sequence** context manager, the `measure/key/confirm`
  callables bound to the x86 executor, and the `LaneCounterexample` → violation-report renderer.
- `rvzr/config.py` — four knobs (see `config_excerpt.txt`):
  `enable_cross_input_priming` (bool, replaces standard priming),
  `cross_input_priming_localizer` (`"any"`|`"optimal"`),
  `cross_input_leaks_only` (bool, strictly-cross-input only),
  and `inputs_per_class` (default raised to **2**; the check **requires ≥2** — validated in
  `_check_config`, which also requires `enable_priming`).
- `rvzr/fuzzer.py` — the priming stage dispatches to `_cross_input_priming_check()` when
  `enable_cross_input_priming` is set (else the stock `_priming_check()`); and the `noise` /
  `priming_large` stages **skip the sample-size sweep** when cross-input priming is on (it
  self-verifies at the largest sample within its own search — see Pitfall 5).

## 2. How the code maps to the spec

- **Lanes.** With `inputs_per_class = L`, the boosted input sequence is `L` lanes of `n_classes`
  each; `lanes_of()` reshapes it. A "class" is one position across all lanes; its members are
  contract-equal (as far as taint pins them — Pitfall 2).
- **Detecting candidates.** `detecting_candidates()` finds the class positions where two lanes'
  read-outs disagree (the observed divergence), best-first. `_localize` tries **every** candidate
  and **both lane directions** (`a→b` and `b→a`), because position 0 has no history and an unstable
  read-out at one candidate can be fine at another.
- **Splice & probe.** For a fixed detecting class `d`, `GeneralizedPrimingDetector` measures
  `σ(t)=base[0..t)++toggle[t..d]` and reads class `d`'s trace `r(t)`. `r(0)` is all-toggle, the
  far end is all-base; they disagree, so some adjacent pair `r(t),r(t+1)` disagrees — that `t` is
  the leaking class (it differs in exactly one input: the two contract-equal members of class `t`).
- **Localizers.** `exponential_search` ("any") gallops from the base end — `O(log k)` and good when
  one boundary dominates; the "optimal" binary variant is proved optimal in the spec. Each splice is
  measured once (memoized `cache`), and a located boundary is **re-verified at `verify_reps`** (a
  larger sample) with the robust `confirm` test to reject jitter.
- **IS (initialization sequence).** `initialization_sequence()` makes every measured batch be
  preceded by the IS and nothing else: it ensures the per-batch flush is on and the ignore-list is
  cleared, but **does not** flush between inputs *within* a batch — that intra-batch cascade is the
  very channel cross-input priming reads.
- **Finding.** `CrossInputFinding(leaking_class, detecting_class, …)`; `is_self_dependent ==
  (leaking_class == detecting_class)`. `LaneCounterexample.full_str()` renders the two
  position-by-position contract-equal sequences for the violation report.

## 3. Pitfalls (read this before porting)

### P1 — `cross_input_leaks_only` + a self-dependent (or mixed) leak — the algorithmic bug fixed here
`cross_input_leaks_only` means "report only leaks whose secret is carried by an **earlier** input"
(`t<d`), excluding self-dependent ones (`t==d`, which standard priming already catches).

**The inaccuracy (naive implementation):** treat the flag as a **post-hoc filter** — localize over
the full chain `[0, d]`, then drop the result if the boundary came out self-dependent. This is
wrong because `exponential_search` **gallops from the all-base end and is biased to land on the
`t==d` boundary first**. Under the flag that boundary is filtered → **the whole violation is
discarded, even when a genuine cross-input boundary existed further down the chain** (`t<d`). So a
*mixed* leak (self **and** cross) loses its real cross-input finding, and a *pure self* leak is
mislabeled "false positive / noise" and triggers a full, wasted re-localization (a blow-up).

**The fix (what the snapshot does):** `exclude_self_dependence` **narrows the searched interval**,
it does not filter. `find_leaking_class` sets `hi = detecting if exclude_self_dependence else
detecting + 1`, and uses `r(detecting)` (the splice whose only toggled class is `d`) vs `r(0)` as
the interval ends:
- **pure self-dependent:** the narrowed ends **agree** (`r(0)==r(detecting)`) → return `None`
  in one comparison (cheap, correct, no blow-up), flagged `is_self_dependent`.
- **mixed (self + cross):** the narrowed ends **disagree** (the `t<d` toggle changes the read-out)
  → the localizer searches `[0, d)` and **returns the cross-input boundary**, ignoring the
  self-dependent one entirely.

**Completeness caveat (documented in the code):** if the narrowed ends *coincide*, no
strictly-cross-input boundary can be **localized** — but that does **not prove none exists**. A
history-folded predictor can hide an interior witness between coinciding ends; guaranteeing
detection there is inherently **linear** (exponential/binary search over `[0,d)` is not complete).
Keep this in mind if you need completeness rather than speed.

*Empirical confirmation on x86:* with `leaks_only=off` the check accepts both self-dependent and
cross-input leaks (it subsumes standard priming via the `t==d` end). With `leaks_only=on` it drops a
pure self-dependent leak (verified: input #22/#122, set 17) and still reports a genuine cross-input
one (verified: the control #42/#92, set 12) — exactly the intended split.

### P2 — a boundary is only a counterexample if its two inputs are contract-equal
Boosting makes lanes contract-equal position-by-position **only as far as the taint tracker pins
every contract-relevant input bit.** Under some observation clauses it doesn't — e.g. under `l1d`
**no branch condition is tainted at all** — so a class's two "members" may have *different* contract
traces, and toggling them proves nothing (a contract is free to let architecturally different inputs
leave different residue). The fix: `find_leaking_class` takes `is_valid_toggle` and **rejects a
located boundary unless the boundary class's two members are genuinely contract-equal**
(`_contract_equal_toggle` compares their ctraces). Do not drop this check when porting.

### P3 — the channel requires NO intra-batch flush (and that makes it noisy)
The residue must survive between inputs inside one measured sequence, so the IS flushes **once per
batch** and runs warm-ups, but must **not** flush between the inputs of a sequence. Consequence:
each measurement is noisier → needs more reps. This is **inherent, not a bug**. Budget sample size
and `verify_reps` accordingly, and use the robust `confirm` (not raw equality) everywhere.

### P4 — determinism contract
The contract-equality pass, the HW pass, and priming must rebuild the **identical** variant set for
a given input. Variant/field selection must be a **pure function of `(input identity, salt)`**.
If any of the three passes disagrees on the variant set, the ignore-list indices and the splices
no longer line up and the localizer chases noise. Preserve this when porting.

### P5 — the sample-size sweep is redundant for cross-input priming
Cross-input priming self-verifies at the largest sample in one pass (`verify_reps`). So the fuzzer's
`noise`/`priming_large` sweep stages are **skipped** when `enable_cross_input_priming` is set (see
`fuzzer_seam_excerpt.txt`). Keep that opt-out — otherwise you re-measure for nothing.

### P6 — try all detecting candidates and both directions
Position 0 has no history; an unstable read-out at one candidate yields nothing even when another
position would localize cleanly. `_localize` iterates **all** candidates, each from **both** lanes
as base. A single arbitrary choice is often unusable.

### P7 — "DCU-IP-gated" ≠ "information-flow cross-input" (interpretation pitfall)
During x86 validation we repeatedly conflated two different notions of "cross-input":
1. **microarchitectural** — the leak needs predecessor inputs present to *train* a prefetcher (e.g.
   a stride predictor needs K≥4 warm-up accesses);
2. **information-flow** — the discriminating **secret** lives in an *earlier* input (`t<d`).
A leak can be (1) without being (2): the prefetcher merely *carries/trains*, while the secret is in
the detecting input itself (`t==d`, self-dependent). Cross-input priming keys on (2), **not** (1).
So a prefetcher-carried, predecessor-trained leak whose secret is in the detecting input is
correctly **not** reported by `leaks_only` — that is not a bug in the algorithm, it is the scope.
Verified example: input #22/#122 set-17 is DCU-IP-stride-carried and needs K≥4 predecessors, yet its
secret is a single byte of the *detecting* input → `t==d` → `leaks_only` correctly stays silent.

### P8 — validation-side traps (not in the algorithm, but they bite)
- **Weak near-threshold signals flip classification.** A single-pass, single-threshold "present/
  absent" test on a weak channel (≈0.5) straddling the cutoff mis-reads as present on some reps and
  absent on others, producing false "clean cross-input" findings. Use a **margin gate** (present
  ≥0.6 **and** absent <0.25) and average over passes before claiming a boundary is real.
- **The contract model over-approximates.** At high `model_max_nesting` (e.g. 30) the emulator
  reaches sets via deep nested speculation that real bounded hardware never takes; a set being
  "named" by the model does not mean a demand access produced it on HW. Confirm the true carrier on
  hardware (e.g. NOP the suspected instruction and re-measure) rather than trusting the model.
- **Prefetcher identity.** To attribute a signal to a specific prefetcher, use the disable-bit
  sweep (MSR 0x1a4): a signal present with only the IP-stride bit enabled and **absent** when the
  next-line/adjacent bit is enabled but IP-stride disabled is IP-stride, not next-line — even if its
  learned stride is one cache line and looks like next-line.

## 4. Validation status on x86

The port runs end-to-end on x86 and localizes genuine boundaries. Representative confirmed cases
(DCU IP-stride prefetcher leaks, Intel i5-12500):
- **Genuine cross-input (`t<d`)** — reported by `leaks_only=on` only with the prefetcher enabled:
  the control (classes #41/#91 → #42/#92, carrier set 12) and campaign finds #14/#114, #62/#162.
- **Self-dependent (`t==d`)** — correctly reported by stock/`leaks_only=off`, dropped by
  `leaks_only=on`: #22/#122 (set 17, pure, no v1), #75/#175 (set 5, with coexisting v1).
These match the intended semantics of §3-P1 exactly.

## 5. How to reuse this
1. Read `priming_generalization.pdf` (spec) and `docs/cross_input_priming_handoff.md` (seams).
2. Diff your AArch64 `cross_input.py` against `x86_port_snapshot/cross_input.py` — the algorithm
   core should be identical; only the `measure` binding differs. **In particular verify your
   `find_leaking_class` narrows the interval (P1) rather than post-filtering.**
3. Apply the config knobs (`config_excerpt.txt`) and the fuzzer seam (`fuzzer_seam_excerpt.txt`).
4. Honor P2 (`is_valid_toggle`), P3/P4 (IS + determinism), P5 (sweep opt-out).
