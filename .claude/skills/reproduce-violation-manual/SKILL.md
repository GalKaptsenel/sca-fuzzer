---
name: reproduce-violation-manual
description: Reproduce/verify ANY Revizor violation on hardware with the controlled batch-context protocol — load the test case's full input batch once, swap ONLY the leaking slot between the counterexample inputs, and confirm the htrace divergence persists (genuine) or washes out (noise/batch-contamination). Runs local or remote (whatever reproduce.yaml selects) via verify.py, or by hand through /dev/executor with executor_userland. Handles REIF and legacy bin inputs, 2..V counterexample inputs, and multiple swap positions. Pair with revizor-violation-triage (predicts the leaking sets first).
---

# Reproducing / verifying a Revizor violation on hardware

A violation = V counterexample inputs that are contract-equivalent yet produce divergent hardware
traces. To confirm it's real (not measurement noise) you must recreate the **exact same
micro-architectural context** for every one of the V inputs, vary only the one slot under test, and
check the htrace **distribution** divergence is stable, input-determined, and lands on the sets
**revizor-violation-triage** predicted. Do the contract classification there first.

## THE pitfall this defends against — batch/prefetcher contamination
An input's htrace depends on its **predecessors within the same test case's input batch** (the
prefetcher/BPU state carries across the super-batch). The executor **resets between test cases**, so you
do NOT reiterate other test cases — only this TC's inputs. Measuring the counterexample inputs *in
isolation* gives them different contexts and huge spurious differences. Always fix one prefix and swap
only the tested slot.

## Protocol (tool: `verify.py`)
`verify.py <violation-dir> [--position lower|higher|last|all] [--reps N] [--sets s1,s2,...] [--inputs A,B,...]`
- Loads the **whole** input batch (`input_*.reif`/`.bin`) in order + the test case, once.
- `prefix = slots 0..P-1` (identical for every measurement); **swaps only slot P** between the V inputs;
  one batch `trace_test_case` call each; records the last slot's htrace **distribution** (per-set
  frequency over all reps — not just the dominant pattern).
- **Positions** (try several — a real leak persists at all):
  - `lower` → P = min(counterexample idx): prime up to the first leaking slot.
  - `higher` → P = max: prime up to the last leaking slot.
  - `last` → P = N-1: all iids as context, most symmetric.
  - `all` → runs lower+higher+last and cross-checks.
- Runs **local or remote** exactly as `reproduce.yaml`'s `executor_*` selects — no separate path.
- Pass `--sets` the triage-predicted sets; the verdict prints whether the persistent difference lands on
  them. Bin numbering: raw htrace int bit b == cache **set b** (identity); triage.py already reversed the
  `report.txt` MSb-first strings (position p = set 63-p) into set space, so `--sets` are set indices.

## Verdict
- Persistent per-set spread (≥3% over high reps) **on a triage-predicted set** → **GENUINE**.
- No set spreads / spread only off the predicted sets / washes out with reps → **NOISE / contamination**.
- Real leaks are **intermittent** (misprediction fires part of the time) — a partial-frequency
  difference IS the signal; don't expect every rep to fire, and DON'T "confirm noise" by naive isolated
  high-rep repetition (that trains the BPU and hides Spectre-v1 — see revizor-violation-triage pitfall).

## Assertions verify.py makes (correctness)
Identical prefix across all V measurements; `1 ≤ P < N`; ≥2 counterexample inputs; inputs exist. It
does not silently fall back — it errors on a bad position/missing files. When in doubt also sanity-check
the **channel** first (P+P prime correctness): every CE architectural set must show as a hot htrace bit
at freq ≈ 1.0 with ~0 spurious hot bits; a noisy prime fabricates violations.

## Regime matters
Measure under the **same regime** the violation was found (`executor_mode` P+P vs F+R, `enable_pre_run_flush`).
A leak can be strong under one and invisible under the other — notably the kernel **P+P prime trains the
BPU and can hide Spectre-v1**, so a genuine F+R leak may look absent under the stock P+P prime (use a
lean single-loop prime, or L2-sized prime, to recover it).

## By hand (no Python) — executor_userland
When you need to drive `/dev/executor` directly, use the **executor-userland** skill. Same protocol:
`CLEAR_ALL_INPUTS`; for k in 0..P-1 `ALLOCATE`→`CHECKOUT`→`WRITE input_k`; `LOAD_TEST_CASE`→`WRITE tc`;
then per counterexample input: `CHECKOUT slot P`→`WRITE input_V`→`TRACE`→`MEASUREMENT slot P`, read the
htrace bit for the leaking set (**set N → report-string char 63-N**), repeat for statistics.
- Reload the module first so `ALLOCATE` hands out 0,1,2,… (the iid counter persists).
- Set the regime in sysfs **before** loading (`measurement_mode`, `enable_pre_run_flush 0`).
- Never assume input ids — parse them from `ALLOCATE`. Don't run while a Python campaign is live.
- Legacy bin inputs need the per-flag NZCV slot (byte 0x2030) reconstructed to PSTATE; REIF inputs
  already carry it (`input_gen.load` handles both).

## Cross-references
- **revizor-violation-triage** — CE contract classification (run FIRST; gives the predicted leaking sets).
- **executor-userland** — the raw /dev/executor CLI.
- **reproduce-spectre-v4** — the store-bypass (v4) argument end-to-end.
