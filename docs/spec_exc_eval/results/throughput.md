# Fuzzer throughput — regular vs speculative-exception (strict / contract_driven)

Fuzzing **throughput** of the three modes on the same machine and a workload-matched config, so the
cost of the speculative-exception instrument (sealing + genuine/decoy variants + the CE resolve trace)
is isolated. This is the *fuzzer's* speed, a different metric from the covert-channel bits/s in
`rate_table.md`.

## Two numbers, and why they diverge

- **tc/s** — test cases generated + measured per second. One test case = one program; the fuzzer
  generates it, then measures a batch of inputs through it.
- **traces/s** — hardware measurements per second (one trace = one input run on the device).

A regular (`basic`) test case measures its input batch once. A **speculative-exception** test case
measures each input **as several variants** — the genuine seal image plus decoy images (and, in
contract_driven, the boosted priming inputs) — so it runs *more traces per test case*. Hence spec-exc
has **lower tc/s but higher traces/s**: fewer programs per second, more hardware measurements per
second. Both are expected and consistent.

## Local (Neoverse N3, `/dev/executor`, 2026-10-30)

All three: identical workload — same instruction categories, 5/5 basic blocks, clause `[cond, bpas]`,
`-n 400 -i 10 --nonstop` (the `--nonstop` matters: strict/cd find canonicality violations and would
otherwise stop early, truncating the run; basic finds none). Vary only `fuzzer` + `speculative_exception_mode`.

| mode | fuzzer / mode | test cases / duration | **tc/s** | **traces/s** |
|---|---|---|---:|---:|
| regular | `basic` | 400 / 39.9 s | **10.03** | 2011.7 |
| strict | `speculative-exception`, strict | 400 / 56.9 s | **7.03** | 3286.8 |
| contract_driven | `speculative-exception`, ct-bpas | 400 / 53.5 s | **7.48** | 3234.5 |

Spec-exc is **~0.70–0.75× the tc/s of regular** but **~1.6× the traces/s** — the instrument trades
programs-per-second for measurements-per-second because every seal site turns one input into a genuine
+ decoy family. contract_driven is marginally faster than strict (7.48 vs 7.03 tc/s): its boosting
re-randomises only the untainted lanes, slightly cheaper than strict's full seal-variant expansion.

**What regime these numbers reflect (read before quoting them).** Almost every test case here is
discarded via the **fast path** — regular 399/400, strict 393/400, contract_driven similar — meaning
the test case was measured once on hardware, showed no input-dependent htrace difference, and was
dropped before the expensive confirmation pipeline (priming / boosting / large-sample). So these are
**clean-test-case throughput** numbers (the common case when the run is not actively confirming a
violation), NOT full-pipeline throughput — a run spending time confirming real violations would be
slower. The comparison is still apples-to-apples because all three modes fast-path at essentially the
same rate; the mode difference is entirely in traces-per-test-case: regular ≈200 (2012 ÷ 10.03),
strict ≈467 (3287 ÷ 7.03), contract_driven ≈432 (3235 ÷ 7.48) — spec-exc measures ~2.3× more
hardware traces per test case (the genuine + decoy variants), which is exactly where its lower tc/s
comes from. Every trace counted is a real HW measurement; nothing is skipped on one side only.

**Reasonableness:** these are healthy rates for a JIT-hooked HW fuzzer — ~2–3.3 K device measurements
per second, 7–10 programs per second including generation, assembly, the CE resolve trace, and the
HW batch. The ~30% tc/s overhead is the price of the genuine/decoy sealing, and it buys 1.6× more
traces, which is where leak detection actually happens.

## Remote (Pixel 8 / Cortex-A715 over the ADB reverse tunnel, 2026-10-30)

Same three configs with the remote-execution block added (transport `adb`, device core 4), driven
from the GCP fuzzer across the reverse SSH tunnel (GCP → laptop adb → USB → Pixel 8), device
`47061FDJH001K3`. `-n 120 --nonstop` (remote is network-bound, so per-batch cost is constant and
steady state is reached quickly; 120 is representative). traces/s required wiring the trace counter
into the remote backend (`RemoteHWExecutor.run_batch`); it was previously only counted on the local
path, so earlier remote reports printed 0.0.

| mode | fuzzer / mode | test cases / duration | **tc/s** | **traces/s** | HW errors |
|---|---|---|---:|---:|---:|
| regular | `basic` | 120 / 140.5 s | **0.85** | **170.9** | 0 |
| strict | `speculative-exception`, strict | 120 / 109.5 s | **1.10** | **377.5** | 0 |
| contract_driven | `speculative-exception`, ct-bpas | 120 / 121.6 s | **0.99** | **197.3** | 0 |

**Remote is ~8–13× slower than local** in tc/s — each input batch is an ADB round-trip over the
two-hop tunnel, and that latency, not the CPU, sets the pace.

**On remote the tc/s of the three modes nearly converge** (0.85 / 1.10 / 0.99), the opposite of
local: wall-clock is dominated by the per-batch round-trip, so a spec-exc test case's extra
genuine/decoy variants are largely absorbed into round-trips that happen anyway. The cost of the
sealing instrument shows up as compute (lower local tc/s) but is nearly free on a latency-bound link.
The **traces/s ordering strict (378) > cd (197) > regular (171)** matches local — strict runs the
most hardware measurements per test case — confirming the variant cost is real and just hidden from
tc/s by the network.

**Run-to-run variance (remote).** Unlike local, remote tc/s drifts between runs (a prior run measured
0.80 / 0.81 / 1.00 for the same three) because tc/s depends on how many test cases hit the expensive
slow path, which depends on the random violations a run finds (no fixed `input_gen_seed`). Likewise
traces-per-test-case is NOT a config constant (remote cd ≈199 here; local cd ≈432) — it tracks the
per-run fast/slow-path mix. Hence traces/s is measured per run, never derived from local. All runs
clean (0 HW errors); strict found 1 canonicality violation (expected — completed all 120 under
`--nonstop`).
