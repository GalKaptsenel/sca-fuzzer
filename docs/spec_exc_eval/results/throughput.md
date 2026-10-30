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

**Reasonableness:** these are healthy rates for a JIT-hooked HW fuzzer — ~2–3.3 K device measurements
per second, 7–10 programs per second including generation, assembly, the CE resolve trace, and the
HW batch. The ~30% tc/s overhead is the price of the genuine/decoy sealing, and it buys 1.6× more
traces, which is where leak detection actually happens.

## Remote (Pixel 8 / Cortex-X3 over the ADB tunnel) — PENDING

Blocked: the ADB reverse tunnel is down (`adb devices` empty). Restore on the laptop
(`adb start-server`; `ssh -N -T -R 5037:localhost:5037 <gcp-host>`), then re-run the same three
configs with `executor: remote`. Remote tc/s will be lower (network round-trips per batch dominate);
the ratio between the three modes is the portable number.
