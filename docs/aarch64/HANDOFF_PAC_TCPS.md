# Handoff: diagnose the PAC-fuzzing throughput (tc/s) drop on the physical box

You are a fresh Claude Code session running on **revizor-physical** (a Google Compute Engine
`bondibeach` instance: ARM **Neoverse-N3**, a **real PMU** so `measurement_supported = 1`, and
**FEAT_PAC + FEAT_FPAC** present). Your job is to **find the root cause of the large reduction in
test-cases-per-second (tc/s) when fuzzing with PAC**, compared to non-PAC fuzzing, and propose (and,
if agreed, implement) a fix.

This document gives you everything you need: the theory behind Revizor, the concepts (contracts,
htraces, PAC, the seal), the codebase architecture, the task, a measurement plan, and the safety rules
for this box. Read it fully before touching anything.

> **This task is a dedicated, project-critical goal — treat it as deciding the success or failure of
> the project.** The PAC-fuzzing throughput is not a nice-to-have: if PAC fuzzing is too slow to cover
> a meaningful test-case volume, the PAC leak search never reaches statistically useful coverage and
> the whole PAC effort fails. Give it the weight that implies.
>
> **No handwaves. Be thorough — this is extremely important.** Do not guess, do not assert a cause
> from plausibility, do not stop at "this is probably it." Every claim about where the time goes must
> be backed by a *number you measured on this box* (a timer, a counter, a profile, a traced kernel
> event). If you cannot measure something, say so explicitly and say why, rather than papering over it.
> A ranked suspect list is a starting point to *test*, not a conclusion to *repeat*. Refute or confirm
> each suspect with evidence, attribute the wall-clock quantitatively, and only then propose a fix.

---

## 0. First things first — safety and setup

**PAC fuzzing executes real `AUT*` instructions on hardware. On FEAT_FPAC, an `AUT*` on a wrong
signature raises a synchronous fault at EL1 → kernel panic → the box reboots.** Two seal bugs used to
cause this; both are now fixed upstream. Before you run anything:

1. `git pull` and make sure you have commits **`9d8cfdd`** (CE base-register fix) and **`9ce1662`**
   (NI executor seals the filter/generic HW path). Without these, PAC fuzzing WILL reboot the box.
2. Rebuild both binaries after pulling:
   - kernel module: `make -C src/aarch64/executor` then reload it (`sudo rmmod revizor_executor;
     sudo insmod src/aarch64/executor/revizor-executor.ko`), and `sudo chmod -R a+rwX /sys/executor
     /dev/executor`.
   - contract executor: `make -C src/aarch64/contract_executor contract_executor`.
3. The human operator is on the **serial console** of this box — if it panics, they can read the
   register dump and it auto-reboots (`/proc/sys/kernel/panic = 10`). Still, treat a reboot as
   expensive: prefer measurement/profiling over blind runs.
4. This is a **performance** investigation, not a safety one — but you are running the real PAC path,
   so keep the fixes in place and don't disable the seal.

Run tests / the fuzzer with the project venv:
`/home/gal_k_1_1998/revizor/revizor-venv/bin/python` (the system python lacks xxhash/numpy).

---

## 1. What Revizor is and the theory behind it

**Revizor is a fuzzer for microarchitectural information leaks** — Spectre-class vulnerabilities where
a CPU leaks secrets through its microarchitectural state (caches, predictors) even though the
*architectural* result is correct.

It is built on **hardware-software leakage contracts**. A *contract* is a formal specification of what
a program is *permitted* to leak to a microarchitectural attacker. Example: "a program may leak the
*addresses* of its memory accesses and its control flow, including down mispredicted branches, but
nothing else." If the real hardware leaks *more* than the contract allows, that gap is a potential
vulnerability — a **contract violation**.

Revizor finds violations by **Model-based Relational Testing (MRT)**:

1. **Generate** a random test case (a short assembly program) and a batch of random **inputs**
   (initial register + memory state).
2. For each input, compute the **contract trace (ctrace)** with a software **model** (the *contract
   executor*, CE). The ctrace is the sequence of contract-permitted observations (e.g. the set of
   accessed addresses, the taken branch directions), *including* the speculative paths the contract
   says are observable (e.g. one level of branch misprediction).
3. Group inputs into **contract-equivalence classes**: inputs with the *same* ctrace. The contract
   says these inputs are **indistinguishable** to any attacker.
4. For each input, run the test case on **real hardware** and measure the **hardware trace (htrace)**
   — the actual microarchitectural footprint left by the run.
5. **Violation:** two inputs in the same contract class (same ctrace) but with **different htraces**.
   The hardware distinguished inputs the contract said were indistinguishable → it leaks more than the
   contract permits.

Key point for performance: the contract lets Revizor **boost** — generate many inputs but only run the
*model* (expensive) on enough of them to populate classes; inputs proven equivalent by the contract
don't each need independent contract tracing. (This matters below.)

---

## 2. Core concepts you'll see in the code

- **Contract trace (ctrace):** the model's permitted-observation trace for one input. Built by the
  CE from an **observation clause** (what is observed: L1D cache sets, PC, memory addresses, ...) and
  an **execution clause** (what speculation is modeled: `COND` = conditional-branch misprediction,
  `BPAS` = store-to-load bypass, etc.). Default here is the `l1d` observation over `COND` speculation.

- **Hardware trace (htrace):** a measurement of the microarchitectural state after running the test
  case on hardware. On this box it's an **L1D cache-set bitmap** collected by **Prime+Probe**: prime
  every cache set, run the test case, probe which sets were evicted → one bit per set, packed into a
  64-bit word. Collected together with **PFCs** (performance counters: instructions retired,
  instructions speculated, branch mispredictions).

- **The contract executor (CE):** `src/aarch64/contract_executor/` — a **userspace software model**
  that single-steps the test case, tracks architectural + speculative state, and emits the per-
  instruction register/memory trace and the ctrace. It talks to the Python side over a **persistent
  pipe** (the `RVZR` stream protocol — see `unit_stream_ipc`), so it is a long-lived subprocess, not a
  fork per trace. It **emulates PAC** via kernel ioctls (it never issues a real `AUT*`, which could
  fault the CE process).

- **The kernel executor:** `src/aarch64/executor/` (`/dev/executor`) — loads a test case + inputs,
  runs the **measured** execution on real hardware inside a JIT'd Prime+Probe harness, returns
  htraces + PFCs. Driven by ioctls (allocate/checkout/write input, load test case, TRACE, MEASUREMENT
  — see the `executor-userland` skill for the full protocol).

---

## 3. PAC and the "seal" — the AArch64-specific machinery

**PAC (Pointer Authentication, ARMv8.3):** `PACIA/PACDA/...` sign a pointer by inserting a MAC into
its unused top bits; `AUTIA/AUTDA/...` authenticate and strip it; `XPACI/XPACD` strip without
checking. **FEAT_FPAC:** a failed `AUT*` raises a synchronous fault. The security question Revizor
asks: **does PAC leak speculatively?** (e.g. PACMAN — can misspeculation reveal whether a signature
was correct?).

**The seal.** To run a test case containing `AUT*` on hardware without an FPAC reboot, every
architecturally-reachable `AUT*` must authenticate. Revizor **seals** the test case:
- It runs a CE trace to learn each pointer's value at each `AUT*`, computes the *correct* signature via
  the kernel `PAC_SIGN` ioctl (same keys the measured run uses), and inserts a `MOVK` that stuffs that
  signature into the pointer's top bits right before the `AUT*`. Now the `AUT*` always succeeds
  architecturally.
- Sealing slots are **register-only** (a `MOVK` + the `AUT*`), so they don't perturb the cache
  footprint — the htrace is unaffected.

**Non-interference (NI) fuzzing** (`fuzzer: non-interference`, what `config_pac.yml` uses): rather than
contract-vs-hardware, it compares **two hardware runs** that should be indistinguishable if PAC
doesn't leak speculatively:
- a **baseline** ("genuine") variant: every slot correctly signed;
- a **decoy** variant: the *speculative-only* slots get a *wrong* signature (a forgery that only ever
  executes down a mispredicted path, so it never faults architecturally).
If baseline and decoy produce **different htraces**, the speculative wrong-signature was observable →
a speculative PAC leak.

Relevant code:
- `src/aarch64/seal/sealer.py` — the sealing pipeline: `Sealer.seal(tc)` inserts placeholders;
  `SealedTestCase.resolve(input)` runs a **CE trace per input** and computes each slot's value;
  `ResolvedSealingTestCase.genuine()/decoy()` mint the two hardware variants.
- `src/aarch64/aarch64_executor.py` — `Aarch64NonInterferenceExecutor`:
  `trace_test_case_with_taints(inputs, nesting)` (builds the variants from CE traces),
  `_run_sealed_on_hw(...)` (the single sealed hardware entry point), `trace_test_case` (filter/generic
  path — runs the genuine baseline), `_seal_trace` (one CE trace).

---

## 4. THE TASK: why is PAC tc/s so much lower?

Fuzzing throughput (test cases per second) drops sharply under `config_pac.yml` (NI + PAC) versus a
plain non-PAC config (e.g. a spectre_v1 config). **Find the dominant cause, prove it with
measurement, and propose/implement a fix.** Do NOT just theorize — profile.

### Ranked suspects (from prior analysis — confirm or refute each with numbers)

1. **No boosting on the NI path.** The NI/sealed executor disables fast boosting
   (`_supports_fast_boosting` returns `False` for the sealed executor), because every input resolves
   to its own per-input sealed test case. So the **model runs on every input**, not just class
   representatives — where non-PAC fuzzing traces only a few. This is likely the biggest factor.
2. **Per-input CE traces, ×2.** For each input the pipeline runs a CE trace to *resolve* the seal
   (`SealedTestCase.resolve` → `_seal_trace`), plus the placeholder/genuine trace. That's on the order
   of `2 × n_inputs` CE round-trips per test case.
3. **Per-(input × variant) assembly + device writes.** Each variant TC is assembled and written to
   `/dev/executor` per input; NI runs **2 variants** (baseline + decoy) per input, so 2× the
   assemble+write+trace work.
4. **Un-batched `trace()`.** The base executor batches inputs that share a branch-mistraining config
   into a single `trace()` (the kernel measures many loaded inputs in one run). The sealed path runs
   **per input** because each input has a *different* sealed TC, so it cannot batch — losing that
   amortization.
5. **Per-input `PAC_SIGN` ioctls** during seal resolution, and **assembler forks** (`asm_to_bytes` /
   `in_memory_assemble`) if any path shells out to an assembler instead of encoding in-memory.
6. **ICACHE flushes fire far more often on the sealed path — and each one is an SMP-wide IPI
   broadcast.** This is a strong suspect; work it out fully rather than trusting the sketch below.
   The mechanism, traced through the kernel module:
   - **Every** `trace()` (`chardevice.c:281`, one per repetition) calls
     `load_jit_template()` (`templates_jit.c:611`), which **rebuilds the entire measurement harness**
     into the JIT buffer via `build_measurement_code()` and re-maps it. Each build flips the buffer
     RW→RX through `jit_perm_rw()`/`jit_perm_rx()` (`templates_jit.c:477/512/530/562`), and **each of
     those calls `flush_icache_range()` over the whole `MAX_MEASUREMENT_CODE_SIZE`**
     (`jit.c:192`/`jit.c:214`).
   - `load_jit_template()` then adds a loop of `MAX_MEASUREMENT_VIEWS − 1` more `flush_icache_range()`
     calls over the JIT'd range, one per aliased view (`templates_jit.c:645-649`).
   - On arm64, a non-local `flush_icache_range()` broadcasts an **IPI to every CPU
     (`kick_all_cpus_sync`) and waits** — this is the dominant per-flush cost. Note the module already
     has a cheap `local_icache_flush()` (dc cvau / ic ivau, no IPI — `bpu.c:104-118`) that it uses
     during branch training, but `load_jit_template()` does **not** use it; it goes through the
     IPI-broadcasting path.
   - **The redundancy:** `load_jit_template()` runs on every `trace()`, but `_run_sealed_on_hw`
     (`aarch64_executor.py:816-828`) calls `trace()` `n_reps` times per `(input, variant)` with
     **byte-identical** test-case bytes. Reps 2..`n_reps` rebuild and re-flush an *identical* harness
     for nothing. The JIT output is a pure function of `(test_case bytes, template)` — neither changes
     across reps.
   - **Why the sealed path is worse than plain fuzzing:** `trace()` is invoked
     `n_inputs × n_variants × n_reps` times on the NI path (no boosting, 2 variants — suspects 1 & 3),
     versus `n_classes × n_reps` on a boosted plain run. Every one of those extra `trace()` calls drags
     the full rebuild-and-IPI-flush sequence with it, so this suspect *multiplies* suspects 1, 3 and 4.
   - **Confirm with numbers**, don't assume: count `load_jit_template` invocations per second and
     total `flush_icache_range` / `kick_all_cpus_sync` calls and cycles on each path
     (`funccount`/`funclatency`/`bpftrace` on `flush_icache_range`, `__flush_icache_range`,
     `kick_all_cpus_sync`, or an ftrace function profile), and attribute the delta. If the flush cost
     is dominant, the fix is to **memoize the JIT'd harness** — skip `load_jit_template()` when neither
     the loaded test-case bytes nor the template changed since the last build (a dirty flag set on TC
     write / template set), so the `n_reps` repeats reuse the already-mapped RX harness — and/or route
     the required flushes through the CPU-pinned `local_icache_flush()` instead of the IPI broadcast.

### Measurement plan

- **Compare apples to apples:** run a fixed number of test cases with `config_pac.yml` and with a
  non-PAC config that has the *same* `min/max_bb_per_function`, `*_successors_per_bb`, `-i` inputs,
  and `-n` test cases. Record tc/s for each.
- **Phase timing:** instrument (or wrap) the per-test-case phases and print wall-clock per phase:
  generation, seal build (`Sealer.seal`), per-input CE resolve (`resolve`/`_seal_trace`), assembly,
  HW trace (`_run_sealed_on_hw`), analysis. `src/util.py` has `STAT`; add timers there or use a
  `cProfile`/`perf` run of `revizor.py fuzz`.
- **Count the CE round-trips and device writes per test case** (add counters), and multiply by the
  measured cost of one CE trace and one `trace()` to attribute the time.
- **CE cost:** confirm the CE is the persistent-pipe subprocess (it should be — see
  `unit_stream_ipc`), so the cost is per-trace single-stepping + IPC, not process spawn. Time one CE
  `run()`.

### Where to look (files)

- `src/aarch64/aarch64_fuzzer.py` — the fuzzing round (`filter`, `_supports_fast_boosting`).
- `src/fuzzer.py` — the generic round: generation, boosting, the model/HW calls.
- `src/aarch64/aarch64_executor.py` — `trace_test_case_with_taints`, `_run_sealed_on_hw`,
  `trace_test_case`, `_seal_trace`, `read_base_addresses`.
- `src/aarch64/seal/sealer.py` — `resolve` (per-input CE trace), `genuine`/`decoy`.
- `src/aarch64/contract_executor/` + `src/aarch64/aarch64_kernel.py` — the CE IPC and the ioctls.
- `config_pac.yml` — the PAC/NI config; compare against a non-PAC config.

### What a good result looks like

A phase breakdown **backed by measured numbers** that attributes the tc/s drop (e.g. "78% of
wall-clock is per-input CE resolve because boosting is off; each TC does 2·I CE traces" — with the
timer output that proves it), plus a concrete, low-risk optimization — candidates to evaluate:
**memoize the JIT'd harness** so identical `(test-case, template)` reps skip the rebuild + IPI icache
flush (suspect 6); **cache/reuse the CE resolve** so `trace_test_case_with_taints` and any later
re-resolve share one trace per input; **run baseline+decoy in one `trace()`** where the kernel allows
multiple loaded TCs; **skip the decoy** in phases that don't need it; **contract-class the sealed TCs**
(inputs whose resolved slot values match share one sealed TC — `ResolvedSealingTestCase.collapse_key`
already exists for this; the regular-sealed executor uses it, the NI executor may not). Verify any
change keeps the seal correct (no raw/wrong-signed `AUT*` reaches hardware) and doesn't change which
violations are found.

**The bar:** a plausible story is not a result. The result is the *measured* attribution — where the
cycles actually go on this box — and a fix whose effect you *re-measured* (tc/s before vs after). If
the numbers contradict the ranked suspects above, trust the numbers and say so. Remember the stakes:
this determines whether PAC fuzzing is viable at all.

---

## 5. Recent context you should know

- Two FPAC-reboot bugs were just fixed (pull them, §0): **#1** the NI executor's filter/generic HW
  path used to run the *unsealed* sandbox-only test case (a raw `AUT*` → FPAC); it now runs the sealed
  genuine baseline through `_run_sealed_on_hw`. **#2** the CE's base-register `kaddr↔uaddr` fixup
  corrupted a non-canonical register's top byte, so the seal signed an `AUT*` with the wrong
  pointer/context and hardware FPAC'd; fixed with a delta-based register restore.
- Those were *correctness/safety* fixes. The tc/s question is *performance* and (as far as we know)
  independent — but you must run with the fixes so the box doesn't reboot mid-measurement.
- Investigation scaffolding and notes from the correctness work live under
  `~/fpac_investigation/` on the dev box (not this one); you don't need them, but the methodology
  (profile first, measure per-phase, compare PAC vs non-PAC) is the same.

Good luck. Measure first, then optimize.
