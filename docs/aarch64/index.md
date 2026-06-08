# Revizor on AArch64 — Reference Manual

## About this document

A reference for the AArch64 port of Revizor: its components, interfaces, operation, and
limitations. Explanations are layered — a short definition, then an *Under the hood* block for
detail; read as deep as you need. New terms are **bold** on first use and defined in the
[Glossary](#glossary).

Generated from a single Markdown source at `docs/aarch64/index.md`. Edit that file, then
rebuild from the `docs/aarch64/` directory with `make` (or `python3 build_docs.py`).

## 1. What Revizor does

Revizor is a *microarchitectural fuzzer*: it searches for cases where a CPU leaks more through
its microarchitecture than a **[contract](#glossary)** permits. A contract states the only ways
a program's data may influence attacker-observable state (here, the data-cache footprint).

For each generated program, two traces are compared:

- the **[contract trace](#glossary)** (**ctrace**) — what the contract permits to be observable,
  computed by a model;
- the **[hardware trace](#glossary)** (**htrace**) — what the real CPU actually leaked, measured
  as a 64-bit cache-set bitmap.

A **[violation](#glossary)** is a pair of inputs that the contract treats as identical
(**equal ctrace**) but the hardware tells apart (**different htrace**). In words: the contract
promised the two inputs would look the same to an attacker, yet the CPU produced two different
cache footprints — so it leaked something the contract forbade (for example, a Spectre-style
speculative access). The contract/trace framework originates in Guarnieri et al.,
*Hardware-Software Contracts for Secure Speculation* (IEEE S&P 2021).

> **Intuition**
>
> - **ctrace** — what an attacker is *allowed* to observe.
> - **htrace** — what an attacker can *actually* observe.
> - **Violation** — the hardware revealed something the contract forbade.

*Under the hood.* Leaks are about *distinguishing* secrets, so Revizor runs **many inputs**
through one program and searches for a pair that is contract-equal yet hardware-distinct.
**[Taint](#glossary)** marks the input bytes the traced execution actually reads; **[boosting](#glossary)**
then cheaply mass-produces contract-equal inputs by varying only the *other* bytes — the ones
the contract trace cannot depend on (§3).

## 2. Components at a glance

| Component | Language / location | Role |
|---|---|---|
| Fuzzer | Python (`src/`) | Orchestrates generate → model → measure → compare. |
| Generator | Python (`src/aarch64/`) | Emits random AArch64 test cases from the ISA spec. |
| Contract executor (**CE**) | C process (`src/aarch64/contract_executor/`) | Models a program to produce the ctrace. |
| Kernel module | C (`src/aarch64/executor/`) | Runs the test case on the real CPU in a sandbox; measures the cache. |
| `asm_to_bytes` | C (`src/aarch64/asm_to_bytes/`) | Assembles instruction text to encodings, in memory. |
| ISA spec tooling | Python | Downloads/parses the instruction set the generator uses. |

Interfaces, the communication map, and a full end-to-end run are in Part II.

## Glossary

| Term | Meaning |
|---|---|
| **Contract** | Promise of what data may influence observable state. Has an *execution clause* (what the model simulates, e.g. always-mispredict) and an *observation clause* (what is observable, e.g. cache sets). |
| **ctrace** | Contract trace — what the model says is observable for a program+input. |
| **htrace** | Hardware trace — what the CPU actually leaked; a 64-bit cache-set bitmap. |
| **Violation** | Two inputs with equal ctrace but different htrace. |
| **CE** | Contract executor; produces ctraces by modelling the program, including speculation. |
| **Taint** | The input bytes the contract trace depends on — those actually read along the traced execution paths. |
| **Boosting** | Cheaply producing more inputs with the *same* ctrace by mutating only the bytes the trace does not depend on (the untainted ones). |
| **Nesting** | Depth of speculative execution the model explores. |
| **Prime+Probe (P+P)** | Fill the cache with attacker-controlled lines, run the victim, then detect which lines were evicted. |
| **Flush+Reload (F+R)** | Flush lines shared between attacker and victim, run the victim, then detect which lines were reloaded back into the cache. |
| **Cache set** | One of 64 buckets an address maps to; the unit of the htrace bitmap. |
| **Sandbox** | The memory a test case may touch: two input pages (`main` + `faulty`), with a zeroed overflow guard page at the end. |
| **Mistraining** | Purposefully steering the branch predictor before the measured run to provoke misprediction. |
| **PAC / MTE** | Arm Pointer Authentication / Memory Tagging. |

---

# Part II — Architecture & components

## 2.1 Component catalog & interfaces

The fuzzer is the only long-lived coordinator. It drives three external pieces over three
different mechanisms: it **forks** a helper to assemble instructions, talks to the **CE** over a
**pipe**, and talks to the **kernel module** over **ioctl + sysfs**.

### Kernel module — execution state machine

The module behaves as a small automaton. Its state records whether a **test case** and/or
**inputs** are loaded and whether a **measurement** has been taken, and it gates which
operations are legal (`TRACE` needs `READY`/`TRACED`; reading a `MEASUREMENT` needs `TRACED`).
The `Valid in` column of the ioctl table below refers to these states.

| State | Test | Inputs | Measured | Meaning |
|---|---|---|---|---|
| `CONFIGURATION` | no | no | — | Initial; nothing loaded. |
| `LOADED_TEST` | yes | no | — | Test case present, no inputs. |
| `LOADED_INPUTS` | no | yes | — | Inputs present, no test case. |
| `READY` | yes | yes | no | Both loaded; a run is possible. |
| `TRACED` | yes | yes | yes | Run done; per-input measurements readable. |

```
   ┌───────────────┐                         ┌───────────────┐
   │ CONFIGURATION │ ───── write test ─────► │  LOADED_TEST  │
   │   (T0, I0)    │ ◄──── unload test ───── │   (T1, I0)    │
   └───────────────┘                         └───────────────┘
        │   ▲                                     │   ▲
  write │   │ clear                         write │   │ clear
  input │   │ inputs                        input │   │ inputs
        ▼   │                                     ▼   │
   ┌───────────────┐                         ┌───────────────┐     TRACE     ┌──────────────┐
   │ LOADED_INPUTS │ ───── write test ─────► │     READY     │ ────────────► │    TRACED    │  ⟲ re-run
   │   (T0, I1)    │ ◄──── unload test ───── │   (T1, I1)    │               │  (measured)  │
   └───────────────┘                         └───────────────┘               └──────────────┘
```

`T` = test loaded, `I` = inputs loaded (`1` = yes, `0` = no). From `TRACED`, *unload test* /
*clear inputs* behave as from `READY` (see the transition table below).

*Under the hood — transitions:*

| Event | From → To |
|---|---|
| write test | `CONFIGURATION`→`LOADED_TEST`; `LOADED_INPUTS`→`READY` |
| write input | `CONFIGURATION`→`LOADED_INPUTS`; `LOADED_TEST`→`READY` |
| `UNLOAD_TEST` | `LOADED_TEST`→`CONFIGURATION`; `READY`/`TRACED`→`LOADED_INPUTS` |
| `CLEAR_ALL_INPUTS` | `LOADED_INPUTS`→`CONFIGURATION`; `READY`/`TRACED`→`LOADED_TEST` |
| `TRACE` | `READY`→`TRACED`; `TRACED`→`TRACED` (re-run) |
| `MEASUREMENT` (read) | `TRACED` only; no transition |

### Kernel module — `/dev/executor` + `/sys/executor/`

A char device plus a sysfs directory. **Control** goes through `ioctl`; **bulk data** (test
case bytes, input bytes, measurement results) moves with `read`/`write` *after* selecting a
region with a checkout ioctl. Runtime knobs are plain sysfs files.

*Under the hood — ioctl ABI* (magic `'r'`; mirrored in `aarch64_kernel.py`). *Valid in* lists
the states (above) in which the command does its job:

| # | Command | Input | Output | Valid in | Purpose |
|---|---|---|---|---|---|
| 1 | `CHECKOUT_TEST` | — | — | any | Select the test-case region for `read`/`write`. |
| 2 | `UNLOAD_TEST` | — | — | any | Drop the loaded test case. |
| 3 | `GET_NUMBER_OF_INPUTS` | — | `uint64` count | any | Number of allocated inputs. |
| 4 | `CHECKOUT_INPUT` | `uint64` input id | — | any (id must exist) | Select that input for `read`/`write`. |
| 5 | `ALLOCATE_INPUT` | — | `uint64` new id | any | Allocate an input slot. |
| 6 | `FREE_INPUT` | `uint64` input id | — | any | Free that input slot. |
| 7 | `MEASUREMENT` | — | `measurement_t` = `htrace` (u64) + `pfc[3]` (u64 each) | `TRACED`, input checked out | Result for the checked-out input. |
| 8 | `TRACE` | — | — | `READY` / `TRACED` | Run the test case over all inputs and measure. |
| 9 | `CLEAR_ALL_INPUTS` | — | — | any | Free every input. |
| 10 | `GET_TEST_LENGTH` | — | `uint64` length | any | Length of the loaded test case. |
| 11 | `SET_PAC_KEYS` | `pac_keys` (5×128-bit: IA/IB/DA/DB/GA), or `NULL` to clear | — | any | Set the keys the executor signs/auths with; `NULL` reverts to the live hardware keys. |
| 12 | `GET_PAC_KEYS` | — | `pac_keys` | any | Read the keys the executor will use. |
| 13 | `PAC_SIGN` | `pac_sign_req` = `ptr`, `ctx`, `mnemonic` | `result` = signed pointer | any | Sign a pointer (`PAC*`). |
| 14 | `PAC_AUTH` | `pac_sign_req` = `ptr`, `ctx`, `mnemonic` | `result` = authenticated pointer | any | Authenticate a signed pointer (`AUT*`). |
| 15 | `PAC_XPAC` | `pac_sign_req` = `ptr` | `result` = stripped pointer | any | Strip the PAC field (`XPAC`). |
| 16 | `MTE_TAG_REGION` | `mte_tag_region_req` = `sandbox_offset`, `length`, `tag` | — | any | Tag a sandbox region. |

Called outside its *Valid in* states a command fails without changing state: `TRACE` returns
`-EINVAL`; `MEASUREMENT` is rejected (error logged, no data) if the state is not `TRACED` or if
the test region — rather than an input — is checked out; `CHECKOUT_INPUT` with an unknown id
leaves the current selection unchanged.

*Under the hood — sysfs files:*

| File | Mode | Accepted value | Meaning |
|---|---|---|---|
| `measurement_mode` | rw | exactly `P+P` or `F+R` | Prime+Probe vs Flush+Reload. |
| `warmups` | rw | unsigned integer | Micro-architectural warm-up rounds before the measured run. |
| `enable_pre_run_flush` | rw | `0` = off, non-zero = on | Flush the branch-predictor history (PHR) before each run, so the previous run doesn't bias the next. |
| `pin_to_core` | rw | base-10 int; an online CPU id (negative/invalid → pinning cleared) | Pin execution to a CPU; invalid input falls back to the current CPU. |
| `enable_branch_training` | rw | `0` = off, non-zero = on | Apply mistraining before the measured run. |
| `branch_training_config` | rw | `offset:taken,offset:taken,…` — e.g. `12:1,40:0` (`offset` = **byte** offset from the start of the test case) | Mistraining sequence: for the branch at that offset, train taken (`1`) or not-taken (`0`). Writing it also enables training. |
| `print_sandbox_base` | r | (read-only) hexadecimal pointer | Base address of the sandbox `main_region` (input page 0). |
| `print_code_base` | r | (read-only) hexadecimal pointer | The address where the user's test-case code will be loaded, for the selected `measurement_mode` (P+P and F+R place it at different fixed offsets). Known from module load; does not require a test case. |

### Contract executor (CE) — pipe over `stdin`/`stdout`

The fuzzer launches the CE with `subprocess.Popen` and exchanges length-prefixed messages
over its `stdin`/`stdout`. Every message is an **8-byte little-endian header** `{length, type}`
followed by `length` payload bytes; payloads begin with the magic `RVZR`.

*Under the hood — message & request format:*

- **Types:** `REQUEST = 1`, `RESPONSE = 2`.
- A request carries a `configuration` (flags, `max_misspred_branch_nesting`, `contract_type`,
  optional requested code/mem bases) plus optional **code**, **registers**, and **memory**
  sections selected by `sim_flags` (`HAS_CODE` / `HAS_REGS` / `HAS_MEMORY`).
- **`contract_type`:** `0` always-mispredict, `1` arch-only, `2` BPU Neoverse-N3 (TAGE model).
- The response carries the per-instruction trace (registers, PC, memory accesses, speculation
  nesting) the fuzzer turns into the ctrace.
- *(maintainer note)* `max_misspred_instructions` and the physical base requests exist in the
  struct but are **not supported**.

The CE also **embeds CPython** (`Py_Initialize`): the Neoverse-N3 contract calls a Python TAGE
branch-predictor model from inside the C process (§3.1).

### `asm_to_bytes` — fork/exec helper

When the generator needs raw encodings for assembly text (in-memory assembly), it forks the
small `asm_to_bytes` binary and pipes text in / encodings out, avoiding a full assemble-to-ELF
round trip. A **fresh process is spawned per call** (`Popen` + `communicate`, one-shot) — unlike
the **CE**, which is launched once and kept alive to serve many requests.

## 2.2 Communication map

```
   +=======================================================================+
   |                       Fuzzer   (Python, user space)                   |
   |                                                                       |
   |              generate  ->  model  ->  measure  ->  compare            |
   +=======================================================================+
          |                            |                            |
          |  fork / exec               |  pipe (stdin/stdout):      |  ioctl + read/write
          |  asm text -> bytes         |  8-byte {len,type} header  |  + sysfs config files
          |                            |  then the message payload  |
          v                            v                            v
   +-----------------+      +---------------------------+      +---------------------------+
   |  asm_to_bytes   |      |  Contract Executor (CE)   |      |  Kernel module            |
   |  (C helper)     |      |  (C process)              |      |  /dev/executor            |
   |                 |      |                           |      |  + /sys/executor/*        |
   |  assembles      |      |  models the program,      |      |  loads TC + inputs,       |
   |  instructions   |      |  emits the trace          |      |  runs on the real CPU,    |
   +-----------------+      +---------------------------+      |  measures cache -> htrace |
                                        |                      +---------------------------+
                                        |  Py_Initialize                     |
                                        v                                    v
                            +-----------------------+                  +-----------+
                            |  CPython: BPU sim     |                  |    CPU    |
                            +-----------------------+                  +-----------+
                                                                   (device under test)
```

The fuzzer never talks to the CPU directly: the **CE** gives the *expected* (contract) trace,
the **kernel module** gives the *actual* (hardware) trace, and the fuzzer compares them.
## 2.3 Life of a fuzzing round

One round = **one test case measured against many inputs**.

1. **Generate** a random test case (or load one from asm/template).
2. **Seed inputs** — initial register + memory values.
3. **Model (CE).** Trace each input to get its ctrace and its **taint** — the set of input bytes
   that actually affect the ctrace.
4. **Boost.** Manufacture many extra inputs by mutating only the *untainted* bytes; by
   construction they all share one ctrace (this is what makes leak-hunting cheap).
5. **Measure (kernel module).** Run the test case over every input on the CPU and collect
   htraces (with warm-ups and, if enabled, mistraining).
6. **Analyse.** Group inputs by ctrace; within a group, **differing htraces ⇒ a candidate
   violation**.
7. **Filter (slow path).** Re-trace at higher speculation nesting and re-measure with a
   *priming* check to drop false positives.
8. **Report.** Save the confirmed violation — test case, inputs, and htraces — for triage (§4.4).

*Under the hood — fast vs slow path.* The fast path uses minimal nesting and few repetitions to
find candidates quickly; only candidates pay for the expensive slow path (max nesting +
priming) that confirms a real violation.

## 2.4 Diagrams

**Classic round — one program, many inputs.** The same test case is run against many boosted
inputs; the model and the hardware each produce a trace, and inputs are grouped by ctrace.

```
                       ┌───────────────────────┐
                       │   test case (program) │
                       └───────────┬───────────┘
                                   │ boost → inputs i0…iN
                   ┌───────────────┴───────────────┐
            model  │                               │  hardware
             (CE)  ▼                               ▼  (kernel)
            ┌───────────────┐               ┌───────────────┐
            │ ctrace(i0…iN) │               │ htrace(i0…iN) │
            └───────┬───────┘               └───────┬───────┘
                    └───────────────┬───────────────┘
                                    ▼
                         group inputs by ctrace
                                    │
                                    ▼
            ┌───────────────────────────────────────────┐
            │  same ctrace + different htrace ⇒ VIOLATION │
            └───────────────────────────────────────────┘
```

**PAC / non-interference setup — one input, many sibling programs.** One input is run against
many variants of the program (e.g. different PAC signing slots); their htraces are compared.

```
                          ┌───────────┐
                          │ one input │
                          └─────┬─────┘
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        ┌───────────┐     ┌───────────┐     ┌───────────┐
        │ program P0│     │ program P1│ ··· │ program PN│
        └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
              │   (same code, different PAC slot / variant)
              ▼                 ▼                 ▼
        ┌───────────┐     ┌───────────┐     ┌───────────┐
        │ htrace P0 │     │ htrace P1 │     │ htrace PN │
        └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
              └─────────────────┼─────────────────┘
                                ▼
            ┌────────────────────────────────────────┐
            │ htraces differ across variants          │
            │              ⇒ PAC-related leak         │
            └────────────────────────────────────────┘
```

**CE single-step loop.** The CE rewrites every test-case instruction into a `BL hook`
trampoline, then walks them one at a time.

```
   ┌──────────────────────────────────────────────────┐
   │  load TC: rewrite each instruction → "BL hook"     │
   └─────────────────────────┬────────────────────────┘
                             ▼
              ┌────────────────────────────┐ ◄────────────────┐
              │      next instruction i     │                  │
              └─────────────┬──────────────┘                  │
                            ▼                                  │
   ┌──────────────────────────────────────────────────┐       │
   │  BL hook → base_hook_c:                            │  repeat for
   │    • emulate effect (regs / mem / flags, PAC/MTE)  │  every instruction
   │    • record trace entry (regs, PC, EA, nesting)    │───────┘
   │    • at a branch: optionally mispredict            │
   │        (checkpoint → wrong path → roll back)       │
   └─────────────────────────┬────────────────────────┘
                             ▼  (program end)
   ┌──────────────────────────────────────────────────┐
   │       per-instruction trace  →  ctrace             │
   └──────────────────────────────────────────────────┘
```

**Sandbox memory map.** A test case may only touch this fixed arena. During the run `x29`
holds the **main-region** base; every memory access is masked to `x29 + (reg & 0x1fff)`, so it
lands somewhere in `main`+`faulty` (8 KB). Cache set = `(offset // 64) % 64`.

```
   low address
   ╔═══════════════════════════════╗
   ║  eviction_region     (32 KB)  ║   Prime+Probe fills this
   ╠═══════════════════════════════╣
   ║  lower_overflow      ( 4 KB)  ║   guard (zeroed)
   ╠═══════════════════════════════╣
   ║  main_region         ( 4 KB)  ║ ◄── x29   input page 0   (offset 0x000–0x0FFF)
   ╠═══════════════════════════════╣
   ║  faulty_region       ( 4 KB)  ║           input page 1   (offset 0x1000–0x1FFF)
   ╠═══════════════════════════════╣
   ║  upper_overflow      ( 4 KB)  ║   guard (zeroed)
   ╠═══════════════════════════════╣
   ║  stored_rsp          (  8 B)  ║   saved stack pointer
   ╠═══════════════════════════════╣
   ║  latest_measurement           ║   htrace + performance counters
   ╚═══════════════════════════════╝
   high address
```

# Part III — Subsystems in depth

## 3.1 Contract executor (CE)
## 3.2 Mistraining
## 3.3 Instruction tagging
## 3.4 Data-structure reference

# Part IV — Operating it

## 4.1 Installing
## 4.2 Running: download_spec, fuzz, tfuzz
## 4.3 minimize / reproduce
## 4.4 Helper & triage scripts
## 4.5 Troubleshooting & gotchas
## 4.6 Experimental / WIP

# Part V — Reference

## 5.1 Feature parity with x86
## 5.2 Deliberate limitations
## 5.3 Target machine
## 5.4 Shared `src/` code
## 5.5 Provenance & validation
