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
   +---------------+                         +---------------+
   | CONFIGURATION | ----- write test -----> |  LOADED_TEST  |
   |   (T0, I0)    | <---- unload test ----- |   (T1, I0)    |
   +---------------+                         +---------------+
        |   ^                                     |   ^
  write |   | clear                         write |   | clear
  input |   | inputs                        input |   | inputs
        v   |                                     v   |
   +---------------+                         +---------------+     TRACE     +--------------+
   | LOADED_INPUTS | ----- write test -----> |     READY     | ------------> |    TRACED    |  re-run
   |   (T0, I1)    | <---- unload test ----- |   (T1, I1)    |               |  (measured)  |
   +---------------+                         +---------------+               +--------------+
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
- A request carries a `configuration` (flags, `max_misspred_branch_nesting`, `execution_clauses`,
  `branch_predictor`, optional requested code/mem bases) plus optional **code**, **registers**,
  and **memory** sections selected by `sim_flags` (`HAS_CODE` / `HAS_REGS` / `HAS_MEMORY`).
- **`execution_clauses`:** a **bitmask** of independently-composable clauses —
  `EXEC_CLAUSE_COND = 1` (conditional-branch misprediction), `EXEC_CLAUSE_BPAS = 2`
  (speculative store bypass), `EXEC_CLAUSE_BPU = 4` (branch-predictor model). `0` = seq/arch-only.
  Only an explicit allowlist is accepted (e.g. `COND`, `BPAS`, `COND|BPAS`); the CE traps on any
  other combination (see §3.1).
- **`branch_predictor`:** an enum selecting the model used when `EXEC_CLAUSE_BPU` is set —
  `BRANCH_PREDICTOR_NONE = 0`, `BRANCH_PREDICTOR_NEOVERSE_N3 = 1` (TAGE). Chosen from the input,
  not hardcoded; BPU with `NONE` traps.
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
                       +-----------------------+
                       |   test case (program) |
                       +-----------+-----------+
                                   | boost > inputs i0~iN
                   +---------------+---------------+
            model  |                               |  hardware
             (CE)  v                               v  (kernel)
            +---------------+               +---------------+
            | ctrace(i0~iN) |               | htrace(i0~iN) |
            +-------+-------+               +-------+-------+
                    +---------------+---------------+
                                    v
                         group inputs by ctrace
                                    |
                                    v
            +---------------------------------------------+
            |  same ctrace + different htrace > VIOLATION |
            +---------------------------------------------+
```

**PAC / non-interference setup — one input, many sibling programs.** One input is run against
many variants of the program (e.g. different PAC signing slots); their htraces are compared.

```
                          +-----------+
                          | one input |
                          +-----+-----+
              +-----------------+-----------------+
              v                 v                 v
        +-----------+     +-----------+     +-----------+
        | program P0|     | program P1| ... | program PN|
        +-----+-----+     +-----+-----+     +-----+-----+
              |   (same code, different PAC slot / variant)
              v                 v                 v
        +-----------+     +-----------+     +-----------+
        | htrace P0 |     | htrace P1 |     | htrace PN |
        +-----+-----+     +-----+-----+     +-----+-----+
              +-----------------+-----------------+
                                v
            +----------------------------------------+
            | htraces differ across variants          |
            |              > PAC-related leak         |
            +----------------------------------------+
```

**CE single-step loop.** The CE rewrites every test-case instruction into a `BL hook`
trampoline, then walks them one at a time. At each *speculation point* — a conditional branch
(mispredict) or, under the `bpas` clause, a store (bypass) — it checkpoints, explores the
speculative path, and rolls back; the enabled execution clauses decide what counts as one (§3.1).

```
   +----------------------------------------------------+
   |  load TC: rewrite each instruction > "BL hook"     |
   +--------------------------+-------------------------+
                              v
              +------------------------------+<---------------+
              |     next instruction i       |                |
              +---------------+--------------+                |
                              v                               |
   +----------------------------------------------------+     |
   |  BL hook > base_hook_c:                            |     |  repeat for
   |    * emulate effect (regs / mem / flags, PAC/MTE)  |     |  every instruction
   |    * record trace entry (regs, PC, EA, nesting)    |-----+
   |    * at a branch: optionally mispredict            |
   |        (checkpoint > wrong path > roll back)       |
   +--------------------------+-------------------------+
                              v  (program end)
   +----------------------------------------------------+
   |    per-instruction trace  >  ctrace                |
   +----------------------------------------------------+
```

**Sandbox memory map.** A test case may only touch this fixed arena. During the run `x29`
holds the **main-region** base; every memory access is masked to `x29 + (reg & 0x1fff)`, so it
lands somewhere in `main`+`faulty` (8 KB). Cache set = `(offset // 64) % 64`.

```
   low address
   +-------------------------------+
   |  eviction_region     (32 KB)  |   Prime+Probe fills this
   +-------------------------------+
   |  lower_overflow      ( 4 KB)  |   guard (zeroed)
   +-------------------------------+
   |  main_region         ( 4 KB)  | <-- x29   input page 0   (offset 0x000-0x0FFF)
   +-------------------------------+
   |  faulty_region       ( 4 KB)  |           input page 1   (offset 0x1000-0x1FFF)
   +-------------------------------+
   |  upper_overflow      ( 4 KB)  |   guard (zeroed)
   +-------------------------------+
   |  stored_rsp          (  8 B)  |   saved stack pointer
   +-------------------------------+
   |  latest_measurement           |   htrace + performance counters
   +-------------------------------+
   high address
```

# Part III — Subsystems in depth

## 3.1 Contract executor (CE)

The CE produces a ctrace by *modelling* a test case's execution, including the speculation a
contract permits. Its speculation engine is built so contracts are **composable**: a run enables
a set of independent *execution clauses* (the `execution_clauses` bitmask, §2.1), and the engine
runs them together over one instruction stream.

**Three layers, mutually ignorant** (`src/aarch64/contract_executor/`):

| Layer | Files | Responsibility |
|---|---|---|
| **Engine** | `simulation_execution_clause_hook.{c,h}` | Mechanism only: speculation nesting, the `max_misspred_branch_nesting` cap, a LIFO checkpoint stack, `spec_push_frame()` (stamps a frame with the clause that owns it), and `handle_window_end()` (routes a window's end to the owning clause's `on_rollback`, default = reload memory + registers). Knows nothing about branches, stores, or TAGE. |
| **Execution clauses** | `execution_clauses.{c,h}` (registry + `execution_clauses_supported`), `execution_clause_{cond,bpu,bpas}.c`, `branch_speculation.{c,h}` | Each clause is a `struct execution_clause_descriptor { name; clause_bit; on_init; on_reset; on_instruction → redirect\|NULL; on_rollback; }`. Enabled per-run by its bit. |
| **Branch predictor** (DI) | `branch_predictors.{c,h}`, `neoverse_n3_bpu.c` | `branch_predictor_by_id()` maps the input's `branch_predictor` enum to a `{init,reset,predict,update}` vtable. The BPU clause resolves it in `on_reset`; the engine references no predictor. |

**Per-instruction dispatch.** For each instruction the engine iterates the enabled clauses in
registry order, calling `on_instruction(state)`; each may return a redirect PC or `NULL`. The
rule: multiple clauses *may* return a redirect, but every non-`NULL` redirect must be the **same**
address, else the engine hard-traps (it never assumes a clause returns `NULL`).

**Each fork is independent → 2ᴺ flows.** A store (bypass/apply) and a branch (mispredict/correct)
are both 2-way speculation forks. Crucially, when a window ends the engine reverts the checkpoint
and then **re-dispatches the clauses on the instruction it resumes into** — so a store or branch on
an *architectural continuation* re-forks just like one on a speculative path. Thus N forks before a
load produce up to 2ᴺ flows reaching it, and `max_misspred_branch_nesting` bounds how many forks may
be open at once (so a flow that bypasses *k* stores needs depth *k*). Stores and branches are
symmetric: the same program shape with N stores or N branches yields the identical fork tree. (If a
resumed continuation is itself past the code end — e.g. a branch target beyond the program — the
engine keeps unwinding the stack rather than ending the run with frames still pending.)

**Why registry order matters.** The checkpoint stack is strictly **LIFO** — `handle_window_end`
always pops the top frame and returns to *its* `return_addr`. When a store is followed by a
branch, **both** clauses act on that branch instruction in one dispatch: `bpas` (phase B) pushes
the **outer** store-bypass frame and installs the stale memory, then `cond`/`bpu` pushes the
**inner** branch-mispredict frame and redirects. The branch window resolves first (the wrong path
re-converges to the branch's architectural continuation), so by LIFO it must sit on top — i.e. be
pushed **last**. Both frames are pushed in the same dispatch, in registry order, so `bpas` must be
registered **before** `cond`/`bpu`. Reversed, the branch's wrong path would pop the bypass frame
and "return" to the store's PC with the bypass snapshot — the wrong address and the wrong memory
(this is what the `checkpoint_id == current_checkpoint_id - 1` LIFO assert guards).

**The three clauses:**

- **`cond`** (always-mispredict): at a conditional branch, checkpoints the architectural
  continuation and redirects to the mispredicted path; the engine reloads at the window's end.
- **`bpu`**: like `cond`, but the branch's outcome is driven by the injected predictor model
  (Neoverse-N3 TAGE), so only branches the model mispredicts fork.
- **`bpas`** (speculative store bypass / Spectre-v4): **two-phase**. *Phase A* at a store —
  snapshot the whole memory image, then let the model execute the store (register writeback
  included). *Phase B* on the next instruction — push a frame checkpointing the **post-store**
  state (so the store is never re-executed and rollback stays uniform), then restore the
  pre-store snapshot so the speculative window reads the **stale** value. A whole-memory
  snapshot makes `STR`/`STP`/writeback/atomics all uniform. Phase B is skipped if a rollback
  changed the nesting depth since phase A (its snapshot would belong to a discarded context).

**Supported combinations** are validated in **both** Python (`SUPPORTED_EXECUTION_CLAUSES`,
`aarch64_contract_executor.py`) and C (`execution_clauses_supported` → trap): `seq` (empty),
`cond`, `bpas`, `bpu`, and `cond|bpas`. Two branch models together (`cond|bpu`) are rejected.

**Invariants (do not regress):** predictor init is **lazy** in `bpu_on_reset` (eager init breaks
non-BPU runs); checkpoint slots are **reused** on rollback (LIFO free, else the 1024-slot cap
traps in a loop with many windows); `is_conditional_branch` is defined by **inclusion**
(`B.cond`/`CBZ`/`CBNZ`/`TBZ`/`TBNZ`); the `max_misspred_branch_nesting` cap gates **every** clause
(`spec_nesting() < spec_max_nesting()`), so a cap of 0 disables speculation entirely. These are
covered by `test_ce_integration.c` (`bpas_*`, `cond_*`, `unsupported_clauses_rejected`,
`bpu_*`).

## 3.2 Mistraining
## 3.3 Instruction tagging
## 3.4 Data-structure reference

## 3.5 Device lock & fault safety

All `/dev/executor` access (ioctl / read / write) is serialized by one mutex (`executor_lock`
in `chardevice.c`) — the inputs rbtree, the state machine and the test-case/measurement buffers
have no other synchronization.

**The hazard:** a fault *inside* an ioctl while the lock is held — e.g. a failed PAC `AUT*` at
EL1 on FEAT_FPAC hardware Oopses the kernel — kills the faulting task without releasing the
lock. With a plain blocking `mutex_lock`, every later caller then blocks forever in
uninterruptible **D** state: an unkillable deadlock that only a reboot clears (the dead holder
also pins a module reference, so `rmmod` won't work either).

So the device lock is **self-healing** — it can't recover the executor state a faulting op left
half-mutated (that genuinely needs a reboot), but it refuses to become a silent, unkillable hang:

- **killable waits** (`mutex_lock_killable`): a blocked caller can always be killed, never the
  D-state pile-up;
- **wedged fail-fast**: if a new caller finds the lock held longer than any legitimate operation
  (`EXECUTOR_LOCK_DEADLOCK_MS`), it latches a `wedged` flag, logs a **CRITICAL** message naming
  the holder pid, and returns `-EIO` instead of blocking — telling you to check `dmesg` for the
  Oops and reload the module or reboot.

It never force-unlocks or re-initialises the mutex (that would corrupt the wait-list and the
shared state). `handle_pac_auth` runs the real `AUT*` (faithful to hardware), so a bad pointer
still faults; the self-healing lock plus the test runner's dmesg guard turn that into a reported,
contained failure rather than a deadlock. Tests verify auth equivalence *without* faulting via
`XPAC` (strip) + `PAC` (re-sign) + compare — see `unit_pacga`.

# Part IV — Operating it

## 4.1 Installing
## 4.2 Running: download_spec, fuzz, tfuzz

### Downloading the instruction set (`base.json`)

The fuzzer drives generation from an instruction-set spec, `base.json`. It is **not**
checked in — it is generated from ARM's machine-readable A64 ISA. Build it once:

```
python revizor.py download_spec -a aarch64 -o base.json
```

This downloads the ARM A64 ISA XML release (`A64-2025-09`) from `developer.arm.com`
(the tarball is cached locally, so re-runs are fast) and parses it into `base.json`.
All `fuzz`/`tfuzz` invocations below consume it via `-s base.json`. Pass
`--extensions <category> ...` to keep only specific instruction-class categories
(default: the full set).

### Detecting Spectre-v1 (ready-to-run configs)

Two configs in `configs/` detect a Spectre-v1 (conditional-branch-bypass) leak out of the box —
one per measurement channel. Both set the **arch-only** contract (`contract_execution_clause:
[seq]`) and turn the **BPU flush off** (`enable_pre_run_flush: 0`) so the guarding branch
mispredicts naturally; a speculative load then touches a cache set the contract forbids, which
Revizor reports as a violation.

```
# Prime+Probe (preferred - more sensitive):
python revizor.py fuzz -s base.json -c configs/spectre_v1_pp.yml -n 200 -i 50 --save-violations true -w out_pp

# Flush+Reload:
python revizor.py fuzz -s base.json -c configs/spectre_v1_fr.yml -n 200 -i 50 --save-violations true -w out_fr
```

A violation drops a `violation-*/` artifact in the working dir (the test case, the
counterexample inputs as `input_NNNN_nzcv_scheme.bin`, traces, and a `reproduce.yaml`). The leak
is **intermittent** (the branch mispredicts only part of the time), so the fuzzer relies on its
large-sample slow path — expect a hit within a few dozen test cases. **Prime+Probe is markedly
more sensitive** than Flush+Reload to the store-based gadgets that turn up here (≈8× more
positive runs for the same gadget), so use `spectre_v1_pp.yml` for a quick confirmation and give
F+R more test cases. Requires the kernel module loaded (`--test aarch64-ko` or a manual insmod).

## 4.3 minimize / reproduce
## 4.4 Helper & triage scripts

**Debugging utilities** (`src/aarch64/debugging/`):

- `make_input.py` — build a raw `input_t` binary from an optional JSON spec (per-register
  values and per-byte memory overrides; unspecified bytes are randomized). The layout is
  main 4K + faulty 4K + register region 4K; flags/sp are written verbatim. Example:
  `make_input.py --input input_pattern.json --output input.bin [--print]`.
- `to_executor_input.py` — convert a saved input (NZCV flags in the per-flag NZCVScheme
  encoding, i.e. `input_NNNN_nzcv_scheme.bin`) into the form `/dev/executor` accepts (flags
  slot reconstructed to PSTATE). Needed for **manual** reproduction via `executor_userland`,
  which bypasses the Python executor that normally does this conversion on write:
  `to_executor_input.py saved.bin executor_ready.bin`.

**`executor_userland`** (`src/executor_userland/`) is the minimal C tool to drive
`/dev/executor` by hand: numbered ioctls plus `w file` / `r file` to write/read the current
checkout.

### Triaging and reproducing a violation (skills)

A `fuzz` run reports a *contract violation* — two inputs that are architecturally equivalent
yet produce different hardware traces. Two [Claude Code](https://claude.com/claude-code)
**skills** under `.claude/skills/` help you make sense of one; ask Claude Code to "triage" or
"reproduce" a `violation-*/` directory, or run the scripts directly (activate your venv +
`sudo chmod 777 /dev/executor`).

- **`revizor-violation-triage`** — genuine leak vs noise **without re-running on hardware**.
  Runs the CE under `ALWAYS_MISPREDICT` and checks: the two inputs' architectural cache lines
  are identical, their speculative lines differ, and that speculative divergence matches the HW
  htrace divergence *already in the report*. Verdict **GENUINE** / **NOISE** / **INVESTIGATE**.
  `python3 .claude/skills/revizor-violation-triage/scripts/triage_violation.py <violation-dir>`.
- **`reproduce-revizor-violation`** — confirms it on hardware by recreating the exact µarch
  state: load context inputs `0..min(A,B)`, **swap only the violating input**, measure, and see
  which cache set encodes which input ran. Intermittent, so collect statistics over many trials.

> ⚠️ **Caveats.** These are *quick verification heuristics, not proofs* — a GENUINE verdict plus a
> clean reproduction is strong corroboration, not a guarantee; NOISE/INVESTIGATE warrants a manual
> look. They have **not been exercised by other users yet**, so expect to refine them by hand for
> your setup (rough edges will be fixed over time). And never "confirm noise" by re-measuring the
> same test many times — repetition trains the branch predictor and *hides* a genuine leak.

## 4.5 Testing

The installer doubles as the test runner — `src/aarch64/install_revizor_env.sh --test <group>`:

| group | what it runs |
|-------|--------------|
| `aarch64-ce`     | contract-executor C tests (`test_ce` unit + `test_ce_integration`) |
| `aarch64-ko`     | builds + (re)loads the kernel module, then the `/dev/executor` Python tests |
| `aarch64-python` | all Python unit tests (common + `tests/aarch64_tests/`) |
| `aarch64-all`    | all of the above |

Each device-touching group is wrapped in a **dmesg guard**: it scans the kernel log for new
`Oops` / `Internal error` / `FPAC` / `Call trace` lines around the run and fails loudly if the
module faulted, so a kernel exception is reported rather than silently swallowed.

Tests also run directly from the repo root: `python3 -m unittest discover -s tests -p 'unit_*.py'`
(common) and `python3 -m unittest discover -s tests/aarch64_tests -p 'unit_*.py' -t .` (AArch64);
`tests/runtests.sh` runs the whole thing (mypy, flake8, common + AArch64 + x86).

**Hardware caveat:** the `/dev/executor` tests need the module loaded
(`sudo insmod revizor-executor.ko && sudo chmod 777 /dev/executor`). A fault under the device
lock (e.g. a failed PAC AUTH at EL1 on FEAT_FPAC) latches the device "wedged" and fails fast;
recover with a module reload or a reboot.

To read this manual as a styled page, build it to a standalone HTML file with
`src/aarch64/install_revizor_env.sh --doc` (prints the path to open in a browser).

## 4.6 Troubleshooting & gotchas
## 4.7 Experimental / WIP

# Part V — Reference

## 5.1 Feature parity with x86
## 5.2 Deliberate limitations
## 5.3 Target machine
## 5.4 Shared `src/` code
## 5.5 Provenance & validation
