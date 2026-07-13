# Revizor on AArch64 — Onboarding Manual

## About this document

An introduction to the AArch64 port of Revizor: what it does, how its pieces fit together, the
control surfaces you drive it through, and the file formats you work with. It aims to be thorough
without cataloguing every internal micro-mechanic. New terms are **bold** on first use and
collected in the [Glossary](#glossary).

Generated from `docs/aarch64/index.md`. Edit that file, then rebuild from `docs/aarch64/` with
`make` (or `python3 build_docs.py`).

## 1. What Revizor does

Revizor is a *microarchitectural fuzzer*: it searches for cases where a CPU leaks more through its
microarchitecture than a **[contract](#glossary)** permits. A contract states the only ways a
program's data may influence attacker-observable state — here, the data-cache footprint.

For each generated program, two traces are compared:

- the **[contract trace](#glossary)** (**ctrace**) — what the contract permits to be observable,
  computed by a software model;
- the **[hardware trace](#glossary)** (**htrace**) — what the real CPU actually leaked, measured as
  a 64-bit cache-set bitmap.

A **[violation](#glossary)** is a pair of inputs the contract treats as identical (**equal
ctrace**) but the hardware tells apart (**different htrace**): the contract promised the two inputs
would look the same to an attacker, yet the CPU produced two different cache footprints — so it
leaked something the contract forbade (for example, a Spectre-style speculative access). The
contract framework originates in Guarnieri et al., *Hardware-Software Contracts for Secure
Speculation* (IEEE S&P 2021).

> **Intuition**
> - **ctrace** — what an attacker is *allowed* to observe.
> - **htrace** — what an attacker can *actually* observe.
> - **Violation** — the hardware revealed something the contract forbade.

Leaks are about *distinguishing* secrets, so Revizor runs **many inputs** through one program and
looks for a pair that is contract-equal yet hardware-distinct. **[Taint](#glossary)** marks the
input bytes a traced execution actually reads; **[boosting](#glossary)** then cheaply mass-produces
contract-equal inputs by varying only the *other* bytes.

## 2. Components

The **fuzzer** (Python, `src/`) is the only long-lived coordinator. It drives three external
pieces, each over its own mechanism.

| Component | Language / location | Role |
|---|---|---|
| Fuzzer | Python (`src/`) | Orchestrates generate → model → measure → compare. |
| Generator | Python (`src/aarch64/`) | Emits random AArch64 test cases from the ISA spec. |
| Contract executor (**CE**) | C process (`src/aarch64/contract_executor/`) | Models a program to produce the ctrace, including speculation. |
| Kernel module | C (`src/aarch64/executor/`) | Runs the test case on the real CPU in a sandbox and measures the cache. |
| `asm_to_bytes` | C (`src/aarch64/asm_to_bytes/`) | Assembles instruction text to encodings, in memory. |

### 2.1 Communication map

```
   +=======================================================================+
   |                       Fuzzer   (Python, user space)                   |
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

The fuzzer never talks to the CPU directly: the **CE** gives the *expected* (contract) trace, the
**kernel module** gives the *actual* (hardware) trace, and the fuzzer compares them. The CE is
launched once and kept alive to serve many requests. `asm_to_bytes` is a one-shot helper forked per
call, so it is fork-heavy on the sealed/NI path (one variant per input); two optimizations avoid
those forks: `Aarch64Generator.in_memory_assemble` **memoizes** by source text (identical assembly is
served from a cache, not re-forked), and sealed variants are assembled by **relocation** rather than
re-running the assembler (see §6.2).

## 3. A fuzzing round

One round = **one test case measured against many inputs**.

1. **Generate** a random test case (or load one from asm / a template).
2. **Seed inputs** — initial register + memory values.
3. **Model (CE).** Trace each input to get its ctrace and its **taint** — the input bytes the
   ctrace depends on.
4. **Boost.** Manufacture many extra inputs by mutating only the *untainted* bytes; by construction
   they share one ctrace, which is what makes leak-hunting cheap.
5. **Measure (kernel module).** Run the test case over every input on the CPU and collect htraces.
6. **Analyse.** Group inputs by ctrace; within a group, **differing htraces ⇒ a candidate
   violation**.
7. **Filter.** Re-trace at higher speculation nesting and re-measure to drop false positives.
8. **Report.** Save the confirmed violation — test case, inputs, and traces — for triage (§7).

A **fast path** (minimal speculation nesting, few repetitions) finds candidates quickly; only
candidates pay for the slower confirmation path (max nesting + re-measurement).

**Classic round — one program, many inputs.**

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

## 4. Driving the kernel module

The module is a small **state machine**. Its state records whether a **test case** and/or
**inputs** are loaded and whether a **measurement** has been taken, and it gates which operations
are legal.

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

### 4.1 Control: `ioctl` on `/dev/executor`

**Control** goes through `ioctl` (magic `'r'`; mirrored in `aarch64_kernel.py`); **bulk data**
(test-case bytes, input bytes, measurement results) moves with `read`/`write` *after* selecting a
region with a checkout ioctl. *Valid in* lists the states in which the command does its job.

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
| 10 | `GET_TEST_LENGTH` | — | `uint64` length | test loaded (`LOADED_TEST` / `READY` / `TRACED`) | Length of the loaded test case. |
| 11 | `PAC_SIGN` | `pac_sign_req` = `ptr`, `ctx`, `mnemonic`, `keys` | signed pointer | any | Sign a pointer (`PAC*`) under the request's keys. |
| 12 | `PAC_AUTH` | `pac_sign_req` = `ptr`, `ctx`, `mnemonic`, `keys` | authenticated pointer | any | Authenticate a signed pointer (`AUT*`) under the request's keys. |
| 13 | `PAC_XPAC` | `pac_sign_req` = `ptr` | stripped pointer | any | Strip the PAC field (`XPAC`); key-independent. |
| 14 | `MTE_TAG_REGION` | `mte_tag_region_req` = `sandbox_offset`, `n_granules`, per-granule `tags` | — | any | Tag a run of sandbox granules. |

`PAC_SIGN`/`PAC_AUTH` carry the keys in the request (`keys_present` must be set, else `-EINVAL`); the
kernel keeps no PAC-key state of its own, so nothing leaks between tests. There is no set/get-keys
ioctl — a run's keys are generated by the input generator and travel in each input's `PAC_KEYS`
section (§9.4) and in each sign request.

Called outside its *Valid in* states, a command fails without changing state (`TRACE` returns
`-EINVAL`; `MEASUREMENT` is rejected unless the state is `TRACED` and an input is checked out;
`CHECKOUT_INPUT` with an unknown id leaves the selection unchanged).

*Bulk data via `read`/`write`.* To load a test case: `CHECKOUT_TEST`, then `write` the encoded
bytes. To seed an input: `ALLOCATE_INPUT`/`CHECKOUT_INPUT`, then `write` the input image. To read a
result: after `TRACE`, `CHECKOUT_INPUT`, then `MEASUREMENT` returns that input's `measurement_t`.

### 4.2 Configuration: `/sys/executor/` files

Runtime knobs are plain sysfs files under `/sys/executor/`, with read-only capability/identity files
grouped in the `/sys/executor/system/` subdirectory.

The module creates `/dev/executor` and the `/sys/executor/` tree root-owned, so an unprivileged
fuzzer run gets `EACCES`. Either run as root, or grant access after each `insmod` with
`sudo chmod a+rw /dev/executor` and `sudo chmod -R a+rw /sys/executor` (the grant resets on every
module reload). `LocalHWExecutor` raises a `PermissionError` carrying the exact `chmod` to run.

**`/sys/executor/` — runtime knobs:**

| File | Mode | Accepted value | Meaning |
|---|---|---|---|
| `measurement_mode` | rw | `P+P` or `F+R` | Prime+Probe vs Flush+Reload. |
| `warmups` | rw | unsigned int | Micro-architectural warm-up rounds before the measured run. |
| `enable_pre_run_flush` | rw | `0` / non-zero | Legacy combined knob: writing it sets the BPU flush, the PHR flush, and view rotation together (back-compat); reads back the BPU-flush component. |
| `enable_phr_flush` | rw | `0` / non-zero | Flush the path-history register (PHR) before each run, independently of the combined knob. |
| `enable_view_rotation` | rw | `0` / non-zero | Toggle the view-rotation part of the pre-run reset, independently. |
| `enable_ssbs` | rw | `0` / non-zero | Set `PSTATE.SSBS=1` on the core so it may bypass stores (required to observe Spectre-v4). |
| `pin_to_core` | rw | online CPU id | Pin execution to a CPU (invalid → current CPU). |
| `enable_branch_training` | rw | `0` / non-zero | Apply mistraining before the measured run (§12). |
| `branch_training_config` | rw | `offset:taken,…` (byte offset from the start of the test case; `1`=taken, `0`=not-taken) | Mistraining sequence; writing it also enables training. |
| `print_sandbox_base` | r | hex pointer | Base of the sandbox `main_region` (input page 0). |
| `print_code_base` | r | hex pointer | Where the test-case code is loaded, for the selected `measurement_mode`. |
| `jit_memoize` | rw | `0` / `1` | JIT-harness memoization (default `1`). Each `TRACE` calls `load_jit_template()`, which rebuilds the whole measurement harness and issues `MAX_MEASUREMENT_VIEWS+1` icache flushes — each an SMP-wide `kick_all_cpus_sync` IPI. The build is a pure function of `(test-case bytes, tc_size, measurement_template)`, so when none of those changed since the last build the rebuild+flush is skipped and the already-mapped RX harness is reused. Invalidated on test-case load/unload and `measurement_mode` change; a per-`TRACE` self-check (first TC word at the insert offset) forces a rebuild if the cache is ever stale. Set `0` to force a rebuild every `TRACE` (A/B measurement). |
| `jit_stats` | rw | counters (write=reset) | Read: `calls`/`builds`/`skipped`/`build_ns_total`/`build_ns_avg` for `load_jit_template()`. Any write resets the counters. |

**`/sys/executor/system/` — read-only capability & identity:**

| File | Mode | Meaning |
|---|---|---|
| `measurement_supported` | r | `1` if the PMU has enough counters to measure on this core, else `0`. |
| `pmu_event_counters` | r | Number of PMU event counters available. |
| `cpu_info` | r | CPU identity: `CPU ID`, `MIDR_EL1`, `MPIDR_EL1`, `CTR_EL0`. |

## 5. The contract executor (CE)

The CE produces a ctrace by *modelling* a test case's execution, including the speculation a
contract permits.

### 5.1 Protocol: pipe over `stdin`/`stdout`

The fuzzer launches the CE with `subprocess.Popen` and exchanges length-prefixed messages over its
`stdin`/`stdout`. Every message is an **8-byte little-endian header** `{length, type}` followed by
`length` payload bytes; payloads begin with the magic `RVZR`.

- **Types:** `REQUEST = 1`, `RESPONSE = 2`.
- A **request** carries a `configuration` (flags, `max_misspred_branch_nesting`,
  `execution_clauses`, `branch_predictor`, optional requested code/mem bases) plus optional
  **code**, **registers**, and **memory** sections selected by `sim_flags`
  (`HAS_CODE` / `HAS_REGS` / `HAS_MEMORY`).
- **`execution_clauses`** is a **bitmask** of independently-composable clauses:
  `EXEC_CLAUSE_COND = 1` (conditional-branch misprediction), `EXEC_CLAUSE_BPAS = 2` (speculative
  store bypass), `EXEC_CLAUSE_BPU = 4` (branch-predictor model); `0` = architectural only. Only an
  explicit allow-list of combinations is accepted; any other combination traps.
- **`branch_predictor`** selects the model used when `EXEC_CLAUSE_BPU` is set —
  `BRANCH_PREDICTOR_NONE = 0`, `BRANCH_PREDICTOR_NEOVERSE_N3 = 1` (TAGE).
- The **response** carries the per-instruction trace (registers, PC, memory accesses, speculation
  nesting) the fuzzer turns into the ctrace.

The CE also **embeds CPython** (`Py_Initialize`): the Neoverse-N3 contract calls a Python TAGE
branch-predictor model from inside the C process.

### 5.2 How it models a program

The CE rewrites every test-case instruction into a `BL hook` trampoline, then walks the
instructions one at a time. At each *speculation point* — a conditional branch (mispredict) or,
under the `bpas` clause, a store (bypass) — it checkpoints, explores the speculative path, and
rolls back.

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

The speculation engine is built so contracts are **composable**: a run enables a set of independent
*execution clauses* and the engine runs them together over one instruction stream. The clauses:

- **`cond`** (always-mispredict): at a conditional branch, checkpoint the architectural
  continuation and redirect to the mispredicted path.
- **`bpu`**: like `cond`, but the branch outcome is driven by the injected predictor model
  (Neoverse-N3 TAGE), so only branches the model mispredicts fork.
- **`bpas`** (speculative store bypass / Spectre-v4): at a store, snapshot memory and let the model
  run the store, then on the next instruction restore the pre-store snapshot so the speculative
  window reads the **stale** value.

  A bypassed store's **trace entry is deferred**: it is pulled out of the trace at bypass time and
  re-emitted when its window unwinds. Taint is derived from trace order (a byte is must-preserve if
  it is read before it is written on its flow), so the store must be logged *after* the loads that
  read its stale value — otherwise the store's write would mask them and the bypassed **input byte
  would not be tainted**, letting boosting mutate it and produce a false-positive violation. Emitting
  the store post-window instead lets those loads taint the input while post-window loads still see
  the committed store. Nested bypasses stack (one deferred entry per open window, LIFO), so with two
  nested stores the trace visits *bypass-both → inner store committed → bypass-only-outer → both
  committed*, and the inner store is re-emitted before the outer.
- **`barrier`** (honor `SSBB`/`PSSBB`): these are store→load *reordering* barriers. A barrier does
  **not** squash store-bypass speculation — a value already bypassed into a register *before* the
  barrier stays stale and can still be transmitted *after* it (exactly the Neoverse-N3 behavior).
  Instead, at the barrier every open bypassed store is committed to live memory (`after` written back
  at its address, oldest→newest so the last store wins), so loads *past* the barrier see committed
  values — no forwarding across the barrier — while registers and the speculation itself continue
  untouched; each window still rolls back to the architectural state at its natural end. The stronger
  barriers `DSB`/`ISB`/`SB` (which complete prior accesses / flush the pipeline / general speculation
  barrier) instead **end** the mispredicted path — they squash the open bypass windows. `DMB` (ordering
  only) fences nothing here. For nested stores the partial-bypass flows over-approximate, which is
  conservative (the contract predicts more leakage, never less).

**Window-scoped taint.** Every trace entry carries a **`window_id`** — a monotonic id, unique per
speculative excursion (0 = architectural), that the engine stamps from the innermost open window.
Unlike the nesting *depth* (or the checkpoint slot, which is reused LIFO), it is never reused, so a
same-depth re-fork after an unwind is a *distinct* window. `compute_taint` scopes writes by window:
a read is masked only by writes made in windows on its live root-to-node path, so a write on a
squashed sibling flow cannot hide an input read that a later, unrelated flow performs at the same
depth. Without this, a speculative flag write (e.g. `SUBS` on a mispredicted path) would mask a
conditional branch's read of the seed flag and boosting would flip the branch — a false positive.

Supported clause combinations are validated in both Python and C: `seq` (empty), `cond`, `bpas`,
`bpu`, and `cond|bpas`; two branch models together (`cond|bpu`) are rejected.

**Speculation bounds.** Two config caps bound the exploration: `max_misspred_branch_nesting` (how many
windows may be open at once) and `max_misspred_instructions` (a window's instruction budget). The
budget is counted **per window, on its own path** — instructions executed in a nested window that
later unwinds are *not* charged to the enclosing window. Counting against a global instruction total
instead would force-close an outer `bpas` bypass window as soon as a `cond` misprediction ran inside
it, dropping the store-bypass flow and breaking composition monotonicity — the guarantee that enabling
more clauses only *adds* observations (`footprint(cond|bpas) ⊇ footprint(bpas)`). The per-instruction
trace log is sized for this bounded exploration; overflowing it is a **hard error** (the CE aborts
loudly), never silent truncation.

### 5.3 Branch-predictor model (reverse-engineered Neoverse-N3)

The `bpu` clause drives branch outcomes with a model of the target core's predictor — a TAGE
direction predictor for the Arm Neoverse-N3 (`saturating_bp.py::Aarch64NeoverseN3BPU`, configured in
`bootstrap_director.py`). Some of it is reverse-engineered from the silicon; the rest is an
explicit, labelled model.

**Structure.** A bimodal base table plus tagged tables, indexed and tagged by the branch PC folded
with progressively longer slices of a **path-history register (PHR)**. A prediction is the counter
of the longest table whose tag matches; the base is the always-available fallback.

```
  PHR — path history, advanced on TAKEN branches:  PHR = (PHR << 4) ^ footprint(PC, target)
        footprint = 4 bits over fixed PC/target bit positions

  bit 0                              171                             299
   |==================================|===============================|
   |<------ T1 uses PHR[0..171] ------>|                               |
   |<----------------- T2 uses PHR[0..299] ------------------------->|
                            ( T2's slice contains T1's — geometric )

   PC ---> +-----------+   +-----------+   +-----------+
           |  base T0  |   | tagged T1 |   | tagged T2 |
           | bimodal   |   | hist 172  |   | hist 300  |
           | 2^12 ent* |   | 2^11 ent* |   | 2^11 ent* |
           | 3-bit ctr |   | 3-bit ctr*|   | 3-bit ctr |
           | untagged  |   | tagged    |   | tagged    |
           +-----+-----+   +-----+-----+   +-----+-----+
                 |               |               |
                 +-------+-------+-------+-------+
                         v
      predict = counter of the LONGEST table whose TAG matches
                (T2 > T1 > T0); else the base table T0

   index = PC[8:18] combined with the table's PHR slice   (T0: none)
   tag   = PC[2:7], folded with the PHR slice *  (+ deeper PHR bits *)
   ( the PC[8:18] / PC[2:7] split and the deep-PHR tag were established on T2,
     the long-history table; T1 is presumed to follow the same scheme )

  * = NOT reverse-engineered (model placeholder / low-confidence):
      the table entry counts (2^12, 2^11), T1's counter width (presumed 3-bit),
      the TAG fold (algorithm + width) and its deep-PHR dependence, the tag width (= 8),
      and the u-bit / altpred (modelled as plain LRU within a set).
```

**Reverse-engineered facts** (from the N3 silicon; see the `kb-bpu-re` notes):

- The global history is **path** history: it advances on **taken** branches only.
- Each taken branch shifts the PHR left by 4 bits and XORs in a 4-bit footprint of the branch,
  computed from fixed PC and target bit positions:
  - `b0 = pc[2] ^ pc[6] ^ target[3] ^ target[7]`
  - `b1 = pc[3] ^ pc[7] ^ target[4] ^ target[8]`
  - `b2 = pc[4] ^ pc[8] ^ target[5] ^ target[9]`
  - `b3 = pc[5] ^ pc[9] ^ target[6] ^ target[10]`
- The two tagged tables use history lengths **172** and **300**.
- Each entry holds a **3-bit** saturating counter (confirmed on the base table and the
  300-history-length table; the 172-length table is presumed the same).
- PC bits **[2:18]** feed the index/tag functions. Within that range, bits **[2:7]** appear in the
  **tag** (not the index) and bits **[8:18]** appear in the **index** (whether [8:18] also feed the
  tag is not established).

**Tentative — low-confidence, from experiments only (fragile, not implemented):** the tag appears to
combine PC bits with *deep* PHR bits, stepped by 10. Two tag inputs looked like

```
tag_a ~ PC[3] ^ PC[5] ^ PC[7] ^ PHR[299] ^ PHR[289] ^ ... ^ PHR[178]
tag_b ~ PC[2] ^ PC[4] ^ PC[6] ^ PHR[298] ^ PHR[288] ^ ... ^ PHR[177]
```

and probably continue deeper than the range that was observable. Treat this as a hypothesis, not a
result.

**Separating test cases in the BPU.** Because PC bits [2:18] feed the index/tag, changing a bit in
that range moves a branch to a different tag or index and so **de-aliases** it from another branch —
useful for keeping test cases (or inputs) from sharing predictor state. Changing an **index** bit
([8:18]) is the stronger choice: it gives a *physical* separation into a different table entry,
whereas changing a tag-only bit ([2:7]) merely forces a tag mismatch within the same entry.

**Speculation** (textbook TAGE + speculative global history): the PHR is advanced *speculatively* at
fetch with the predicted direction and checkpointed, so a misprediction rolls it back; the counter
tables train **at retire only** (architectural path, `spec_nesting == 0`), so wrong-path branches
never train them and the tables need no rollback.

**Modelling placeholders** — *not* reverse-engineered, and made explicitly at the construction seam
(`bootstrap_director.create_predictor`) rather than baked in silently:

- the PHR fold algorithm and its group width (`PHR_FOLD_GROUP_WIDTH = 11`) for the index/tag;
- the tagged-table tag width (`TAG_WIDTH = 8`, arbitrary — a finite tag does make distinct branches
  alias, as real TAGE does);
- the "useful" (u) bit and everything it drives (u-gated replacement and aging), replaced here by
  plain LRU within a set;
- the `altpred` / `USE_ALT_ON_NA` fallback for freshly-allocated entries.

## 6. Contracts: what you can detect

A contract has two halves, each chosen in the config:

- an **execution clause** (`contract_execution_clause`) — what speculation the model simulates;
- an **observation clause** (`contract_observation_clause`, default `l1d`) — what the model treats
  as observable (for `l1d`, the set of L1 data-cache sets touched).

The execution clause decides which leak class shows up as a violation — anything the hardware leaks
*beyond* what the clause models:

| clause | models | effect |
|---|---|---|
| `seq` | architectural execution only | any speculative leak (Spectre-v1 or v4) is a violation |
| `cond` | architectural + conditional-branch misprediction | branch bypass (v1) is now in-contract; only store bypass (v4) remains a violation |
| `bpas` | architectural + speculative store bypass | used to model / confirm v4 |
| `bpu` | architectural + predictor-driven misprediction | like `cond`, but only mispredicted branches fork |

The **observation clause** (`contract_observation_clause`) is likewise configurable, mirroring the
x86 tracer set: `l1d` (default — the union of L1 data-cache sets touched), `pc` (the control-flow /
program-counter trace), `memory` (the addresses of memory accesses), `ct` (PC + addresses, the
classic constant-time observer), and the remaining x86 variants (`loads+stores+pc`,
`ct-nonspecstore`, `ctr`, `arch`, `tct`, `tcto`). `l1d` records the **same** union of cache sets the
htrace measures, so the ctrace and htrace compare directly; the others let a run target a different
observation model.

**Spectre-v1** uses the arch-only contract (`seq`) with the branch-predictor flush off
(`enable_pre_run_flush: 0`), so a guarding branch mispredicts naturally and a speculative load lands
in a forbidden cache set. **Spectre-v4** uses `cond` (so v1 is in-contract) plus
`enable_speculative_store_bypass: true` (sets `PSTATE.SSBS=1`, so the CPU may bypass stores); the
only remaining unmodeled speculation is store bypass, so a violation can only be v4. **Prime+Probe**
is the more sensitive channel for both.

### 6.1 Non-interference fuzzing (PAC & MTE)

A second scheme targets Arm's **pointer authentication (PAC)** and **memory tagging (MTE)** — leak
primitives a single ctrace cannot capture. Instead of many inputs against one program, it runs
**one input against many sibling programs** and compares their htraces (*non-interference*).

The sibling programs are produced by **sealing**: each speculative-leak primitive in the test case
is replaced by a fixed-width instruction slot the fuzzer fills per variant. Three independent,
composable seals:

- **Sandbox** — clamps a memory access into the sandbox (the only seal that touches memory);
- **PacSign** — signs / authenticates a pointer register (PAC);
- **MteTag** — retags a register (MTE).

A **baseline** variant fills every slot with genuine values; each **decoy** variant perturbs only
the speculative slots of the single primitive under test (the *non-interference target*, PAC or MTE),
leaving everything else — including the Sandbox clamp — genuine. If the primitive does not leak, the
baseline and decoys are architecturally identical and produce the same htrace; a **difference**
between baseline and a decoy is a leak of that primitive.

Sealing *every* primitive (not just the target) is what keeps a run safe and single-pass: an MTE run
whose test case also contains PAC `AUT*` instructions must still authenticate those pointers
genuinely. One CE pass over the sealed program fills every slot (signs the PAC slots, classifies the
MTE slots); the decoy policy then perturbs only the target's speculative slots.

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

### 6.2 Variant assembly by relocation (throughput)

The sibling variants for one input differ from a shared skeleton only at the sealing slots — the
`MOVK` carrying the per-input PAC signature, the `AUT*`/`XPAC` that follows it, and the MTE `ADDG`
retag. Re-assembling and re-loading a full test case per variant would be the bulk of the non-HW
time on the sealed path. Instead the skeleton is loaded **once** and each input carries a small
**code-relocation table** that patches only those slots.

A **relocation** is `(offset, value)` of type `WORD32`: overwrite the 4-byte little-endian word at
`offset` of the test-case body. The Python resolver renders each slot's word directly —
`set_movk_imm16` rewrites the 16-bit signature into the harvested `MOVK`, and the op word is the
harvested `AUT*` / `XPAC` / `ADDG` — so **no assembler runs per variant** (`aarch64_relocations.py`).
The resolver emits a `RelocationPlan`; the encoder packs it into the REIF `CODE_RELOC` section
(§9.4).

The patch is applied **in the kernel**, not in Python. Immediately before an input executes, the
kernel splices that input's relocations into the shared body on the pinned CPU (with local i-cache
maintenance), runs it, then **reverts** the body to the pristine test case — so one loaded test case
serves many differently-sealed inputs, and there is no per-variant assemble *or* per-variant device
load. Unlike PAC/`XPAC` slots, the MTE tag-delta seal is now just another `WORD32` (the `ADDG`
word), so MTE and PAC variants relocate through the same path. Relocation offsets are bounds-checked
against the loaded test case at trace time; a bad splice would run a wrong-signed `AUT*` and
FPAC-fault the box, so correctness is covered by the encoder/round-trip tests rather than a hot-path
check. It composes with the `in_memory_assemble` source-text cache (§2).

## 7. Techniques: triaging a violation

**Goal.** A `fuzz` run reports a *contract violation* — two inputs that are architecturally
equivalent yet produced different hardware traces. That difference is either a genuine speculative
leak or measurement **noise**, and triage decides which. It does so **without re-running the test on
hardware**: re-measuring the same test trains the branch predictor and can *hide* a genuine leak, so
repetition is never a way to "confirm noise".

**How it is done.** Re-run the two violating inputs through the CE (the model) and compare their
cache-line footprints:

1. their **architectural** cache lines must be identical — they are arch-equivalent by construction;
2. their **speculative** (wrong-path) cache lines must **differ** — that is the candidate leak;
3. that speculative divergence must **match** the cache-set divergence the hardware htraces already
   show in the saved report.

All three holding is a **GENUINE** verdict; speculative lines that don't differ, or don't line up
with the HW divergence, are **NOISE**; anything ambiguous is **INVESTIGATE**. This is what the
`revizor-violation-triage` skill (under `.claude/skills/`) computes.

**Spectre-v1 (branch bypass).** Re-trace the pair under **always-mispredict** (`seq`/`cond` with the
branch mispredicting): the speculative divergence is the mispredicted load landing in a different
cache set for each input.

**Spectre-v4 (store bypass).** The branch-mispredict model will not reveal it — re-trace the pair
under the **`bpas`** (speculative-store-bypass) clause instead, and check that the store-bypass
speculative sets diverge between the pair and match the HW divergence. The decisive, v4-specific
proof is the **SSBS on/off** control: re-measure the pair with `enable_speculative_store_bypass`
**off** — a genuine v4 divergence must **vanish** (with store bypass disallowed the leak cannot
occur), which rules out v1, architectural effects, and noise.

A **fully offline** alternative confirms v4 from CE traces alone (no HW at all). For the two inputs
`I0`, `I1`, the pair is already `seq`- and `cond`-equivalent **by construction** — i.e.
`CT-SEQ(I0) == CT-SEQ(I1)` and `CT-COND(I0) == CT-COND(I1)`, which is how the fuzzer grouped them —
so re-checking those equalities is only a sanity check, not the discriminator. The actual test is
that the pair **diverges** under store bypass, `CT-BPAS(I0) != CT-BPAS(I1)`, and that this
divergence matches the cache-set divergence in the measured htraces: only store bypass separates the
pair, and it matches hardware ⇒ v4.

**Reproducing on hardware.** The `reproduce-revizor-violation` skill confirms a leak on the CPU by
recreating the exact micro-architectural state: load the context inputs `0..min(A,B)`, **swap only
the violating input**, measure, and read which cache set encodes which input ran. The leak is
intermittent, so collect statistics over many trials.

These are quick verification heuristics, not proofs: a GENUINE verdict plus a clean reproduction is
strong corroboration.

## 8. Memory layouts

### 8.1 Sandbox

A test case may only touch a fixed arena. During the run `x29` holds the **main-region** base;
every memory access is masked to `x29 + (reg & 0x1fff)`, so it always lands in `main`+`faulty`
(8 KB).

The target core's L1 data cache is **64 KB, 4-way set-associative** with 64-byte lines
(`L1D_SIZE_K=64`, `L1D_ASSOCIATIVITY=4` in the executor Makefile), giving 256 sets and a 16 KB
conflict distance — the stride at which addresses collide in a set, which is what Prime+Probe
evicts along. The htrace is a 64-entry cache-set bitmap; an access at sandbox `offset` maps to bit
`(offset // 64) % 64`.

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

### 8.2 Input register region

An input seeds memory (`main` + `faulty`) and the registers. The register region is 8 × 64-bit
slots:

| Slot | Register |
|---|---|
| 0–5 | `x0`–`x5` |
| 6 | NZCV flags (per-flag encoding, below) |
| 7 | unused (the executor forces `sp` to the sandbox stack base) |

The flags slot uses a **per-flag NZCV encoding**: each of N, Z, C, V occupies bit 0 of its own byte
within the slot. This lets the `aarch64-nzcv` input generator **fuzz each flag separately** — and,
because taint is tracked per byte, lets the trace depend on each flag independently instead of
lumping all four into one word. The slot is converted to ARM `PSTATE` form just before execution.

## 9. File formats

### 9.1 Run config (YAML)

A run is configured by a YAML file passed with `-c`. Options not set take their architecture
defaults (`src/aarch64/aarch64_config.py` and the shared `src/config.py`). A representative config:

```yaml
instruction_set: aarch64

# Which instruction families the generator may emit (§10).
instruction_categories:
  - BASE-ARITH
  - BASE-LOGICAL
  - BASE-SHIFT
  - BASE-BRANCH
  - BASE-MEM-LOAD
  - BASE-MEM-STORE

# Shape of each generated test case.
min_bb_per_function: 2
max_bb_per_function: 5
min_successors_per_bb: 1
max_successors_per_bb: 2

# Contract (§6).
contract_execution_clause: [seq]     # seq | cond | bpas | bpu
contract_observation_clause: l1d     # what the ctrace records

# Measurement.
executor_mode: P+P                   # P+P (Prime+Probe) or F+R (Flush+Reload)
executor_warmups: 5                  # warm-up rounds before the measured run
enable_pre_run_flush: 0              # 0 => branches mispredict naturally (needed for v1)
enable_speculative_store_bypass: false   # true => PSTATE.SSBS=1 (needed for v4)

# Inputs.
input_generator: aarch64-nzcv
input_gen_seed: 0
input_gen_entropy_bits: 16           # 1..32; entropy of generated input values

fuzzer: basic
```

Commonly-set options, by group:

| Group | Options |
|---|---|
| Scope | `instruction_set`, `instruction_categories`, `supported_instructions`, `instruction_allowlist`, `instruction_blocklist` |
| Program shape | `min_bb_per_function`, `max_bb_per_function`, `min_successors_per_bb`, `max_successors_per_bb`, `avg_mem_accesses` |
| Contract | `contract_execution_clause`, `contract_observation_clause`, `model_max_nesting` |
| Measurement | `executor_mode`, `executor_warmups`, `enable_pre_run_flush`, `enable_speculative_store_bypass` |
| Inputs | `input_generator`, `input_gen_seed`, `input_gen_entropy_bits`, `inputs_per_class` |

The number of test cases and inputs per run are CLI flags, not config keys: `-n <test cases>` and
`-i <inputs>`.

### 9.2 Instruction-set spec (`base.json`)

`base.json` is the instruction-set spec the generator emits from. It is **not hand-edited** — it is
generated from ARM's machine-readable A64 ISA (§11) — but its shape is worth knowing.

The file is a JSON **list** of instruction nodes. `isa_loader` reads these fields per node:

| Field | Meaning |
|---|---|
| `name` | Mnemonic (e.g. `"add"`). |
| `category` | Coarse ISA category; only `"general"` is emittable (§10). |
| `control_flow` | `true` for branches / control-flow instructions. |
| `template` | Python format string used to render the instruction to assembly text. |
| `tags` | List of taxonomy tags (§10); defaults to `[category]` if absent. |
| `operands` | List of operand specs (below). |
| `implicit_operands` | Operands read/written but not spelled out in the text (e.g. flags). |

Each **operand** node has `type_` (`REG`, `IMM`, `MEM`, `LABEL`, `COND`, or `FLAGS`), `name` (the
placeholder used by `template`), `width` (bits), `signed`, `src`/`dest` (read/written), and `values`
(the allowed choices). **Memory operands** (`type_ == "MEM"`) decompose their address into an
`inner` list of operand nodes — base register, index, offset, extend — each carrying a role; the
access *direction* comes from the operand's own `src`/`dest` (load reads, store writes, RMW both),
taken from the ISA definition rather than inferred from register positions. One node:

```json
{
  "name": "add",
  "category": "general",
  "control_flow": false,
  "tags": ["BASE-ARITH"],
  "template": "add {Wd}, {Wn}, {Wm}",
  "operands": [
    {"type_": "REG", "name": "Wd", "width": 32, "signed": false, "src": false, "dest": true,  "values": ["w0", "w1", "..."]},
    {"type_": "REG", "name": "Wn", "width": 32, "signed": false, "src": true,  "dest": false, "values": ["w0", "w1", "..."]},
    {"type_": "REG", "name": "Wm", "width": 32, "signed": false, "src": true,  "dest": false, "values": ["w0", "w1", "..."]}
  ],
  "implicit_operands": []
}
```

### 9.3 Operand model (Python)

Specs and runtime instructions share one operand model (`src/interfaces.py`), common to both ISAs:

- `OperandSpec` — one operand template: `type`, `width`, `signed`, `src`/`dest`, `values`, `name`.
- `MemorySpec(OperandSpec)` — a memory access (`type == OT.MEM`) that wraps its address components
  in `inner: List[OperandSpec]`; `src`/`dest` give the access direction.
- `MemoryRole` — an ISA-subtyped role enum; `AArch64MemRole` = `BASE`/`INDEX`/`OFFSET`/`EXTEND`.
  Only inner components carry a role.
- Runtime mirrors: `Operand` and `MemoryOperand(inner)`. `Instruction.to_asm_string` renders from
  the spec's `template`, flattening each `MemoryOperand`'s inner components into the substitution
  dict by name.

The sandbox finds the `BASE` component **by role** and confines it (`AND base,#mask; ADD base,x29`),
cancelling every other component so the effective address stays `x29 + (base & mask)`; it iterates
every memory operand, so multi-access instructions are all confined.

### 9.4 Executor input file (REIF)

One input is a **REIF** file — the Revizor Extensible Input File (`executor_input_format.h`; full
spec in [`docs/reif_input_format.md`](../reif_input_format.md)). It is self-describing: the kernel
validates it strictly and locates sections **by type**, never by a hardcoded offset, so new
per-input state is added without a layout break. Every field is little-endian `u64`; the layout is a
fixed preamble, a section table, then the payloads.

The **same bytes** serve both roles. On the wire they are written to `/dev/executor`. On disk they
are saved with the **`.reif`** extension, and a `.reif` file is complete and executor-ready — it
loads verbatim into the device (flags already in PSTATE form, every section self-located) or back
into Python via `ExecutorInput.deserialize()` for a `reproduce` run. There is no separate "convert
before executing" step.

**Preamble** — `struct revisor_input_header` (48 bytes):

| Field | Meaning |
|---|---|
| `magic` | `RVZRI` sentinel (`0x49525A5652`) |
| `version` | format version (1) |
| `header_len` | `48 + 32·n_sections` — offset of the first payload |
| `n_sections` | number of section descriptors |
| `flags` | reserved |
| `total_len` | total byte count (equals the bytes written to the device) |

**Section table** — `n_sections` × `struct revisor_input_section` (32 bytes each):
`{type, flags, offset, length}`, `offset`/`length` locating the payload within the file.

**Section payloads** (unknown types are skipped):

| Type | Contents |
|---|---|
| `MEMORY_MAIN` (1) | sandbox `main_region` bytes |
| `MEMORY_FAULTY` (2) | sandbox `faulty_region` bytes |
| `GPR` (3) | `registers_t` = x0..x5, the flags register (ARM PSTATE, NZCV in bits 31:28), sp |
| `SIMD` (4) | v0..v7 (256 B; reserved, not yet loaded) |
| `PAC_KEYS` (5) | the PAC keys this input's signatures were signed under — generated per-campaign by the input generator; the kernel keeps no key state of its own |
| `MTE_TAGS` (6) | one 4-bit tag per 16-byte granule of main‖faulty, packed two per byte (granule 2i in bits 3:0, 2i+1 in bits 7:4) |
| `CODE_RELOC` (7) | per-input code-relocation table — `WORD32` patches spliced into the shared test-case body by the kernel, then reverted (§6.2) |
| `BPU_TRAINING` (8) | per-input branch-training entries (`enable_branch_mistraining`, off by default) |

`MEMORY_MAIN`/`MEMORY_FAULTY`/`GPR` are required; the rest are optional (absent ⇒ kernel default).
The `GPR` section carries the flags **already** in PSTATE form — the writer reconstructs the per-flag
NZCV encoding (§8.2) before packing. `PAC_KEYS`/`MTE_TAGS`/`CODE_RELOC` are the per-input state the
non-interference seal (§6.1, §6.2) uses.

### 9.5 Contract-executor request format

A CE request travels as one pipe message: the 8-byte `{length, type}` header (§5.1) wraps a payload
of a **16×u64 little-endian envelope**, then the code, then one input:

| u64 | Field |
|---|---|
| 0 | magic (`RVZRCE`) |
| 1 | version |
| 2 | arch (`AARCH64 = 2`) |
| 3 | `sim_flags` (`HAS_CODE` \| `HAS_INPUT`) |
| 4 | `config_flags` (which base-address requests are present) |
| 5 | `max_misspred_branch_nesting` |
| 6 | `max_misspred_instructions` (per-window instruction budget, §5.2) |
| 7–10 | requested code/mem base phys/virt (optional) |
| 11 | `execution_clauses` bitmask (§6) |
| 12 | `branch_predictor` enum |
| 13 | `code_size` |
| 14 | `input_init_size` |
| 15 | reserved |

followed by `code_size` bytes of machine code, then an `input_init_size`-byte **input in exactly the
REIF format of §9.4**. The CE reads the `GPR` slots from that input (plus the `PAC_KEYS` / `MTE_TAGS`
sections in PAC/MTE runs). The response carries the per-instruction trace of §5.1.

## 10. Instruction taxonomy

The generator emits only instructions the spec marks as supported, scoped to a set of
**categories**. Both are driven by tags in `base.json`.

**Tags.** Every spec carries a `tags` list, coarse-to-fine: an ISA prefix (`BASE`/`SVE`/`SME`), a
coarse family (`BASE-ARITH`, `BASE-MEM`, `BASE-BRANCH`, `BASE-FLAGS`, …), and optional fine tags
(`BASE-MEM-LOAD`/`-STORE`/`-ATOMIC`, `BASE-BRANCH-COND`, …). Every spec gets at least one tag.

**Filtering.** `isa_loader` decides the emittable set as a conjunction of gates — a spec is emitted
only if it passes **all** of them:

1. **Category gate** — only `category == "general"` specs are eligible (FP/SIMD, SVE/SME, float, and
   system categories are excluded: the generator and CE model only the general-purpose subset).
2. **`supported_instructions`** — a curated mnemonic allow-list; the emittable set is essentially a
   hand-picked subset of the general category.
3. **Per-name overrides** — `instruction_allowlist` forces a name in; `instruction_blocklist` drops
   permanent hazards (traps, stalls, or instructions absent on the target core).
4. **`instruction_categories`** (from the run config) — the spec is kept only if at least one of its
   tags is an enabled category. This is how a run scopes itself.
5. **Address-register blocklist** — specs whose memory base/index would use a reserved register are
   dropped.

In short: **emittable ≈ general ∩ supported_instructions ∩ ¬blocklist ∩ (tag ∈ categories)**, with
the allow-list as a per-name override.

## 11. Running

Build the instruction-set spec once (downloads ARM's A64 ISA XML and parses it into `base.json`;
the download is cached, so re-runs are fast):

```
python revizor.py download_spec -a aarch64 -o base.json
```

Then fuzz — pass the spec with `-s`, the config with `-c`, the number of test cases with `-n`, and
inputs per test case with `-i`:

```
python revizor.py fuzz -s base.json -c configs/spectre_v1_pp.yml -n 200 -i 50 --save-violations true -w out
```

A violation drops a `violation-*/` artifact in the working directory (the test case, the
counterexample inputs, traces, and a `reproduce.yaml`). Measuring on hardware needs the kernel
module loaded (`--test aarch64-ko`, or a manual `insmod`).

`revizor.py minimize` reduces a saved violation to a minimal reproducer. The minimizer is
architecture-agnostic — it sources the NOP, the speculation barrier (`DSB SY`), and branch
detection from the AArch64 target description — so the standard passes (instruction removal, NOP
replacement, constant/mask simplification, input-sequence and differential-input minimization) all
run on AArch64.

### Testing

`src/aarch64/install_revizor_env.sh --test <group>` runs the suites:

| group | what it runs |
|---|---|
| `aarch64-ce` | contract-executor C tests (`test_ce` unit + `test_ce_integration`) |
| `aarch64-ko` | builds + (re)loads the kernel module, then the `/dev/executor` Python tests |
| `aarch64-python` | all Python unit tests (common + `tests/aarch64_tests/`) |
| `aarch64-all` | all of the above |

Tests also run from the repo root: `python3 -m unittest discover -s tests -p 'unit_*.py'` (common)
and `python3 -m unittest discover -s tests/aarch64_tests -p 'unit_*.py' -t tests/aarch64_tests`
(AArch64). `tests/runtests.sh` runs the whole thing.

## 12. Experimental: work in progress

**Mistraining.** Branch mistraining saturates a conditional branch *opposite* its architectural
direction before the measured run, so it mispredicts on the measured run and opens a speculative
window. It is driven by the CE's architectural branch directions (`branch_mistraining_entries` →
`apply_branch_mistraining`) and applied through the `enable_branch_training` +
`branch_training_config` sysfs knobs (§4.2). It is **off by default**
(`enable_branch_mistraining = False`): the training direction is correct (opposite = mispredict) but
its effectiveness on the target core is not yet confirmed, so Spectre-v1 detection currently relies
on the branch mispredicting *naturally* (`enable_pre_run_flush: 0`). Re-enable once validated on
hardware.

**Directed fuzzing.** `src/directed_fuzzing/` is a self-contained alternative to purely random
generation: it uses Monte-Carlo Tree Search over a micro-architecture simulator to steer test-case
generation toward programs more likely to leak, rather than sampling uniformly. It is not wired into
the main fuzzer yet.

## Glossary

| Term | Meaning |
|---|---|
| **Contract** | Promise of what data may influence observable state. Has an *execution clause* (what the model simulates) and an *observation clause* (what is observable). |
| **ctrace** | Contract trace — what the model says is observable for a program + input. |
| **htrace** | Hardware trace — what the CPU actually leaked; a 64-bit cache-set bitmap. |
| **Violation** | Two inputs with equal ctrace but different htrace. |
| **CE** | Contract executor; produces ctraces by modelling the program, including speculation. |
| **Taint** | The input bytes the ctrace depends on — those read along the traced execution paths. |
| **Boosting** | Cheaply producing more inputs with the *same* ctrace by mutating only the untainted bytes. |
| **Nesting** | Depth of speculative execution the model explores. |
| **Prime+Probe (P+P)** | Fill the cache with attacker lines, run the victim, then detect which lines were evicted. |
| **Flush+Reload (F+R)** | Flush shared lines, run the victim, then detect which lines were reloaded. |
| **Cache set** | One of 64 buckets an address maps to; the unit of the htrace bitmap. |
| **Sandbox** | The memory a test case may touch: `main` + `faulty` input pages, with zeroed overflow guards. |
| **Mistraining** | Steering the branch predictor before the measured run to provoke misprediction. |
| **PAC / MTE** | Arm Pointer Authentication / Memory Tagging. |
