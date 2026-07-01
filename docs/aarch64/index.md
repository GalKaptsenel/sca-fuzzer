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
launched once and kept alive to serve many requests; `asm_to_bytes` is a one-shot helper forked per
call.

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
8. **Report.** Save the confirmed violation — test case, inputs, and traces — for triage (§8).

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
| 10 | `GET_TEST_LENGTH` | — | `uint64` length | any | Length of the loaded test case. |
| 11 | `SET_PAC_KEYS` | `pac_keys` (5×128-bit: IA/IB/DA/DB/GA) | — | any | Set the keys the executor signs/auths with. |
| 12 | `GET_PAC_KEYS` | — | `pac_keys` | any | Read the keys the executor will use. |
| 13 | `PAC_SIGN` | `pac_sign_req` = `ptr`, `ctx`, `mnemonic` | signed pointer | any | Sign a pointer (`PAC*`). |
| 14 | `PAC_AUTH` | `pac_sign_req` = `ptr`, `ctx`, `mnemonic` | authenticated pointer | any | Authenticate a signed pointer (`AUT*`). |
| 15 | `PAC_XPAC` | `pac_sign_req` = `ptr` | stripped pointer | any | Strip the PAC field (`XPAC`). |
| 16 | `MTE_TAG_REGION` | `mte_tag_region_req` = `sandbox_offset`, `n_granules`, per-granule `tags` | — | any | Tag a run of sandbox granules. |

Called outside its *Valid in* states, a command fails without changing state (`TRACE` returns
`-EINVAL`; `MEASUREMENT` is rejected unless the state is `TRACED` and an input is checked out;
`CHECKOUT_INPUT` with an unknown id leaves the selection unchanged).

*Bulk data via `read`/`write`.* To load a test case: `CHECKOUT_TEST`, then `write` the encoded
bytes. To seed an input: `ALLOCATE_INPUT`/`CHECKOUT_INPUT`, then `write` the input image. To read a
result: after `TRACE`, `CHECKOUT_INPUT`, then `MEASUREMENT` returns that input's `measurement_t`.

### 4.2 Configuration: `/sys/executor/` files

Runtime knobs are plain sysfs files.

| File | Mode | Accepted value | Meaning |
|---|---|---|---|
| `measurement_mode` | rw | `P+P` or `F+R` | Prime+Probe vs Flush+Reload. |
| `warmups` | rw | unsigned int | Micro-architectural warm-up rounds before the measured run. |
| `enable_pre_run_flush` | rw | `0` / non-zero | Flush the branch-predictor history before each run so the previous run doesn't bias the next. |
| `pin_to_core` | rw | online CPU id | Pin execution to a CPU (invalid → current CPU). |
| `enable_branch_training` | rw | `0` / non-zero | Apply mistraining before the measured run (§7). |
| `branch_training_config` | rw | `offset:taken,…` (byte offset from the start of the test case; `1`=taken, `0`=not-taken) | Mistraining sequence; writing it also enables training. |
| `print_sandbox_base` | r | hex pointer | Base of the sandbox `main_region` (input page 0). |
| `print_code_base` | r | hex pointer | Where the test-case code is loaded, for the selected `measurement_mode`. |

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
*execution clauses* and the engine runs them together over one instruction stream. The three
clauses:

- **`cond`** (always-mispredict): at a conditional branch, checkpoint the architectural
  continuation and redirect to the mispredicted path.
- **`bpu`**: like `cond`, but the branch outcome is driven by the injected predictor model
  (Neoverse-N3 TAGE), so only branches the model mispredicts fork.
- **`bpas`** (speculative store bypass / Spectre-v4): at a store, snapshot memory and let the model
  run the store, then on the next instruction restore the pre-store snapshot so the speculative
  window reads the **stale** value.

Supported clause combinations are validated in both Python and C: `seq` (empty), `cond`, `bpas`,
`bpu`, and `cond|bpas`; two branch models together (`cond|bpu`) are rejected.

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

**Spectre-v1** uses the arch-only contract (`seq`) with the branch-predictor flush off
(`enable_pre_run_flush: 0`), so a guarding branch mispredicts naturally and a speculative load lands
in a forbidden cache set. **Spectre-v4** uses `cond` (so v1 is in-contract) plus
`enable_speculative_store_bypass: true` (sets `PSTATE.SSBS=1`, so the CPU may bypass stores); the
only remaining unmodeled speculation is store bypass, so a violation can only be v4. **Prime+Probe**
is the more sensitive channel for both.

### PAC / non-interference — one input, many sibling programs

Some campaigns instead run **one input against many variants of the same program** (e.g. different
PAC signing slots) and compare their htraces; a difference across variants is a
primitive-related leak.

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

## 7. Mistraining

Branch mistraining saturates a conditional branch **opposite** its architectural direction before
the measured run, so the branch mispredicts and a speculative window opens. It is driven by the
CE's architectural branch directions (`branch_mistraining_entries` → `apply_branch_mistraining`) and
applied through the `enable_branch_training` + `branch_training_config` sysfs knobs (§4.2).

It is **off by default** (`enable_branch_mistraining = False`), pending hardware confirmation that
the training is effective on the target core. With it off, Spectre-v1 relies on the guarding branch
mispredicting naturally (`enable_pre_run_flush: 0`).

## 8. Techniques: triaging a violation

A `fuzz` run reports a *contract violation* — two inputs that are architecturally equivalent yet
produce different hardware traces. Two [Claude Code](https://claude.com/claude-code) **skills**
under `.claude/skills/` help make sense of one; ask Claude Code to "triage" or "reproduce" a
`violation-*/` directory, or run the scripts directly.

- **`revizor-violation-triage`** — genuine leak vs noise **without re-running on hardware**. Runs
  the CE under always-mispredict and checks three things: the two inputs' *architectural* cache
  lines are identical, their *speculative* lines differ, and that speculative divergence matches the
  HW htrace divergence already in the report. Verdict: **GENUINE** / **NOISE** / **INVESTIGATE**.
- **`reproduce-revizor-violation`** — confirms it on hardware by recreating the exact
  micro-architectural state: load context inputs `0..min(A,B)`, **swap only the violating input**,
  measure, and see which cache set encodes which input ran. The leak is intermittent, so collect
  statistics over many trials.

These are quick verification heuristics: a GENUINE verdict plus a clean reproduction is strong
corroboration. Note that re-measuring the *same* test many times trains the branch predictor and can
hide a genuine leak, so it is not a way to "confirm noise".

## 9. Memory layouts

### 9.1 Sandbox

A test case may only touch a fixed arena. During the run `x29` holds the **main-region** base;
every memory access is masked to `x29 + (reg & 0x1fff)`, so it always lands in `main`+`faulty`
(8 KB). The cache set of an access is `(offset // 64) % 64`.

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

### 9.2 Input register region

An input seeds memory (`main` + `faulty`) and the registers. The register region is 8 × 64-bit
slots:

| Slot | Register |
|---|---|
| 0–5 | `x0`–`x5` |
| 6 | NZCV flags (per-flag encoding, below) |
| 7 | unused (the executor forces `sp` to the sandbox stack base) |

The flags slot uses a **per-flag NZCV encoding**: each of N, Z, C, V occupies bit 0 of its own byte
within the slot, so the four flags stay independent under taint tracking; the slot is converted to
ARM `PSTATE` form just before execution.

## 10. File formats

### 10.1 Run config (YAML)

A run is configured by a YAML file passed with `-c`. Options not set take their architecture
defaults (`src/aarch64/aarch64_config.py` and the shared `src/config.py`). A representative config:

```yaml
instruction_set: aarch64

# Which instruction families the generator may emit (§11).
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

### 10.2 Instruction-set spec (`base.json`)

`base.json` is the instruction-set spec the generator emits from. It is **not hand-edited** — it is
generated from ARM's machine-readable A64 ISA (§12) — but its shape is worth knowing.

The file is a JSON **list** of instruction nodes. `isa_loader` reads these fields per node:

| Field | Meaning |
|---|---|
| `name` | Mnemonic (e.g. `"add"`). |
| `category` | Coarse ISA category; only `"general"` is emittable (§11). |
| `control_flow` | `true` for branches / control-flow instructions. |
| `template` | Python format string used to render the instruction to assembly text. |
| `tags` | List of taxonomy tags (§11); defaults to `[category]` if absent. |
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

### 10.3 Operand model (Python)

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

## 11. Instruction taxonomy

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

## 12. Running

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
