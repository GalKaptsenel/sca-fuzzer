---
name: revizor-violation-workflow
description: The end-to-end method for turning an AArch64 Revizor violation into a verdict — genuine speculative leak (and which variant), Revizor/modeling bug, or measurement noise. Ties together the three analysis tools (contract triage, mechanistic leak-flow, hardware reproduction) and the contract-checking discipline that decides everything. Use this as the top-level guide whenever a violation-* directory needs to be understood; it points at the specific tool-skills for each step. Covers the contract lattice, the "no fallbacks / check every contract" rule, device transport + cache-geometry portability, and all the pitfalls.
---

# Revizor violation → verdict: the full workflow

A violation = **V counterexample inputs** that the *contract* says are equivalent, yet produce
**divergent hardware cache traces**. The job is to decide which of three it is:
1. **Genuine speculative leak** — a real µarch leak beyond the contract (and *which* speculation).
2. **Revizor / modeling bug** — the inputs weren't actually contract-equivalent, so it should never
   have been flagged (a CE, base, comparison, or generator error).
3. **Measurement noise** — prefetcher/jitter that fails to reproduce.

Three tools, run in this order. Each is its own skill; this file is the glue + the decision logic.

| step | tool-skill | question it answers | needs device? |
|---|---|---|---|
| 1. Triage (contracts) | **revizor-violation-triage** (`triage.py`) | Do the inputs AGREE where they must and DISAGREE where a leak would show? Which contract raises it? | no (CE is local) |
| 2. Pinpoint | **revizor-leak-flow** (`flow_table.py`) | *Which instruction / speculative access / address* leaks, and does it match the HW htrace (vs prefetcher noise)? | no |
| 3. Reproduce | **reproduce-violation-manual** (`verify.py`) | Does the difference PERSIST on hardware in a controlled batch context, or wash out? | **yes** |

Low-level driving of `/dev/executor` by hand: **executor-userland**. Building the kernel module for a
device: **[[wsl-executor-build-recipe]]**. Device transport (adb `:5037`): **[[pixel8-physical-fuzzing]]**.

---

## The contract-checking discipline (this is the core — get it right)

Contracts form a lattice from **strict** (few behaviors allowed) to **permissive** (more speculative
behaviors allowed). Each *execution clause* adds a speculation kind on top of `seq`:

- `seq`  — architectural only (strictest; the "nothing speculative" baseline).
- `cond` — + conditional-branch mis-speculation  →  **Spectre‑v1**.
- `bpas` — + speculative store-bypass            →  **Spectre‑v4**.
- `cond-bpas` — + both. (Others exist: `bpu` indirect/BTB, `barrier`, `sls`, …)

A more-permissive contract's ctrace is a **superset** of a stricter one's (it may touch extra cache
sets under the speculation it allows). "Allowing" / "permissive" = it *permits* the speculative access
that a real leak would make.

### Which contracts to run — ALWAYS, no fallbacks
Run **exactly the contracts the user asks for**; on an unknown name, **error — never silently
substitute** (`triage.py` enforces this; `--contracts all` enumerates every supported clause). When the
user doesn't pin a set, the standard battery is **all three of these groups**:

1. **The contract the fuzzing actually ran** (e.g. ct‑seq → the `seq` clause) **plus every stricter
   ("more basic") contract**. The V inputs **MUST AGREE** here — that is the definition of a violation
   (contract-equivalent yet HW-divergent). Purpose: confirm the inputs are *legitimate candidates*.
   - If they **DISAGREE** under the run/stricter contract → it is **NOT a real leak, it's a Revizor /
     modeling bug** (bad base, CE error, comparison bug, generator bug). The fuzzer flagged inputs that
     were never contract-equivalent. Stop and chase the bug, not the µarch.

2. **At least one MORE-PERMISSIVE contract that is *expected to raise* (explain) the leak.** It should
   **DISAGREE**, and the divergent set(s) should **match the observed HW htrace-diff**. The *least*
   permissive contract that raises it **names the variant**:
   - Spectre‑v1 → `cond` raises it.  Spectre‑v4 → `bpas` raises it.
   - Such a contract **does not always exist.** A genuine, **novel** leak may have **no** modeled
     contract that raises it — every contract AGREES, yet the HW difference reproduces (step 3). That is
     the most interesting outcome, not a dead end: a real leak the contract model doesn't yet describe.

So the canonical set is: `{stricter contracts} ∪ {the run contract} ∪ {≥1 permissive "raiser"}`. Check
them **all**, every time — checking the stricter ones is what distinguishes a **Revizor bug** (they
disagree) from a **genuine candidate** (they agree), and checking the permissive one is what
**identifies the variant** (or flags it as novel when none raises it).

### The verdict table
| run/stricter contracts | a permissive contract | HW reproduces (step 3) | verdict |
|---|---|---|---|
| AGREE | DISAGREE on the HW set | yes | **GENUINE** leak of that variant (cond=v1, bpas=v4, …) |
| AGREE | none disagree / no matching set | **yes** | **GENUINE but NOVEL** — no modeled contract explains it |
| AGREE | (any) | no (washes out) | **NOISE** (measurement / prefetcher false positive) |
| **DISAGREE** | — | — | **REVIZOR / MODELING BUG** — not a real violation; fix the tooling |

---

## Step-by-step

**1. Triage — `triage.py <vd> --base 0xHEX --contracts seq,cond,bpas,cond-bpas`**
Runs the CE under each contract, compares the V inputs, prints AGREE/DISAGREE + divergent sets + whether
they match the parsed HW htrace. **Base is load-bearing** (ct records absolute addresses; a wrong base
fakes a `seq` DISAGREE) — `triage.py` auto-validates the base against the report's ct-seq hash and aborts
if none matches. Apply the discipline above to the output.

**2. Pinpoint — `flow_table.py <vd> --base 0xHEX --contract <the raiser> --input both [--html out.html]`**
For the contract that raised the leak, dumps the per-instruction trace (idx, phase arch/spec,
instruction, R/W, size, VA, set, mem before→after) and a per-set table classified **only from real
accesses** (arch / spec-only / arch+spec / HW-only), with the observed **HW %** per set and the
**UNEXPECTED** (HW-only, prefetcher-noise) sets listed. This is what separates a genuine leak (the
distinguishing set is a **spec-only** access — one instruction, secret-dependent address) from noise
(the distinguishing set is **HW-only**, touched by nothing in the model). Renders terminal and/or HTML.

**3. Reproduce — `verify.py <vd> --repo <repo> --position all --reps 2000 --sets <predicted>`**
Controlled batch-context HW re-measure under the violation's own regime: loads the whole input batch
once, swaps ONLY the tested slot between the V inputs, and reports whether the predicted set's spread
**persists** (GENUINE) or **washes out** (NOISE) — at lower/higher/last positions (a real leak persists
at all). This is the arbiter over the model prediction.

**Two worked outcomes from this repo:**
- fr_violation (F+R): `seq` AGREE, `cond` DISAGREE on set 23, reproduces → **genuine Spectre‑v1**; the
  leak is a speculative store whose address is secret-dependent (set 23 vs set 1 across the two inputs).
- A P+P violation (in#40/#90): `seq` AGREE, `cond` DISAGREE on sets 37/52, HW-reproduced across all
  positions → **genuine Spectre‑v1 found under P+P** (speculative `ldr w4,[x2]`), even though *most* P+P
  violations on this core are noise. Contrast: another P+P violation's distinguishing set was `HW-only`
  → noise. The tools separate them; never assume by regime.

---

## Pitfalls baked into the tools (know them)
- **Real sandbox base** — read `/sys/executor/print_sandbox_base`; wrong base → spurious `seq` DISAGREE.
- **`report.txt` htrace is MSb-first** — string pos p = cache set (63−p); raw ints / CE `(addr>>6)&63`
  are identity. `triage.py` reverses; compare in *set* space.
- **Batch/prefetcher contamination** — an htrace depends on its predecessors *within the same test
  case's* input batch (reset between test cases). Never measure the counterexamples in isolation; use
  the fixed-prefix / swap-one-slot protocol (`verify.py`).
- **PC vs data in the `ct` trace** — the `ct` observation records code addresses *and* data addresses;
  only **data** accesses map to the L1D htrace. `flow_table.py` reads the structured per-instruction
  trace (`speculation_nesting`, `MemAccess`), never the flattened `.raw`, so PC sets never masquerade as
  data sets. Classify a set from a real `MemAccess` only.
- **`instr_index` is an execution counter, not a static index** — map to a basic block via the PC.
- **P+P prime trains the BPU** ([[pp-prime-bpu-training]]) — the kernel P+P prime's ~thousands of taken
  branches suppress many mispredictions, so P+P misses leaks F+R finds and floods the FP-filter with
  noise. It does *not* suppress all of them (see the genuine P+P leak above). Regime ≠ verdict.
- **`Test Case ID` fix** — in windowed super-batch runs the report used to print the window's last index
  (K−1) for every violation; fixed to stamp `test_case.test_id` at generation. The unique id is the
  **program seed**.

---

## Cache-geometry portability (READ before running on a new device, e.g. the Google server)
Every layer here assumes the **module's compiled L1D geometry matches the target core**:
`_cache_sets` uses `(ea−base)//64 % 64`; the P+P prime/probe fill `L1D_SIZE`/`L1D_ASSOCIATIVITY`; the
htrace folds to the set count. The module bakes these at build time (`-DL1D_SIZE_K`,
`-DL1D_ASSOCIATIVITY`). Wrong geometry → empty P+P (prime too small to evict — see the X3 32KB→64KB
episode in [[x3-ctseq-triage]]) or mis-indexed sets.

Porting to a new machine:
1. **Probe the target's real geometry** — `docs/a510_prefetcher/cross_el_poc/ccsidr_probe.c` reads
   `CCSIDR_EL1` per core (line size, ways, sets; CCIDX-aware) and L2.
2. **Rebuild the module for that geometry** — see [[wsl-executor-build-recipe]]; set `L1D_SIZE_K` /
   `L1D_ASSOCIATIVITY` to the probed values, and match the target kernel's vermagic/toolchain.
3. **Re-read the sandbox base on that device** and pass it to `triage.py`/`verify.py`.
4. Confirm the channel first (every CE architectural set shows hot at ~1.0 in the HW htrace) before
   trusting any verdict.

**Planned improvement (do this to remove the footgun):** make the **kernel measurement-code generation
self-configure** — read `CCSIDR_EL1`/`CTR_EL0` at module load, derive L1D size / associativity /
n_sets / line size, and size the prime, probe, eviction region, and htrace fold from those at runtime
instead of the compile-time `-D` defines. Then one module binary is correct on any core, and the
32KB↔64KB class of bug disappears.

## Cross-references
- Tools: **revizor-violation-triage**, **revizor-leak-flow**, **reproduce-violation-manual**, **executor-userland**.
- Build/transport: [[wsl-executor-build-recipe]], [[pixel8-physical-fuzzing]].
- Findings/context: [[x3-ctseq-triage]], [[pp-prime-bpu-training]], [[prefetch-batch-contamination]].
