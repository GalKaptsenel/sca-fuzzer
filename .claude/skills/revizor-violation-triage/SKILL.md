---
name: revizor-violation-triage
description: Triage an AArch64 Revizor violation to classify it as a GENUINE Spectre leak (and which variant) vs measurement noise, by running the contract executor (CE) locally under a configurable set of speculation contracts and comparing its speculative cache sets to the hardware htrace. Works for REIF and legacy bin inputs, and for both local and remote HW runs (the CE always runs locally). Use when a violation-* dir / report needs classification, BEFORE any hardware re-measurement.
---

# Revizor violation triage (contract-level classification)

A violation = two (or more) inputs that are **contract-equivalent** (identical contract trace under the
run's contract) yet have **different hardware traces**. Triage decides whether that HW difference is a
real speculative leak — and which Spectre variant — using only the **CE (local userland subprocess)**,
without re-running on hardware. Classify here first; confirm on HW with **revizor-violation-verify**.

## The three checks
1. **seq AGREE** — under the sequential (non-speculative) contract the inputs produce the *same* trace
   (the violation premise; if it fails they were never contract-equivalent — or your base is wrong).
2. **a speculative clause DISAGREES** — under some `contract_execution_clause` their traces diverge.
   *Which* clause first splits them names the variant.
3. **the CE's speculative-divergent sets match the observed htrace difference**.

seq✓ + one-clause✓ + sets-match✓ → **genuine** (that variant). Checks 1–2 pass but sets don't match →
likely noise → escalate to HW verification.

## ⚠️ CRITICAL: use the REAL sandbox base
`ct` records **absolute addresses** (`base + offset`); sandbox masking makes address collisions
**base-dependent**. A wrong/placeholder base gives *spurious seq DISAGREE*. Always pass the run's real
base and sanity-check that the CE seq hash equals `report.txt`'s `Contract trace (hash)`:
```python
base = int(open("/sys/executor/print_sandbox_base").read(), 16)   # remote: read over the connection
# or, from a live executor (local or remote): base, _ = executor.read_base_addresses()
```
(The line→bin mapping is base-independent since the base is page-aligned; the hash/agreement is not.)

## General contract set (configurable — not just seq/cond)
The CE accepts any `ExecutionClause` combination in `SUPPORTED_EXECUTION_CLAUSES`
(`aarch64_contract_executor.py`). Triage a **list** so the skill generalizes to any leak type:
```python
from src.aarch64.aarch64_contract_executor import ExecutionClause as E
CONTRACTS = [("seq", E.SEQ, 0), ("cond", E.COND, 5), ("bpas", E.BPAS, 5),
             ("cond-bpas", E.COND|E.BPAS, 5), ("bpu", E.BPU, 5), ("barrier", E.BARRIER, 5),
             ("sls", E.SLS, 5)]        # add/remove to match the campaign's contract + hypotheses
```
`seq` runs at nest 0; speculative clauses at nest ≥ 5 (some leaks are several speculated instructions
deep — sweep nest 1/5/30; nest 1 alone often misses it).

## Method (REIF inputs, current API)
```python
import sys; sys.path.insert(0, "<repo>")
from src.config import CONF; CONF.load(f"{VD}/reproduce.yaml")
from src.factory import get_input_generator
from src.aarch64.aarch64_generator import Aarch64Generator
from src.aarch64.aarch64_contract_executor import ContractExecution, ContractExecutorService, ExecutionClause, SimArch
from src.aarch64.aarch64_trace import compute_ctrace
from src.aarch64.aarch64_executor import _ce_memory_regs

tc_bytes = Aarch64Generator.in_memory_assemble(open(f"{VD}/sandboxed_test_case.asm").read())
inA, inB = get_input_generator(0).load([f"{VD}/input_{A:04d}.reif", f"{VD}/input_{B:04d}.reif"])
ce = ContractExecutorService("<repo>/src/aarch64/contract_executor/contract_executor")
def ctrace(inp, clause, nest, base):
    mem, regs = _ce_memory_regs(inp)
    ex = ContractExecution(tc_bytes, mem, regs, SimArch.RVZR_ARCH_AARCH64, nest,
                           CONF.model_max_spec_window, req_mem_base_virt=base, execution_clauses=clause)
    return compute_ctrace(ce.run(ex))         # .hash_ = agreement key; .raw = addresses
```
- Counterexample `A`,`B`: `report.txt` → `## Counterexample Inputs`. Files `input_{A:04d}.reif`.
- Divergent sets for a disagreeing clause: `{(v>>6)&63 for v in ctrace(...).raw}`, symmetric difference.
- Observed htrace diff: parse **every** `<pattern> [count]` line per input in `report.txt`, per-bit
  frequency, bits with large `|freq_A − freq_B|` (never just the dominant pattern).

**Tool:** `scripts/ce_full.py <violation-dir> <A> <B>` — runs the CONTRACTS list, prints AGREE/DISAGREE +
divergent sets, flags which match the htrace diff. Pass the real base.

## Local vs remote
The CE is a **local** subprocess in both cases — triage code is identical. Only the *base* differs in
where you read it (local `/sys/executor` vs over the SSH/ADB connection). Do NOT build a remote
HWExecutor just to triage — that reconfigures the device (pin/mode) and can corrupt a running campaign;
read the base once, or reuse the value in `report.txt`/logs.

## Verdict
| result | meaning |
|---|---|
| seq AGREE, cond DISAGREE, bpas AGREE, sets match htrace | genuine **Spectre-v1** (branch) |
| seq AGREE, bpas DISAGREE, cond AGREE, sets match htrace | genuine **Spectre-v4**; confirm it vanishes with `enable_speculative_store_bypass=0` |
| seq AGREE, only cond-bpas / bpu / sls DISAGREE | that combined/other speculation is required |
| seq AGREE, a clause DISAGREES, sets ≠ htrace-diff | contract says leakable, HW shows other sets → **likely noise** → verify |
| seq DISAGREE | not contract-equivalent — **recheck the base first** |

## Do NOT confirm noise by naive high-rep single-input re-measurement
Re-running the *same* input alone many times trains the BPU to predict correctly and **hides**
Spectre-v1. The contract check here is training-free. For HW, use the controlled-batch protocol in
**revizor-violation-verify** (fixed preceding context, swap only the leaking slot), never isolated repetition.

## Manual CE emulation (hand cross-validation)
Each `ContractExecutionResult` entry `ite`: `ite.cpu.gpr[0..30]`/`sp`/`nzcv`/`pc`/`encoding` (disasm via
`aarch64_disasm.disassemble_instruction`), `ite.metadata.speculation_nesting` (0=arch, >0=wrong-path),
`.memory_access.effective_address`/`.is_write`/`.element_size`/`.before`/`.after` (loaded value = the
"secret"). Sandbox masking `EA = x29 + (reg & 0x1fff)`, set `= ((EA−x29)//64)%64`. Arch (nest 0) sets
identical between the pair; the speculative dependent-load set is the divergence and must equal the
HW-divergent sets.

## Legacy bin format
`input_NNNN.bin` uses a per-flag NZCV slot (byte 0x2030) reconstructed to PSTATE before use; contract
logic is identical. Prefer REIF + `input_gen.load`.
