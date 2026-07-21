---
name: revizor-leak-flow
description: Build a mechanistic, instruction-by-instruction execution-flow table for an AArch64 Revizor violation by running the contract executor (CE) and reading its STRUCTURED per-instruction trace. For each executed instruction it shows the disassembly, basic block, speculation phase (architectural vs speculative), and for every memory access the actual effective address, L1D cache set, load/store, pair elements, and memory contents before→after. It classifies cache sets STRICTLY from real accesses (never guessed) and pins the leaking set + instruction against the hardware htrace. Use AFTER revizor-violation-triage confirms a violation, when you need to SEE why it leaks (which instruction, which speculative access, what content) rather than just AGREE/DISAGREE. REIF + legacy bin; CE runs locally.
---

# Mechanistic leak-flow table for a Revizor violation

`revizor-violation-triage` tells you a violation is a genuine Spectre and which contract disagrees.
This skill shows you **exactly how it leaks**: the instruction stream, the speculative access, the
address, and the memory content — as a table you can read and trust.

## Tool
`flow_table.py <violation-dir> [--base 0xHEX] [--contract seq|cond|bpas|cond-bpas] [--pair A B] [--input A|B|both] [--nest 5] [--hw-min 0.05] [--html PATH] [--no-terminal]`

Renders the same two tables per input as **terminal** text (default) and/or a self-contained **HTML**
page (`--html out.html`, theme-aware, leak row highlighted; add `--no-terminal` for HTML only). The HTML
is publishable directly as an artifact.

For each counterexample input it prints **two tables** (so `--input both` yields two per input):

**TABLE 1 — the trace** (one row per executed instruction, an LDP/STP pair = two rows):
`idx` (trace execution index) · `ph` (arch / spec, straight from `speculation_nesting`) · `instruction`
(disassembled from the trace encoding) · `rw` (R/W/RMW) · `sz` (access byte size = `element_size`) ·
`VA` (full 64-bit effective address) · `set` (L1D data set) · `mem_before` · `mem_after` (**verbatim
from the CE `MemAccess.before`/`.after` — never computed**; a load's before==after). Non-memory
instructions show only idx/phase/instruction. The leaking speculative access is tagged `<== LEAK`.

**TABLE 2 — per set** (one row per set, **model trace vs the OBSERVED HW htrace for that input**): `set`
· `class` (`arch` / `spec-only` / `arch+spec` / `HW-only`) · `HW%` (this input's htrace frequency for
that set, from report.txt) · `leak` · the instruction `idx[phase]`(s) that accessed it · a note. A set
touched in both phases is `arch+spec` (NOT a leak); `spec-only ∩ HW-htrace-diff` is THE LEAK. Sets that
are hot in HW (≥ `--hw-min`, default 5%) but the model never accessed are listed as **`HW-only`** —
the prefetcher/noise floor F+R rides on — so nothing observed is hidden. Everything is derived from
Table 1 + the report htrace; nothing hand-labelled. Typical read: arch sets ≈100% HW for both inputs;
the leak set is spec-only and high-HW% for one input only; a spec-only set that is high-HW% for *both*
(e.g. set 63) is correctly not in the htrace-diff. Each per-set table ends with an **UNEXPECTED** line
listing every set hot in that input's HW htrace that the model never accessed (neither arch nor spec) —
the prefetcher / noise floor that F+R rides on, called out explicitly so nothing observed is unaccounted.

## Why it reads the STRUCTURED trace, not compute_ctrace().raw (the whole point)
`compute_ctrace().raw` flattens everything and, under the `ct` clause, **interleaves code-fetch PC
addresses with data addresses** — so `(v>>6)&63` over `.raw` yields phantom "sets" that are really code
addresses, and set-differencing two `.raw`s can BOTH invent sets and hide real ones. This tool instead
iterates `ContractExecutionResult` (`aarch64_trace.py`): each `InstrTraceEntry` gives
`cpu.pc/encoding/gpr`, `metadata.speculation_nesting`, `is_pair`, and `metadata.accesses()` →
`MemAccess{effective_address, before, after, element_size, is_write, is_atomic}`. Cache set =
`_cache_sets(ma, sandbox_base)` on **data accesses only** (identical to the model's `l1d` observation,
which is what the HW F+R trace corresponds to).

## The rigor rule (do not violate)
A cache set is attributed **only** from a real `MemAccess`, and its phase **only** from
`speculation_nesting`:
- **architectural** iff some `speculation_nesting == 0` access touches it;
- **speculative-only** iff some `nesting > 0` access touches it AND no arch access does.
Never assign a set to an instruction by eye. Never count a PC/code address as a data set. What a
contract does not actually access, that contract does not predict — so it is not classified under it.
(The same discipline extends to bpas/cond-bpas: run that contract and read its `nesting>0` accesses.)

## Pitfalls baked in
- **Base is load-bearing.** `_cache_sets` is relative to `sandbox_base`; a wrong base shifts every set.
  The tool auto-validates the base against the report's ct-seq hash and aborts if none matches (pass
  `--base` from `/sys/executor/print_sandbox_base`).
- **`instr_index` is an execution counter, not a static index.** A statically-single instruction that
  runs both speculatively and architecturally appears twice with different `instr_index`. Map to a
  basic block via the **PC** (`(pc - code_start)//4` into the static asm), never via `instr_index`.
- **The leaking instruction can be a STORE, and can live past the mispredicted branch.** The
  mispredicted direction speculatively runs on into later blocks; a store there whose address derives
  from the speculative path is the channel. Its architectural twin (same static PC) may hit a totally
  different set — that's expected, and is why arch vs spec must be split per access.
- **Both htrace-diff sets can be the same instruction.** Two counterexample inputs often drive the
  same speculative store to two different sets (e.g. set 23 vs set 1); both show up in the HW diff and
  both are "the leak" — one per input. Don't dismiss the second as an artifact.

## Worked example (docs/x3_ct_seq_triage/fr_violation, X3 ct-seq)
`stp w0, w0, [x3], #0x58` (idx 45, `speculation_nesting=1`, in the speculatively-entered `.bb_0.2`)
writes to a **secret-dependent address**: set 23 for in#39 (`ea 0x…25f1`), set 1 for in#89
(`ea 0x…1055`). Architectural footprints are identical `{2,3,9,13,16,17,25,40,45,46,52,54,58}`; the HW
F+R diff `{1,23}` is exactly the two speculative-store targets (in#39 set23 79% vs 0.3%, in#89 set1 38%
vs 1%). Textbook Spectre-v1; the leaked quantity is which cache set the speculative store addresses.

## Cross-references
- **revizor-violation-triage** — run FIRST (AGREE/DISAGREE classification + which contract disagrees).
- **reproduce-violation-manual** — HW confirmation (verify.py, controlled batch-context re-measure).
- Trace internals: `src/aarch64/aarch64_trace.py` (`InstrTraceEntry`, `MemAccess`, `_cache_sets`,
  `_ct_l1d` = the data-only bitmap the HW L1D htrace matches).
