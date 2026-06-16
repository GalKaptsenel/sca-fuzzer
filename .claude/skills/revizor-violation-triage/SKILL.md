---
name: revizor-violation-triage
description: Triage an AArch64 Revizor violation to decide GENUINE Spectre-style leak vs measurement noise, by comparing the contract executor's speculative (wrong-path) cache lines against the hardware htrace — WITHOUT re-running the test on HW (which trains the BPU and hides the leak). Use when a violation-* dir or a "Mismatching CTraces"/violation report needs classification.
---

# Revizor violation triage (genuine Spectre-v1 vs noise)

## Core principle
A Revizor violation = two inputs that are **architecturally equivalent** (same contract trace under the run's contract) but have **different hardware traces**. To confirm it is a genuine speculative leak (not noise):

1. Confirm the inputs are **arch-equivalent** (identical nest=0 cache lines in the CE).
2. Confirm their **speculative (nest>0) cache lines DIFFER** in the CE (run the CE under `ALWAYS_MISPREDICT`).
3. Confirm the **CE's speculative divergence explains the observed HW htrace divergence** (bit-for-bit).

If all three hold → **genuine Spectre-v1**. If the HW shows no stable divergence, or the CE shows no speculative divergence → **noise**.

> **v4 (store-bypass) triage:** the same three-step logic applies, but the speculative divergence is
> not branch-driven — run the CE under the **`bpas`** contract (`contract_execution_clause=['bpas']`,
> `ExecutionClause.BPAS`) instead of `cond`, and confirm the divergence vanishes with `enable_speculative_store_bypass=0`.
> See **`reproduce-spectre-v4`** for the full v4 argument and **`reproduce-violation-manual`** /
> **`executor-userland`** for replaying the pair on hardware.

## CRITICAL anti-pattern (do NOT do this)
**Do NOT "confirm noise" by re-running the same test on HW many times (high reps/warmups).** Repetition *trains the branch predictor* to predict the branch correctly, which **removes the misprediction that causes the leak** — so a genuine Spectre-v1 will look like clean/identical htraces after training. Repeated-measurement stability is the WRONG tool here. Use the CE's speculative trace + the violation report's *already-captured* htrace distribution instead.

## Method

### 1. Get the counterexample pair
From `violation-*/report.txt`, the `## Counterexample Inputs` section lists `Input #A` and `Input #B` (indices into the boosted input set). The saved `input_AAAA_nzcv_scheme.bin` files (older runs: `input_AAAA.bin`) are those boosted inputs.

### 2. CE arch vs speculative cache lines
Run the CE under **ALWAYS_MISPREDICT** (`CONF.contract_execution_clause=['cond']`) on the pair. For each input, split memory accesses by `speculation_nesting`:
- `nest==0` → architectural lines; cache line = `(EA - x29) // 64 % 64`.
- `nest>0`  → speculative (wrong-path) lines.
Genuine leak signature: **arch lines identical, speculative lines differ** (a data-dependent wrong-path access).

### 3. Map CE lines → HW htrace bits
- `K = (sandbox_base // 64) % 64` (read via `executor.read_base_addresses()`; usually 0 since the sandbox is 64-aligned at `…0000`).
- HW set index = `(line + K) % 64`.
- **Two different representations — do not confuse them (this caused a wrong "MISSING" verdict once):**
  - The **raw htrace integer** (`HTrace.raw`, list of 64-bit ints from `trace_test_case`): **bit b == cache set b directly (IDENTITY map).** Expected raw bit for a CE line `l` is just `(l + K) % 64`.
  - The **printed htrace string** in `report.txt` (the `^.^…` patterns): **bit-REVERSED**, string position = `63 - HW_set`.
  - So: working from `HTrace.raw` → identity; working from `report.txt` strings (as `triage_violation.py` does) → `63 - set`. Both are self-consistent; just stay in one space.

### 4. Per-bit HW divergence from the report (NOT re-measurement)
Parse the **full** htrace distribution per input from `report.txt` (every `<pattern> [count]` line, not just the top one), compute per-bit set-frequency, and find bits where `|freq_A - freq_B| > 0.3` (stably divergent). Using only the single dominant pattern is too coarse and gives false "doesn't fit" results.

### 5. Verdict
- HW-divergent bits **⊆ CE speculative-divergent bits** (and non-empty) → **GENUINE**.
- HW-divergent bits with **no stable divergence** → **noise**.
- HW-divergent bit **not in CE set** → **investigate**: deeper speculative nesting than explored, or a CE modeling gap (e.g., the known STP-2nd-element / pair / signed-load gaps).

## Tool
`scripts/triage_violation.py <violation-dir>` — runs steps 1–5 and prints the per-bit verdict. It loads the TC + the counterexample inputs, runs the CE under ALWAYS_MISPREDICT, maps lines→bits, parses the report's per-bit frequencies, and classifies.

## Manual CE emulation (conscious cross-validation of a violation)
When you need to *prove* a violation by hand (not just trust the script), step through the CE's own trace and re-derive every address yourself. Authoritative per-instruction fields on each `ContractExecutionResult` entry `ite`:
- `ite.cpu.gpr[0..30]` (x30 = lr), `ite.cpu.sp`, `ite.cpu.nzcv`, `ite.cpu.pc`, `ite.cpu.encoding` (raw 32-bit — feed to `aarch64_disasm.disassemble_instruction(encoding, pc)` to get the exact instr, including immediates that confirm the source asm).
- `ite.metadata.has_memory_access`, `.speculation_nesting` (0 = architectural, >0 = wrong-path), `.instr_index`, `.memory_access` with `.effective_address`, `.is_write`, `.element_size`, and crucially **`.before` / `.after`** = the memory value loaded/stored (this is how you read the "secret" a gadget leaks).
- Sandbox masking: every memory base register is rewritten to `addr = x29 + (reg & 0x1fff)` (main 0x000–0xFFF, faulty 0x1000–0x1FFF; the original immediate/index is neutralized); hand-check `set == ((EA - x29) // 64) % 64`.

### NZCV initialization methodology (DO NOT skip when emulating)
Flags are NOT stored as a PSTATE word in the input — they use a **per-flag byte encoding** in GPR slot 6 (`NZCVScheme` in `aarch64_input_layout.py`):
- Register region starts at input byte `0x2000`; slot i = x{i} for i<6, **slot 6 (byte 0x2030) = NZCV**, slot 7 = sp. Each slot is 8 bytes; the low 32 bits are mirrored into the high 32 (Revizor duplication).
- Within slot 6: **bit 0 of byte 0 = N, byte 1 = Z, byte 2 = C, byte 3 = V** (bytes 4–7 mirror 0–3). So slot `01 01 01 01` ⇒ N=Z=C=V=1; `00 01 01 00` ⇒ N=0,Z=1,C=1,V=0.
- Just before execution `_reconstruct_pstate` converts that slot to ARM PSTATE: `pstate = N<<31 | Z<<30 | C<<29 | V<<28`, and `set_registers_from_input` does `msr nzcv, x6`. To emulate by hand you MUST do this reconstruction — read the four bytes, build PSTATE, then evaluate conditions (`HI = C&!Z`, `PL = !N`, `NE = !Z`, …).
- Per-flag bytes give byte-granular taint separation (N→byte48, Z→49, C→50, V→51). The prime/probe harness must never clobber NZCV between init and the TC (use CBNZ/CBZ + SUB/ADD, never `subs/cmp/b.cond`); see memory `feedback_harness_no_flag_clobber`. A wrong NZCV init flips conditional-select/branch outcomes and silently changes the traced path.

Procedure: dump **both** paths for **both** counterexample inputs.
- **Architectural (nest 0)**: the set sequence must be **identical** between the two inputs (store *values* may differ — that's fine and expected, it proves the inputs really differ in memory yet share the arch cache footprint). If arch sets differ, the inputs are NOT contract-equivalent → not a real violation.
- **Speculative (nest>0)**: find the gadget — a load whose *address* comes from a previously-loaded value. The first load reads the SAME address in both inputs (the secret location) but `.before` differs; the dependent load's `(secret & 0x1fff)` becomes its address → different cache set per input. Those sets must equal the HW-divergent bits.
`scripts/manual_emulate.py <violation-dir>` dumps exactly this (disasm + set + loaded value for every mem access, both paths, with a per-line `hand==CE` check).

## Validate the P+P channel itself BEFORE trusting any violation
A noisy prime produces garbage htraces and false violations. Verify the channel with `scripts/check_pp_correctness.py` (loads a TC, N=200 reps, arch-only `seq`):
- **No false negatives**: every CE architectural line must appear as a hot htrace bit with frequency ~1.0.
- **Noise metrics**: spurious hot bits (set ≥50% but not architectural), extra evictions/rep, dominant-pattern stability, distinct-patterns/N. A clean channel ≈ 0 spurious, ~1 distinct pattern, stability ≈ 1.0.

## PITFALL — the P+P prime trains the BPU and HIDES Spectre-v1
The kernel `prime()` in `templates_jit.c` runs many store-loop iterations with conditional branches; that **trains the BPU to predict correctly and suppresses the very misprediction the leak needs** → P+P reports 0 Spectre-v1 even when F+R finds them. Confirmed: original prime (32 passes × 3 nested branch sites, 16384 stores) → **0** in P+P; a lean **single-loop** prime found a GENUINE Spectre-v1 in P+P (see memory `project_pp_prime_bpu_training`).
- Build-time knobs (added to `templates_jit.c`): `-DPRIME_SINGLE_LOOP=1` (one linear pass per rep, 1 inner branch) vs `=0` (original); `-DPRIME_REPS=N`. Build: `make KDIR=… KCFLAGS="-DPRIME_SINGLE_LOOP=1 -DPRIME_REPS=8"`.
- **Cleanliness depends on re-priming PASSES, not loop structure**: knee at **4 passes** (single-loop ×4 == original ×32 in cleanliness). 1 pass is far too noisy.
- F+R (uses `flush`, not `prime`) does not have this problem — it's the reliable Spectre-v1 detector.

## PITFALL — non-deterministic test-case generation (confounds A/B runs)
`Generator.update_seed()` reads `self._state` (set from `program_generator_seed` **at construction**), NOT a later `CONF.program_generator_seed` assignment, and `_state==0` → a *random* seed each run. For a reproducible TC in a script, set `gen._state = SEED` directly **before** `create_test_case(...)`. `create_test_case(path)` takes an asm-file PATH, not a seed.

## Gotchas / environment
- Activate your Revizor Python venv (the one created by the installer); `chmod 777 /dev/executor` after insmod.
- The CE binary must be rebuilt after any `contract_executor/*.c|*.h` change (`make -C src/aarch64/contract_executor`; the Makefile now tracks header deps). Kernel module: `make KDIR=/usr/src/linux-headers-6.12.90+deb13-cloud-arm64`.
- Module reload can leave `refcnt` stuck if a `contract_executor` subprocess was SIGKILLed while holding `/dev/executor` (orphaned holder) → `rmmod` fails, only a reboot clears it. Kill CE subprocesses cleanly before `rmmod`.
- `map_register_to_offsets` must map `wN` to the same slot as `xN` (taint completeness — a w-reg gap there caused spurious "Mismatching CTraces"). See memory `project_ce_audit_findings`.
- Known CE speculative-modeling gaps that can leave HW bits "unexplained": pair load/store 2nd element not traced, SUBPS flags, SIMD/FP not modeled (see memory `project_doc_what_we_dont_do`).
