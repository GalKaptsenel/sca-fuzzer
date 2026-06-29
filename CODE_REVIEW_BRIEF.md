# Code-Review & Refactor Brief — AArch64 PAC/MTE/NI sealing subsystem

> Standalone catch-up for a FRESH session. Goal: a professional, top-to-bottom code review +
> refactor of the AArch64 sealing/PAC/MTE/non-interference code, ending in clean, well-named,
> well-separated, well-tested, self-documenting code with correctness preserved. Branch:
> `review/engine-fixes`. Run with maximum thoroughness (/effort max).

## How to work this brief
1. Read this file, then `MEMORY.md` and the memories it points to (especially the safety + style ones).
2. **Review first, refactor second.** Produce a written, file-by-file review (findings + severity)
   BEFORE changing code. Get the findings list agreed, then execute in small verified steps.
3. After every change: run the suite and keep it green. Commit only when the user asks.

## Effort & token strategy (a starting push, NOT a cap)
Quality of the review is the goal; tokens are not the constraint. Spend what the work needs.
- **Default to thorough.** This is a grading-ready review across 8 dimensions × ~12 files — err
  toward depth. Never trade correctness, completeness, or coverage to save tokens.
- **Be smart about *where* the effort goes, not *how much*.** Read in parallel; batch independent
  searches; don't re-read files already in context; don't narrate options you won't take. That
  efficiency frees budget for the hard parts (encodings, contract soundness, algorithmic edge cases)
  — it doesn't mean doing less of them.
- **Scale up freely** when a file or finding warrants it: deeper reads, cross-checking the C/kernel
  side, extra regression tests, adversarial verification of any "this is correct" claim. If in
  doubt, go deeper.
- A multi-agent **workflow** is a good fit here (parallel per-file/per-dimension review +
  adversarial verification). It's the user's explicit opt-in ("use a workflow" / "ultracode") — when
  used, lean comprehensive (larger finder pool, verify findings before recording them).

## Non-negotiable safety constraints (this VM crashes / reboots)
- Python: `/home/gal_k_1_1998/revizor/revizor-venv/bin/python` (system python lacks xxhash/numpy).
- After any module reload: `sudo chmod -R a+rwX /sys/executor; sudo chmod a+rw /dev/executor`.
- The VM reboots spontaneously; `/tmp` is wiped, `/home` survives. Keep durable artifacts in `/home`.
- **NEVER** run a forged/unproven `AUT*` through CE/HW at EL1 — a failed auth FPAC-faults and hard-
  resets the box. The kernel AUTH net (`pac_auth_with_keys`, verify-then-auth) must stay; keep
  `CE_DEBUG_PAC_AUTH_NO_FAULT = 0`. Do not weaken either.
- Fix bad scenarios **in the generator**, never mask downstream in the seal. (User rejected seal-side
  strips for self-context auths; see [[fix-at-source-not-downstream]].)
- A wire-format change must rebuild+reload BOTH the kernel module (insmod) and the CE binary, else a
  stale parser produces garbage regs → genuine AUT* FPAC reset. See [[input-format-rebuild-both-sides]].
- HW measurement is UNAVAILABLE on this box: PMU has <4 counters,
  `/sys/executor/system/measurement_supported` = 0, so `REVISOR_TRACE` ioctl returns ENODEV (errno
  19). You CANNOT run the HW fuzzer here. Verify behaviour via the unit suite + CE traces +
  generated-assembly inspection only.

## Code-style requirements (user-stated, override defaults)
- **Strip comments aggressively.** We wrote far too many. Keep only a few, trimmed and succinct.
  Self-documenting code (names + structure) is the priority. Future devs will NOT use LLMs.
- Pythonic, OOP, extensible, intuitive. Good naming everywhere. Coherent file/class separation.
- No default function arguments — pass explicitly. ([[code-style-no-default-args]])
- No fallbacks ever; no defaults unless genuinely reasonable/intuitive. Everything explicit; no
  hidden assumptions.
- "from" sets over "exclude" sets; configurable data, not hardcoded ranges.
- Yoda conditions are C-only; Python uses `var == const`. ([[yoda-conditions-c-only]])
- Dependency injection for testability (trace_fn is already injected — extend that discipline).
- Reduce coupling between the common `src/*` files and the `src/aarch64/*` files.

## Review dimensions (cover ALL — this is the grading rubric)
1. **Low-level correctness**: instruction encodings, shift/LSL amounts, immediate field widths,
   system registers / MSRs, alignment (16B MTE granule), TBI/tag-byte handling, sign/zero extension,
   PAC field bits [63:48], MTE tag bits [59:56]. We have HAD bugs here (wrong shifts, wrong sysregs,
   missing kernel MSR config, encoding mistakes).
2. **Emulation / contract fidelity**: does the model match the kernel + CE + hardware? Any
   inconsistency between the three (initial MTE tag, PAC keys, speculation nesting semantics)?
   Adherence to Revizor's goal: sound contract *simulation* and contract *trace* generation.
3. **Parsing**: CE blob parsing (`ContractExecutionResult`), input wire (de)serialization, ELF —
   correct, complete, no silent truncation.
4. **Algorithmic soundness & completeness**: the sandbox-taint dataflow, the seal/resolve staging
   (MTE→PAC two-trace ordering), decoy subset selection, tag-state speculation stack. Are they SOUND
   (no false genuine/decoy) and COMPLETE (no missed sites/corner cases)?
5. **Security robustness**: can any path send a forged AUT* to HW? Any unclamped memory access? Any
   way a decoy perturbs an architectural (non-speculative) slot? `_assert_no_arch_forgery`-class
   invariants must hold.
6. **Structure / naming / coupling**: file separation, class placement, "helper" classes that are
   badly named or misplaced, common↔aarch64 coupling.
7. **Testing**: DI, corner cases, completeness. Remove unneeded/duplicative tests; add missing
   coverage (esp. encodings, speculation unwind, decoy-subset invariants, wire round-trip edge cases).
8. **Dead code / cleanup**: unused code, files, tests, docs, imports, helper files.

## File map — sealing subsystem (this session's main work; primary review target)
- `src/aarch64/aarch64_seal.py` (176) — shared primitives: slot helpers (`make_nop`, `fill_slot_at`,
  `index_instructions`, `inst_at`), `_SANDBOX_MASK`, `_SandboxInstrumentationBase` (taint/address
  helpers + offset-cancelling SUBs + topo sort). MTE_TAG_STORE/LOAD name sets live here.
- `src/aarch64/aarch64_sealer.py` (565) — the live engine: `Sealing` (Sandbox/Pac/Mte), `_Resolved`,
  `ResolvedSealingTestCase` (genuine/decoy), resolvers (`_PacResolver`/`_MteResolver`),
  `SealedTestCase` subclasses (Pac/Mte/MtePac), `Sealer`+`SandboxWalk`, the three concrete sealers,
  `make_sealer`.
- `src/aarch64/aarch64_pac.py` (139) — `PacSign` (slot encoders), `PacSigner` (kernel SIGN only),
  `AuthInstructionSpec` (ctx≠ptr guarantee), `build_pac_specs`, `PACKey`, maps.
- `src/aarch64/aarch64_mte.py` (67) — `MteTagState` (speculation-layered tag map), `MTE_INITIAL_TAG=6`,
  `mte_tag_store_effect`, `MTE_GRANULE`.
- `src/aarch64/aarch64_generator.py` (436) — `_patch_auth_context_collision` (forces ctx≠ptr; the
  GENERATOR-side fix for self-context FPAC), MTE generation gating.
- `src/aarch64/aarch64_executor.py` (950) — `Aarch64RegularSealedExecutor` +
  `Aarch64NonInterferenceExecutor`; wires `make_sealer`, PAC keys, the CE trace_fn, decoy policy.

### Consolidation questions to decide in review
- **4 files for one idea** (seal + sealer + pac + mte). Propose a coherent grouping — likely a
  `seal/` subpackage: `seal/primitives.py`, `seal/sealings.py`, `seal/resolvers.py`,
  `seal/sealers.py`, `seal/pac.py`, `seal/mte.py`. Decide and justify; don't split for its own sake.
- **`aarch64_seal.py` vs `aarch64_sealer.py` names** are confusingly close. Rename for intent.
- **`input_*` trio** are 3 concerns: `input_generator` (NZCV randomization), `input_layout`
  (register-file layout/`NZCVScheme`), `input_wire` (executor-ABI serializer). Regroup/rename:
  e.g. an `input/` subpackage, and `input_wire` → something like `executor_input_abi`.
- **`aarch64_trace.py`** is misnamed: it's the CE-result PARSER + contract-trace derivation (core),
  not "analysis." Split the debug-only `show_context` out; rename the core (e.g.
  `ce_result.py` / `contract_trace.py`). `ContractExecutionResult`, `compute_taint`, `compute_ctrace`
  are LIVE (used by `fuzzer.py`, `aarch64_contract_executor.py`, executor) — keep them.
- **`aarch64_log.py`** is pure observability, executor-only, inline call sites. User wants it gone
  ("I don't need logs for now, maybe only for debugging"). Decide: delete (and strip the executor
  call sites) vs. quarantine behind a single debug toggle. Removing it also de-clutters the executor.
- **`aarch64_elf_parser.py`** — live but only for the minimizer; predates this work. Likely
  out-of-scope; confirm it isn't dead and leave alone unless coupling demands otherwise.

## Known-suspect areas to scrutinize first (we had real bugs here)
- Self-context `AUT* Xn,Xn` is unsatisfiable once sealed (MOVK contaminates the auth context). Fixed
  in the generator (`_patch_auth_context_collision`). Verify the fix is COMPLETE (all auth forms,
  memory-base auths via `_pick_mem_auth`, and the standalone `_seal_auths` path) and that nothing
  re-introduces ptr==ctx after patching. See [[pac-mte-context-tag-mismatch]].
- Zero-context auth case (ctx==0) — a separate class; confirm handled.
- MTE memory-tag family (LDG/STG) shares opcode bit 22 with loads → a base_hook kaddr↔uaddr bug
  zeroed context. Confirm the CE `mte_is_mem_tag_access()` guard is correct and complete.
- MTE→PAC two-trace staging in `MtePacSealedTestCase.resolve`: PAC must sign over a trace where the
  pointer/context already carries its genuine MTE tag (else EL1 FPAC). Verify soundness.
- Decoy subset (`_decoy_subset`) — never perturbs sandbox; only speculative slots; PAC strip prob
  vs MTE never-strip invariant (baseline/decoy must agree on the arch path).
- Sandbox AND-mask zeroes top-16 bits where the PAC sig lives → clamp MUST precede the value seal in
  one slot. Verify ordering in all three sealers and that offset-cancel SUBs come after.

## Verification
- Full aarch64 suite (baseline must stay green):
  `cd /home/gal_k_1_1998/sca-fuzzer && /home/gal_k_1_1998/revizor/revizor-venv/bin/python -m unittest discover -s tests/aarch64_tests -p 'unit_*.py' -v`
  (32 unit_*.py files; some CE tests need `/dev/executor` + the chmod above; HW-trace tests are
  skipped/ENODEV on this box.)
- Generated-assembly inspection without HW: the `investigate_seal_fpac.py` pattern —
  `gen.create_test_case(path, disable_assembler=True)` then
  `Aarch64Printer(td).print_layout(Aarch64ASMLayout(tc))`; load via the executor to seal, dump the
  sealed TC asm, confirm per-cell corner cases (regular/N × none/MTE/PAC/both).
- Matrix configs exist in `/home/gal_k_1_1998/`: `cfg_reg_mte.yml`, `cfg_reg_pac.yml`,
  `cfg_ni_none.yml`; `config_pac_mte_basic.yml`, `config_mte.yml`, `config_pac.yml`,
  `config_pac_mte.yml`. Driver: `/home/gal_k_1_1998/run_matrix.sh` (HW cells ENODEV here).

## Suggested execution order
1. Read everything; write the file-by-file findings doc (no edits). Agree severity/scope with user.
2. Comment strip + naming + dead-code/import cleanup (lowest risk), suite green after each file.
3. Structural moves (subpackages, renames, split trace/log) — mechanical, re-run suite + imports.
4. Correctness fixes found in step 1 (encodings/sysregs/algorithmic) — each with a regression test.
5. Coupling reduction (common↔aarch64) — interfaces/DI seams.
6. Test consolidation + corner-case additions.
7. Docs/README update last.
