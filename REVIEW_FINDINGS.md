# AArch64 Seal Subsystem — Review Findings

> REVIEW phase complete (read-only; NO code edited). One row per finding. Severity:
> **P0** correctness/security (wrong trace, FPAC risk, unsound) · **P1** structural
> (naming/coupling/dead code/safety-hardening) · **P2** style/comments/tests polish.
> Status: `open` → `agreed` → `fixed` → `verified`. All findings below are `open`.

## Verdict summary
- Files reviewed: 13 Python (full seal subsystem + executor + trace/CE/disasm + input trio + log)
  + cross-checked C ground truth (pac_sign_plugin.c, mte_tag_plugin.c, simulation_output.{c,h},
  chardevice.c, executor headers).
- **P0 open: 3 · P1 open: 8 · P2 open: ~14.**
- Suite baseline at start: **371 pass / 2 fail / 13 skip** (373 ran). The **2 failures are a test-isolation
  bug, NOT product bugs**: `unit_pacga` passes 20/20 in isolation; `unit_ni_random.setUpClass` mutates
  global kernel PAC keys with no `tearDownClass`, leaking deterministic keys into `unit_pacga`
  (runs after, alphabetically). **True baseline is effectively green** (see F-T1). Fix F-T1 → 373/0/13.
- Safety net intact: `CE_DEBUG_PAC_AUTH_NO_FAULT = 0`; CE auth uses `REVISOR_PAC_AUTH` (kernel
  verify-then-auth net); `mte_is_mem_tag_access` covers the full LDG/STG/STGP family.

---

## ⚠️ Reviewer guidance — cross-layer claims need an empirical check, not code reasoning alone

F-P0-1 / F-P0-2 were initially suspected over-flags (the CE models no MTE tag-check fault and applies
an after-access tag correction, so a Python-only reading suggested the placeholder trace already
matched the genuine register state; `MtePacSealedTestCase.resolve` was briefly collapsed to a single
placeholder trace on `07ce424`). **A deterministic repro refuted that** — `investigate_seal_fpac.py`
trial 52, `AUTDA x1, x2`: ctx `0xa60ea833c47f1100` (placeholder) vs `0x960ea833c47f1100` (genuine),
so the placeholder-signed sig was `0xffd4` while the genuine auth needs `0xff98` → would-FPAC.

**Mechanism (confirmed real):** an earlier MTE seal `ADDG reg` retags a register later used as an
`AUT*` context, with an address `SUB reg,reg,x0` between them. The CE after-access correction repairs
only the tag nibble `[59:56]`; the `SUB`'s borrow into bits `[63:60]` is computed on the ADDG-modified
register in genuine but the raw register in the placeholder — so the placeholder register at the auth
≠ genuine, and the placeholder-signed sig is wrong. The after-access correction is NOT positionally
equivalent to the genuine `ADDG`.

**Fix (at source):** the real cause was the seal slot ORDER — the MTE retag (`ADDG`) was placed
*before* the offset-cancel `SUB`. Reordered so the MTE retag is LAST, immediately before the access
(`sandbox -> auth -> sub -> mte -> access`); now no address op runs between the retag and the access,
the CE after-access correction is positionally equivalent to the genuine retag, and the placeholder
trace is correct. So `MtePacSealedTestCase.resolve` signs PAC over the single placeholder trace `cer`
(no `cer_tagged` re-trace). Slot-ordering ownership moved into the per-combination `SealedTestCase`
(the `Sealer`/walk no longer assumes value-seal order; no composite). Verified: every PAC slot signs
the same sig in placeholder vs genuine-tagged (`compare_seal_traces.py`), 0 would-FPAC in 150 trials
(`investigate_seal_fpac.py`), suite 373 OK. Regression test: `tests/aarch64_tests/unit_seal_ordering.py`.

**Instruction to the review agents:** verify any finding whose correctness depends on CE/kernel/HW
behavior with an empirical repro, not code reasoning alone — the C-only argument here was wrong
because it missed the `SUB`/ordering interaction.

## TOP PRIORITY — P0 correctness/soundness

### F-P0-1  MTE `spec_nesting` classified over the PLACEHOLDER trace, not the genuine-tag trace
- (P0, open) `aarch64_sealer.py:302-307` (`MteSealedTestCase.resolve`) traces only the placeholder
  `cer = trace_fn(self._tc, inp)` and resolves MTE (incl. `spec`) over it. `aarch64_sealer.py:322-324`
  (`MtePacSealedTestCase.resolve`) resolves MTE over `cer` (placeholder) while PAC correctly uses
  `cer_tagged`. The placeholder MTE slot is `[NOP]` (pointer keeps its raw/mismatching tag); the
  genuine slot is `[ADDG]` (pointer retagged to the cell tag). A tag mismatch changes the reached
  path, so a slot's `speculative` flag (`_Resolved.speculative = spec_nesting != 0`, line 124) can be
  mis-derived. `ResolvedSealingTestCase._fill` (line 155) only perturbs `speculative` slots, so a slot
  that is architectural in the genuine execution but classified speculative over the placeholder can be
  decoy-perturbed → breaks the baseline/decoy arch-path agreement (false violations) and can fire an
  architectural MTE tag-check fault. This is the documented-but-unfinished "BUG #2".
- Fix: classify MTE `spec` over the MTE-genuine re-trace. `MtePac` already builds `cer_tagged` — use it
  for MTE `spec_nesting` too (keep `correct_tag` from placeholder: cell allocation tag is independent of
  the pointer's own tag). `Mte`-only must ADD the re-trace (resolve tags → fill MTE genuine → re-trace →
  classify spec over it).
- Verify: unit test — a TC where an MTE slot is architectural in genuine but speculative-looking in the
  placeholder; assert it is NOT in the decoy-eligible set. Re-run full suite.

### F-P0-2  Zero-context `AUT*ZA/ZB` (ctx==0) still would-FPAC (resolve-over-placeholder ≠ genuine reg state)
- (P0, open) `aarch64_sealer.py:367-381` (`_seal_auths`) seals every generator-emitted AUT* including
  the zero-context variants; `_resolve_pac` (`:185-208`) signs the pointer read at the slot's XPAC in
  the (placeholder for Pac-only / `cer_tagged` for MtePac) trace. For a standalone non-memory auth the
  pointer register is NOT sandbox-clamped, and the placeholder `[NOP,XPAC]` chain can leave it at a
  value that differs from the genuine `[MOVK,AUT*]` chain at auth time → wrong `correct_sig` → the
  genuine architectural AUT* FPAC-faults. Mitigated to non-fatal by the kernel net (returns canonical),
  but that means the genuine TC's architectural result silently diverges from what the seal intended —
  an unsound contract trace, and reliance on the net rather than correctness. The generator fix
  (`_patch_auth_context_collision`) only covers the explicit-context family `_AUTH_CTX`.
- Fix (per "fix-at-source"): iterative PAC resolution — resolve sigs in program order, applying each
  genuine sig into the TC before resolving the next so each sig sees genuine register state; OR a
  genuine-PAC re-trace pass. Confirm with `investigate_seal_fpac.py` (it hits trial 85 today) + `dmesg |
  grep "would FPAC"`.
- Verify: `investigate_seal_fpac.py` many trials → 0 would-FPAC; full suite green.

### F-P0-3  `PAC_KEYS_WORDS = 9` is wrong — the kernel struct is 10×u64 (80 B) [latent]
- (P0, open) `aarch64_input_wire.py:36` packs `9*8=72` B; `struct pac_keys` (executor/pac.h:6-12) and
  `struct ce_pac_keys` (userapi/executor_pac_api.h:27-33) are 5 keys ×(lo,hi)=**10 u64 = 80 B**, and the
  kernel rejects on size (`chardevice.c:583 if (sizeof(struct pac_keys) != sec->length) return
  -EINVAL`). Dormant only because `pac_keys` is never serialized (`aarch64_contract_executor.py:97`
  `pac_keys=None`; `serialize_input` only ever called with `mte_tags=`). The moment per-input PAC keys
  are wired, every input_init is rejected on both CE and kernel — exactly the stale-parse hazard from
  `[[input-format-rebuild-both-sides]]`.
- Fix: `PAC_KEYS_WORDS = 10`; correct both comments (`input_wire.py:36`, `contract_executor.py:97`).
- Verify: `struct.pack('<10Q', *range(10))` is 80 B == `sizeof(struct pac_keys)`; add a PAC_KEYS-section
  round-trip parse test.

---

## Per-file findings

### src/aarch64/aarch64_seal.py  (176, shared seal primitives)
- [S1] (P2) `_make_sandbox_insts(self, reg, align16=False)` default arg (`:131`); also `_topo_sort`
  fine. — make `align16` explicit at the call site (`aarch64_generator.py:391`).
- [S2] (P1) Coupling: both the generator's `Aarch64SandboxPass` and the sealer's `SandboxWalk` build
  sandbox instrumentation off this base; two parallel sandboxing paths. Acceptable, but note for the
  coupling-reduction step (shared base is the right seam — keep it).
- [S3] (P2) `fill_slot_at` docstring still says "padding short fills with NOPs" while the assert now
  rejects overflow only; short fills DO still NOP-pad (`:59`). Comment is fine; just verify intent.

### src/aarch64/aarch64_sealer.py  (565, live seal engine)
- See F-P0-1, F-P0-2.
- [SR1] (P2) Default args: `make_sealer(..., signer=None)` (`:554`), `_wrong_sigs(..., size=6,
  tries=64)` (`:210` — user explicitly rejected `size=6` before), `_random_sigs(..., size=6)` (`:225`),
  `Sealing.seal(self, value=None)` + all overrides (`:44,59,84,104`). Make required; pass explicitly.
- [SR2] (P2) Magic constants: `_STRIP_PROB=0.1` (`:75`), decoy `0.5` (`:150`), `_wrong_sigs` `0.85`
  (`:214`) — user prefers configurable data over hardcoded probabilities.
- [SR3] (P1, minor-soundness) `PacSealing.seal` rolls the `_STRIP_PROB` coin independently inside each
  `genuine()` and `decoy()` `_fill` call (`:87`), so (a) `genuine()` is non-deterministic and (b) an
  architectural PAC slot can be `[NOP,XPAC]` in genuine but `[MOVK,AUTH]` in decoy (both canonical, so
  arch RESULT matches — but instructions differ on the arch path). Also a decoy slot meant to carry a
  wrong sig can silently strip (neutering that decoy). Document/justify or move the strip decision out
  of the per-call `seal`.
- [SR4] (P2) `_resolve_pac` reads `ptr=_read_reg(ite.cpu, value_reg)` at the XPAC PC (`:202`); confirm
  the trace's `ite.cpu` is the pre-instruction state and the sign basis (canonical low bits) matches the
  genuine AUTH input. (Verified sound for clamped memory pointers; the unclamped standalone case is
  F-P0-2.)
- [SR5] (P2) Heavy comments throughout (class docstrings 10+ lines) — user wants aggressive trimming.

### src/aarch64/aarch64_pac.py  (139, PAC encoders/signer/specs)
- [PA1] (P1, dead) `_AUTH_TO_KEY` (`:34`) is never referenced anywhere — delete.
- [PA2] (P2) Default args: `field_mask16(self, mn, samples=64)` (`:97`); `_pick_mem_auth`/
  `AuthInstructionSpec.generate` hardcode `range(20)` retries (`:70,115`) — make the retry budget an
  explicit/configurable value.
- [PA3] (non-finding, verified) `_PAC_INFO` key/auth/xpac mapping correct (XPACI for I-keys, XPACD for
  D-keys); `field_mask16` bit-55 guard (`assert not (m & 0x80)`) correct; `make_movk` masks `&0xFFFF`.

### src/aarch64/aarch64_mte.py  (67, MTE tag-state model)
- [MT1] (P2) `_reg_value` returns `0` for an unrecognised reg name (`:53`) — a silent fallback the user
  dislikes; raise on an unexpected name (xzr/wzr→0 is legitimate but should be explicit).
- [MT2] (P2) `set(self, addr, tag, n_granules=1)` default arg (`:36`) — always called with 3 args;
  drop the default.
- [MT3] (non-finding, verified) tag = `(Xt>>56)&0xF` = bits[59:56] correct; `granule()` drops tag byte
  + in-granule offset correctly; `to_depth` speculation stack grow/pop correct.

### src/aarch64/aarch64_generator.py  (436, generator-side fixes)
- [G1] (non-finding, verified) `_patch_auth_context_collision` (`:193-198`) + `_AUTH_CTX` (`:128`) force
  ctx≠ptr for autia/autib/autda/autdb, runs on every body instruction via `_patch_instruction` — the
  self-context fix is COMPLETE for explicit-context auths (no index error: zero-context variants are
  excluded from `_AUTH_CTX`). Zero-context remains F-P0-2 (different mechanism, correctly out of scope
  here).
- [G2] (P2) `_replace_reg` (`:354-357`) "guard against it rather than crashing" raises RuntimeError on
  empty candidates — fine, but the comment is stale-ish; trim.
- [G3] (P1) `Aarch64SandboxPass.sandbox_memory_access` docstring (`:380-382`) documents a real
  limitation: the mask bounds base, not base+access_size, so a 16-B LDP/STP near the region end can
  spill. Pre-existing; record as a known soundness caveat (widen mask to region−max_access_size).

### src/aarch64/aarch64_executor.py  (950, regular-sealed + NI executors)
- [E1] (P1, safety-hardening) No defensive arch-forgery assertion before assembling/writing a
  genuine/decoy TC to HW (`_assemble_and_write_test_case` ~839/934; `trace_test_case_variants_hw`
  ~837-846). Safety rests implicitly on the `speculative` flag. Add an assert that no perturbed entry
  has `speculative == False` (in `ResolvedSealingTestCase._fill` or pre-write). Verify: force an
  arch-slot perturbation in a unit test → assert fires.
- [E2] (P1) Temporal coupling: the injected trace_fn reads mutable `self._sandbox_base`/`self._nesting`
  (`:716-722`, set at `:750-751,917`) with no guard (contrast `self._sealed is not None` at `:748`). A
  `resolve()` before `read_base_addresses()` silently passes `req_mem_base_virt=None` to the CE.
  Fix: assert `_sandbox_base is not None` in `_seal_trace`, or pass base/nesting as explicit trace_fn
  args (extend the DI discipline).
- [E3] (P1, dead/dup) `_print_ce_trace_comparison` re-implements a nested `_run_ce_on_tc` (`:530-540`)
  identical to the method at `:459-469` — delete the nested copy, call `self._run_ce_on_tc`.
- [E4] (P2, dead) Unused imports `SANDBOX_BASE_REGISTER` (`:24`), `HardwareTracingError` (`:20`, only in
  docstrings), dead attr `previous_num_inputs` (`:85`); stale TC1/TC2/TC3 docstring (`:520-525`); dead
  `max_mispred_instructions` plumbed but "unused by CE" (`:293,299`).
- [E5] (P2) Default args `_make_ce_execution(..., enable_mismatch_check_mode=False, bp=NONE,
  mte_tags=None)` (`:89,294,295`).
- [E6] (non-finding, verified) `_mte_tags_for` returns `[MTE_INITIAL_TAG]*MTE_TAG_COUNT` only when 'mte'
  active else None (`:708-714`); PAC keys pinned per TC (`:736`); `collapse_key` grouping SOUND (two
  inputs share a TC iff every entry has identical `(value, speculative)`; decoy perturbs only
  speculative slots → arch-safe for the whole class).

### src/aarch64/aarch64_trace.py  (299, CE-blob parser + contract-trace)
- [T1] (P1) Latent `extra_data` layout divergence: Python parses `extra_data` as inline variable-length
  between `cpu` and `metadata` (`:71-74`), but C `trace_cpu_state_t` has no flexible-array member and
  the trace is fixed-stride; they agree only because the serializer hardcodes `extra_data_size=0`
  (`simulation_output.c:318`). Any nonzero extra_data → permanent desync. Fix: assert
  `extra_data_size == 0` on the Python side (or add a real FAM + variable stride in C).
- [T2] (P1, structural) `show_context` (`:290-299`) + `pretty_print` methods are debug/violation-dump
  only (imported lazily by `fuzzer.py:590-595`), NOT in `compute_taint`/`compute_ctrace` — split into a
  debug module; rename the core to `ce_result.py`/`contract_trace.py`. Keep `ContractExecutionResult`/
  `compute_taint`/`compute_ctrace` (live).
- [T3] (P2) Default args on `pretty_print(..., indent=0)` (`:43,84,127,150`) and `show_context(...,
  window=-1)` (`:290`). No length cross-check after the parse loop (`:158-172`) — add
  `assert off == len(self._buf)`.
- [T4] (non-finding, verified) Blob layout matches the CE field-for-field; the stream_ipc false-EOF
  regression is fixed (`stream_ipc.py:41-90` treats empty `read1()` as would-block, EOF only after
  `proc.poll()`); gpr indexing is natural order `gpr[N]=xN` (CE `simulation_output.c:278-308`), x29 read
  correct.

### src/aarch64/aarch64_contract_executor.py  (245, CE request encoding + process mgmt)
- [CE1] (P2) `pac_keys: Optional[list]=None` (`:97`) comment says "9 u64" — wrong (see F-P0-3).
- [CE2] (non-finding, verified) AARCH64_NUM_GPRS=31, request encoding matches the CE.

### src/aarch64/aarch64_disasm.py  (137, capstone decode helpers)
- [D1] (non-finding, verified) `decode_tag_store` correctly returns (mnemonic, Xt, base, signed disp)
  for STG/STZG/ST2G/STZ2G; `is_conditional_branch` live.

### src/aarch64/aarch64_input_wire.py  (137, executor-ABI serializer)
- See F-P0-3.
- [IW1] (P2) Yoda in Python: `if 0 == i % 2:` (`:91`) — should be `i % 2 == 0`.
- [IW2] (P2) Default args `build_input_init(..., simd=None, mte_tags=None, pac_keys=None)` (`:105-107`),
  `serialize_input(inp, mte_tags=None, pac_keys=None)` (`:125-126`).
- [IW3] (non-finding, verified) Magic, preamble (48 B), section descriptor (32 B), GPR section (64 B,
  x0..x5/sp/flags), MTE 512 tags → 256 B low-nibble-first all match the kernel headers + chardevice.c.

### src/aarch64/aarch64_input_layout.py  (96) / aarch64_input_generator.py  (24)
- [IL1] (non-finding, verified) NZCV bits[31:28] placement and randomization correct; layout indexing
  matches the CE order.

### src/aarch64/aarch64_log.py  (235, observability)
- [L1] (P1, delete) Fully severable: imported only by `aarch64_executor.py`; ~10 side-effect-only call
  sites + 4 `log.register` + `_maybe_log_ni_table`/`_ni_table_columns`; ZERO C coupling. `log_pac_op`
  (`:186`) and `log_slot` (`:192`) are dead (never imported/called). Recommend DELETE the module + the
  ~16 executor lines (user: "I don't need logs"). `_maybe_log_ni_table` already self-gates on
  `REVIZOR_NI_TABLE` — that's the precedent if quarantine is preferred instead.
- [L2] (P2) Default args `ch: Optional[str]=None` (`:48,66,100,119,199`) — moot if deleted.

### Tests
- [F-T1] (P1, testing) `tests/aarch64_tests/unit_ni_random.py:60-63` (`setUpClass`) sets deterministic
  global kernel PAC keys with NO `tearDownClass` to restore them → leaks into `unit_pacga`
  (alphabetically later) → 2 false failures (`test_real_auth_roundtrip_instruction_key`,
  `test_live_keys_roundtrip`; both assert `signed != PTR`, the deterministic APIA yields no PAC field
  for the test pointer). Fix: add `tearDownClass` calling `set_pac_keys(None)` (restore live). Verify:
  full discover run → 373/0/13.

---

## Cross-cutting findings (by rubric dimension)

1. **Low-level correctness:** F-P0-3 (PAC keys 80 B), [T1] extra_data. Verified-correct: MTE
   bits[59:56], PAC field/XPAC mapping, MOVK LSL#48 mask, 16-B STG alignment, gpr natural order, NZCV
   bits, sandbox mask. Caveat [G3] (mask bounds base, not base+size — 16-B access can spill).
2. **Emulation/contract fidelity:** F-P0-1 (MTE spec must come from genuine-tag trace) and F-P0-2
   (PAC sign over genuine reg state). PAC keys pinned model↔kernel↔CE↔HW; MTE_INITIAL_TAG=6 uniform;
   CE auth via kernel net. [T1] is a fidelity time-bomb gated on extra_data_size.
3. **Parsing:** CE blob + input wire verified correct except F-P0-3 (dormant) and [T1] (latent); no
   silent truncation; stream_ipc false-EOF fixed.
4. **Algorithmic soundness/completeness:** F-P0-1 (decoy may perturb arch MTE slot), F-P0-2 (zero-ctx),
   [SR3] (independent strip coin). `collapse_key`, `_decoy_subset`, MTE tag stack verified sound.
5. **Security robustness:** No forged-AUT*-to-HW path found for the explicit-context + MtePac PAC flow
   (PAC resolved over `cer_tagged`). Gaps: F-P0-2 (zero-ctx relies on the net), F-P0-1 (arch MTE
   tag-fault possible), [E1] (no defensive assertion). `CE_DEBUG_PAC_AUTH_NO_FAULT=0` confirmed.
6. **Structure/naming/coupling:** [T2] split trace; [L1] delete log; aarch64_seal/sealer name clash;
   [E2] extend DI to base/nesting.
7. **Testing:** F-T1 isolation bug; add regressions for F-P0-1/2/3, decoy-subset arch-invariant, wire
   PAC-keys round-trip, MTE spec-classification.
8. **Dead code/cleanup:** [PA1] `_AUTH_TO_KEY`, [E3] dup `_run_ce_on_tc`, [E4] unused imports/attr/stale
   docstring, [L1] dead `log_pac_op`/`log_slot` + whole module, pervasive over-commenting.

---

## Consolidation decisions
- [x] **seal/sealer/pac/mte grouping — DECISION: adopt a `seal/` subpackage** (after correctness fixes,
  as one mechanical move): `seal/primitives.py` (← aarch64_seal.py), `seal/sealings.py` (Sealing
  classes), `seal/resolvers.py` (_Pac/_MteResolver), `seal/cases.py` (SealedTestCase hierarchy +
  ResolvedSealingTestCase), `seal/sealers.py` (Sealer/SandboxWalk/concrete + make_sealer),
  `seal/pac.py`, `seal/mte.py`. Justified: 4 tightly-coupled files for one subsystem; package clarifies
  the layering. Moderate import churn (executor + tests) — do as a single step with the suite as the
  guard.
- [x] **aarch64_seal vs aarch64_sealer rename — DECISION:** the subpackage resolves it
  (`seal/primitives.py` vs `seal/sealers.py`). If the subpackage is deferred, rename
  `aarch64_seal.py` → `aarch64_seal_primitives.py`.
- [x] **input_* trio — DECISION: LEAVE as-is.** Already well-named, small, minimal coupling; a
  subpackage forces import churn for no behavioral gain. Optional low-priority rename `input_wire.py`
  → `executor_input_abi.py` to signal it mirrors the kernel ABI.
- [x] **aarch64_trace split + rename — DECISION: YES.** Move `show_context` + `pretty_print` to a debug
  module; rename core to `ce_result.py` (or `contract_trace.py`). Keep the live
  ContractExecutionResult/compute_taint/compute_ctrace.
- [x] **aarch64_log delete vs quarantine — DECISION: DELETE.** Fully severable, zero C coupling, 2 dead
  fns, user doesn't want logs. Strip the ~16 executor call sites.

## Recommended fix execution order
1. **F-T1** (test isolation, trivial) → restores a truly green baseline to guard every later step.
2. **F-P0-3** (PAC_KEYS 9→10, trivial, dormant) + add the PAC-keys wire round-trip test.
3. **F-P0-1** (MTE spec over genuine-tag re-trace) — with a decoy-subset arch-invariant regression test.
4. **F-P0-2** (zero-context PAC iterative/genuine-retrace resolution) — hardest; gate on
   `investigate_seal_fpac.py` + dmesg.
5. **Safety hardening** [E1] arch-forgery assertion, [E2] base/nesting DI guard.
6. **Dead code / cleanup**: [PA1], [E3], [E4], [L1] (delete log), [T1] assert, unused imports; then
   aggressive comment stripping.
7. **Structural moves**: [T2] trace split, `seal/` subpackage, renames (suite green after each).
8. **Style**: default args, [IW1] Yoda, magic constants → configurable.
9. **Test consolidation + corner cases**; README/docs last.

## Execution log (after agreement)
| step | change | files | suite result | commit |
|------|--------|-------|--------------|--------|
|      |        |       |              |        |
