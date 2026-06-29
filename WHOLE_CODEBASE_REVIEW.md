# Whole-Codebase Review (excl. x86 as fix target; x86 used as parity oracle)

Scope: AArch64 Revizor fork — Python harness (`src/`), the contract executor (CE,
`src/aarch64/contract_executor/`), the kernel module (`src/aarch64/executor/`), the
userland CLI (`src/executor_userland/`), and the test suites. x86 is treated as a
read-only parity oracle, never a fix target. This document is the full-detail record;
the calling summary is reproduced at the end of the workflow message.

NO SOURCE WAS EDITED to produce this review. This is the only file written.

---

## 1. Verdict summary

### Post-verification severity counts
- **P0: 2** — F-KERN-1 (MTE sync tag-faulting silently disabled), F-SEAL-2 (zero-context AUT* would-FPAC / silently-unsound contract trace).
- **P1: 9** — F-ABI-1, F-ERR-1, F-ERR-2, F-CE-1, F-CE-2, F-CE-3, F-TEST-1, F-TEST-2, F-TEST-3.
- **P2: 58** — the remainder (ABI/flow/CE-decoder/CE-plugin/kernel-IO/taint/gen/trace/postproc/IR/test/dead-code/style).
- **Total verified findings: 69.**

### Overall health per area
- **Kernel arch (mte.c / sctlr / pmu / bpu):** one P0 (TCF bit position) that silently disables the entire MTE-fault contract on HW; otherwise a cluster of latent/foot-gun P2s. The P0 fix is correct but its *activation* is a box-reset hazard and is deferred behind a recovery net.
- **Seal subsystem:** one P0 (zero-context AUT* resolution). The whole seal/NI safety net leans on a per-slot `speculative` flag that the resolver itself can mis-set; current guard tests trust that same flag, so the subsystem is "green-but-unproven" until F-TEST-2's independent oracle lands.
- **CE simulation core / plugins:** several genuine decode-soundness bugs (sign-extending loads classified as stores, LDPSW double-mishandled, no barrier/length window termination). All AArch64-CE-internal, no ABI churn, but the C test suite is **not in CI** (F-TEST-1), so these regress silently.
- **Common flow (`fuzzer.py`):** structurally the weakest area — the AArch64 fused-executor tuple shapes leaked into shared code, the Unicorn-model wiring is half-dead, and `self.model`/`arch_*` are declared but never assigned. The generic loop AttributeErrors on x86. This is parity damage, not just cleanliness.
- **Error handling / glue:** AArch64 lacks the x86 `HardwareTracingError` invariant; a CE crash or kernel `-EIO` aborts the whole campaign despite the respawn design (P1 pair).
- **ABI mirrors (PY/CE/kernel):** one real latent size mismatch (PAC_KEYS 72B vs 80B) plus several "agree-only-by-coincidence" couplings (extra_data_size==0, HTRACE_WIDTH duplicated). Dormant today, pre-emptive fixes.
- **Taint / generator:** soundness rests on a hand-maintained capstone fixup denylist; safe today only because the unsound mnemonics are out of `supported_instructions`. Over-tainting fixes are the safe direction.
- **Tests / CI:** the highest-leverage gap. C suites + aarch64 mypy not wired into CI; the seal guard tests trust the resolver; a PAC-key leak causes the 2 known baseline failures.

### Unit-suite baseline (known-good reference)
**371 pass / 2 fail / 13 skip.** The 2 failures are the `unit_ni_random` deterministic-PAC-key isolation leak into `unit_pacga` (F-TEST-3 / seal-plan F-T1) — a cross-test contamination, **not a product bug**. `unit_pacga` passes 20/20 in isolation. After F-TEST-3 the expected baseline is **373/0/13**.

---

## 2. P0 findings (full detail)

### F-KERN-1 — SCTLR_EL1.TCF written to the wrong bit position; MTE synchronous tag-check faulting is silently DISABLED on hardware
- **Severity:** P0 · dimension 1 · confidence high · cross-area · **verified**
- **Files:** `src/aarch64/executor/mte.c:79-99` (the write is line 81, inside `mte_set_sync`, reached only via `enable_mte_tag_checking()` called once at module init from `executor.c:106`).
- **Evidence:** `mte_set_sync` does `sysreg_clear_set(sctlr_el1, SCTLR_EL1_TCF_MASK, SCTLR_EL1_ATA | SCTLR_EL1_TCF_SYNC)`. Verified against this box's headers (`sysreg-defs.h`): `TCF_MASK=GENMASK(41,40)`, `TCF_SHIFT=40`, `TCF_SYNC=UL(0b01)` is a **raw, unshifted** field value. So the code clears bits[41:40] then ORs `0b01` into **bit0** (= SCTLR.M, already 1 → no-op), leaving TCF=0b00 = "tag check faults have no effect". `ATA` (bit43, already in position) is set correctly and masks the bug (STG still tags memory), but loads/stores **never fault** on a tag mismatch. The CE/PY model *does* model MTE tag faults → HW can never reproduce a modeled MTE fault → the MTE NI/sealed contract comparison is unsound on HW.
- **Fix:** place the field value in its field: `sysreg_clear_set(sctlr_el1, SCTLR_EL1_TCF_MASK, SCTLR_EL1_ATA | (SCTLR_EL1_TCF_SYNC << SCTLR_EL1_TCF_SHIFT))`. Prefer the shift form over `FIELD_PREP` — `<linux/bitfield.h>` is not currently included and the shift form needs no new dependency. This yields bit40 set / bit41 clear (TCF=0b01 synchronous), ATA(bit43) still set, TCF0[39:38] untouched.
- **Verify (non-destructive only):** after `enable_mte_tag_checking` read back SCTLR_EL1 and `printk`-assert `(val & TCF_MASK) == (TCF_SYNC<<TCF_SHIFT)`. **Do NOT trigger a real tag-mismatch access on this VM.**
- **Impact / blast radius:** the *textual* change is one line, but the **behavioral** blast radius is large. The JIT'd test case runs at EL1 (`measurement.c:173`, IRQs disabled) with **no exception fixup / extable / recovery handler anywhere in the executor tree**. With TCF=sync, the first tag-mismatched access inside a test case becomes an unhandled synchronous EL1 data abort → kernel Oops → **hard VM reset** (same hazard class as the EL1 AUT* FPAC reset). Enabling it also makes *live* the MTE-fidelity divergences currently masked because nothing faults: F-KERN-2 (stale tags), F-KERN-8 (unmasked tag), F-CE-10 (IRG), F-CE-11 (STG-as-store), F-CE-2/F-CE-3 (load misclass).
- **Risk:** **high.** The bit-position correction is correct and should be made; *enabling* sync mode is the hazard.
- **Reconciliation outcome:** **the bit-position fix lands (with the readback assert) but synchronous enablement is DEFERRED** until (a) the MTE-fidelity fixes F-KERN-2/F-KERN-8/F-CE-2/F-CE-3/F-CE-11/F-CE-10 land and (b) an EL1 tag-fault recovery net (analogous to the PAC verify-then-auth net) exists. Consider async/asymm (TFSR_EL1) mode as a non-faulting interim. ABI-neutral (no PY/CE change); kernel module recompile only.

### F-SEAL-2 — Zero-context AUT*ZA/ZB (ctx==0) standalone non-memory auth still would-FPAC; placeholder reg state ≠ genuine, so the wrong `correct_sig` is signed
- **Severity:** P0 · dimension 4 · confidence high
- **Files:** `src/aarch64/aarch64_sealer.py:367-381` (`_seal_auths` zero-context AUT* sites), `185-208` (`_PacResolver._resolve_pac`).
- **Evidence:** for a standalone non-memory auth the pointer reg is **not** sandbox-clamped (memory-base auths get a `SandboxSealing` clamp first; standalone ones do not), and there is no context operand (ctx=0). `correct_sig` is signed over the pointer read from the **placeholder** `[NOP,XPAC]` trace, but the genuine TC runs `[MOVK sig, AUT*]`. For these unclamped/program-order-dependent regs the placeholder reg state at the XPAC offset ≠ the genuine reg state at the AUTH, so `correct_sig` is wrong → the genuine AUT* **would FPAC**, saved only by the kernel verify-then-auth net (which returns canonical) → **silently unsound contract trace**. The generator's `_patch_auth_context_collision` covers only explicit-context auths, not zero-context. Durable repro `investigate_seal_fpac.py` hits it at trial 85.
- **Fix:** iterative/program-order PAC resolution or a genuine-PAC re-trace so `correct_sig` matches the genuine register state for zero-context auths. The genuine re-trace already exists at `aarch64_executor.py:766` (after `resolve()` at :764) — reuse it (or an in-resolve fixpoint). **Recompute only the sig VALUE; keep the speculative-flag (`spec_nesting`) classification source unchanged** so decoy/NI eligibility is not perturbed.
- **Verify:** `investigate_seal_fpac.py` no longer reaches a would-FPAC; an end-to-end genuine-resolution check (kernel non-faulting XPAC + re-sign compare) reports WOULD-SUCCEED for every emitted AUT* without executing a faulting one.
- **Impact / blast radius:** Python-only resolution logic; **no wire/CE/kernel rebuild**, no stale-parse hazard. Any genuine AUT* the re-trace runs goes through the CE kernel verify-then-auth net (never a forged AUT* raw). Touches `_PacResolver._resolve_pac`, `PacSealedTestCase.resolve` (290-293), `MtePacSealedTestCase.resolve` (323-327); `MteSealedTestCase` unaffected.
- **Risk:** **medium** (resolution-logic change; must not shift the speculative classification surface).
- **Reconciliation outcome:** implemented **together with F-P0-1** (MTE spec classified over the genuine-tag re-trace) since both restructure `MtePacSealedTestCase.resolve` staging. Sequenced **after** F-GLUE-1 (sandbox-base assert, its re-trace prerequisite). Gated by F-TEST-2 as the acceptance oracle — current `unit_regular_sealed`/`unit_ni_random` trust the resolver's own flag and **cannot catch this**. Re-verify against `simulation_hook.c` first (the REVIEW_FINDINGS C cross-check flags F-P0-1/F-P0-2 as possible over-flags: the CE never models an MTE tag-check fault and applies an after-access tag correction, and the MtePac resolve was already collapsed to a single placeholder trace on this branch).

---

## 3. P1 findings (grouped by area, each with its impact summary)

### ABI (cross-area)
**F-ABI-1 — PAC_KEYS section is 72B (9 words) on PY+CE but the kernel requires 80B (`struct pac_keys`, 10 words); `apga_hi` dropped, any device PAC_KEYS input `-EINVAL`'d.**
- Files: `aarch64_input_wire.py:36`, `aarch64_contract_executor.py:97`, `contract_executor/simulation_input.h:102` + `.c:113-119`, `executor/chardevice.c:581-585`, `executor/pac.h:6-12`, `test_ce.c:788-810`, `unit_input_wire.py:104`.
- Evidence: `struct pac_keys` = 5 keys × {lo,hi} = 80B (correct). `chardevice.c:581` rejects any PAC_KEYS section unless `length==80`. PY `PAC_KEYS_WORDS=9` (72B) and CE `pac_keys[9]` agree at 72B but silently drop `apga_hi`. Latent today: `serialize_input` is never called with pac_keys (device keys flow via the `SET_PAC_KEYS` ioctl using the correct 80B `ce_pac_keys`), and the CE never *consumes* the loaded `sim_input->pac_keys`. Both C and Python unit tests codify the wrong 72B.
- Fix: adopt one canonical 10-word/80B definition matching the kernel field order (`apia,apib,apda,apdb,apga` lo/hi). `PAC_KEYS_WORDS 9→10`; CE `pac_keys[9]→[10]` (sizeof gate auto-moves 72→80); add `_Static_assert(sizeof(struct pac_keys)==80)` in `pac.h`; **no kernel parse change** (already 80B). Update both test sites; add a 72B-rejected negative test.
- Impact: this **is** a wire-format change on the input-init PAC_KEYS section; **PY+CE must move 72→80 together and rebuild the CE** — never a 80B PY serializer against a stale 72B CE. Kernel needs **no** rebuild. Dormant deployment (no live effect until per-input PAC keys are serialized), pre-empts the stale-parse box-reset hazard.
- Risk: **low.** Confidence high.

### Glue / error handling (cross-area)
**F-ERR-1 — AArch64 HW path never produces/translates `HardwareTracingError`; all-zero htraces silently accepted as real, kernel `-EIO` aborts the whole campaign (P1).**
- Files: `aarch64_executor.py:383-393`, `aarch64_kernel.py:318-327` & `344-351`, `x86_executor.py:160-167` (oracle), `aarch64_fuzzer.py:72`.
- Evidence: x86 raises on `htrace==0`/malformed. AArch64 `_aggregate_htraces` never inspects values and `LocalHWExecutor` returns the ioctl result raw. (a) a kernel measurement error yielding an all-zero htrace is accepted as legitimate (zeros compare equal → false equivalence / masked or spurious violations); (b) a real fault returns `-EIO`, `fcntl.ioctl` raises `OSError`, untranslated/uncaught → run aborts. The `try/except HardwareTracingError` at `aarch64_fuzzer.filter()` and `fuzzer.py:400` are **dead for AArch64**.
- Fix (the JSON `proposed_fix:"test"` is underspecified — a test alone does not fix it): source-side translation. Place the `htrace==0` check **at read-time in the 3 collection loops** (`trace_test_case` ~361, `trace_test_case_variants_hw` ~844, sealed `trace_test_case` ~946) or in `LocalHWExecutor.hardware_measurement` — **NOT in `_aggregate_htraces`**, which intentionally zeroes ignored/priming inputs (guarded by `unit_executor.py::test_ignored_input_htrace_is_zeroed`). Scope the `OSError→HardwareTracingError` translation to `errno==EIO` on `trace()`/measurement only, never a blanket `_ioctl` wrap (would mask SET_PAC_KEYS/MTE/allocate errors and the 60s box-wedge signal).
- Impact: restores the shared "htrace==0 means error, skip it" invariant; one bad measurement skips one TC instead of corrupting the analyser or killing the run. Defense-in-depth atop F-KERN-3. Python-only, no ABI.
- Risk: **medium** (placement-sensitive).

**F-ERR-2 — CE crash/hang raises bare `RuntimeError` on the main trace path with no handler; one CE fault aborts the campaign despite the respawn design (P1).**
- Files: `aarch64_contract_executor.py:207-245`, `aarch64_executor.py:429` & `716-722`, `fuzzer.py:388/531/900`.
- Evidence: `ContractExecutorService.run()` respawns the CE on crash/hang but **re-raises `RuntimeError`**. Every production caller is unguarded (`trace_test_case_with_taints`, `_seal_trace`); the fuzzer invokes them with no try/except (only HW sites catch `HardwareTracingError`). A CE segfault/hang — exactly what respawn is for — still kills the run.
- Fix: define `ContractTracingError` as a **subclass of `RuntimeError`** (so the existing `except RuntimeError` debug catchers at `aarch64_executor.py:466`/`530-540` keep working) raised by `run()` after respawn. Catch it where `_collect_traces` already catches `HardwareTracingError` (CE call at 388/900, distinct from the HW catch at 398/907), plus the slow-path sites (531/580). Return `[],[],[]` and bump a stat; add a **consecutive-skip circuit-breaker + warning** so a systemic CE crash cannot produce a silent green-but-empty run. Do NOT swallow silently.
- Impact: campaign continues across a CE fault; the respawn design finally works end-to-end. Python-only, no ABI.
- Risk: **medium** (exception-type plumbing + ordering against the FLOW fixes).

### CE simulation core
**F-CE-1 — CE speculation windows are never terminated by a barrier (DSB/ISB/SB/CSDB) and have no per-instruction length cap; diverges from x86 `UnicornSpec` (P1).**
- Files: `simulation_input.h:58`, `simulation_execution_clause_hook.c:100-180`, `branch_speculation.c:94`, `aarch64_executor.py:300/721`, `model.py:874-878` (oracle).
- Evidence: x86 stops a window on a barrier mnemonic AND when `speculation_window > model_max_spec_window`. The CE does neither: `max_misspred_instructions` is "// NOT SUPPORTED" (PY still passes `model_max_spec_window=250` "# unused by CE"); a grep of the CE tree finds **zero** DSB/ISB/SB/CSDB handling. Result: (a) the contract over-approximates window length and speculates past barriers → two HW-distinguished inputs collapse into one ctrace class → FPs/missed leaks; (b) a wrong-path backward branch loops to the 3600s watchdog.
- Fix (split): **(a)** per-window instruction cap — wire the **already-transmitted** `max_misspred_instructions` field (PY already serializes it at `aarch64_contract_executor.py:148`), add a counter to `execution_mgmt`, force a **single innermost-frame** unwind on overflow (do NOT reuse the `out_of_simulation` full-unwind loop). Also kills the watchdog hang. **(b)** barrier termination — recognize DSB/ISB/SB/CSDB as window terminators; this **shifts the per-entry `speculative` flag** the seal/NI subsystem trusts, so sequence it after the seal fixes and pick the per-clause barrier set deliberately (a DSB does not cut a store-bypass window the way CSDB/SB cut value/branch spec — coordinate with F-POST-1/F-MIS-1).
- Impact: ABI-neutral (the cap field already exists and is transmitted; kernel never parses the CE envelope). **Do NOT delete `CONF.model_max_spec_window` — x86 `model.py:878` consumes it.** Correction to the original evidence: `Aarch64DsbSyPass` is dormant (not in `self.passes`), so barriers reach the CE via the minimizer's `FenceInsertionPass` and the instruction pool, not generator injection.
- Risk: **medium.**

**F-CE-2 — Sign-extending loads (LDRSW/LDRSB/LDRSH, opc=10) misclassified as STORES by `is_load()`'s bit-22-only decode (P1).**
- Files: `instruction_encodings.c:114-115`, `simulation_output.c:113-115`, `simulation_hook.c:209-251`, `aarch64_trace.py:254-257`.
- Evidence: `is_load = (inst>>22)&1` reads opc[0]; for load/store-register opc is bits[23:22] (STR=00, LDR=01, LDRS-64=10, LDRS-32=11). For opc==10 → opc[0]=0 → `is_load==0`/`is_store==1`: the load is recorded as a write. `compute_taint` then calls `on_write` not `on_read` → depended-on input memory is NOT must-preserve → boosting mutates it → boosted inputs leave the class → FPs/FNs or the ctrace-mismatch assert. `base_hook_c`'s aliasing fixup also misfires.
- Fix (**modified** from the JSON "replace is_load/is_store accordingly" — too broad): apply the full `opc=(inst>>22)&3` decode **only to the regular load/store-register path** (`simulation_output.c` regular branch + the regular handling in `base_hook`). **Keep the bit-22/L-bit decode for the PAIR branch** — a global redefine makes a pre-indexed STP (bit23=1 → opc!=0) misclassify as a load, corrupting both taint and the memory fixup. Add the `size==3 && opc==2` PRFM non-access special-case (coordinate with F-CE-4). Regression tests: an LDRSW (expect read) **and** a pre-indexed STP (expect store).
- Impact: CE-internal decode; `mem_access_t`/entry layout unchanged → **CE binary rebuild only**, no kernel/PY change, no stale-parse hazard.
- Risk: **medium** (the naive fix regresses the pair path).

**F-CE-3 — LDPSW mishandled across two CE decoder paths: `__builtin_trap` in `pair_element_access_size` AND misclassified as STGP in `classify_mte` (L-bit ignored) (P1).**
- Files: `instruction_encodings.c:99-105`, `simulation_output.c:178-182`, `mte_tag_plugin.c:107-110`, `aarch64_generator.py:105`.
- Evidence: `pair_element_access_size` returns 4/8 for size 0/2 and `__builtin_trap()`s otherwise; LDPSW has opc=01 → size==1 → trap (SIGILL → CE death → respawn, no trace). The generator explicitly emits LDPSW (`_LDP_ANY={'ldp','ldpsw'}`). Separately `classify_mte` keys STGP on top byte 0x68/0x69 only; LDPSW shares the same top byte (differs only in L=bit22) → classified `MTE_STGP` → emulated as a tag-store pair → wrong result/trace. The trap is **masked today** because `mte_emulator_hook` intercepts LDPSW as STGP before it reaches the trap — so today's symptom is a silent wrong trace, not a crash.
- Fix (**both edits atomic, never the classify gate alone**): in `pair_element_access_size` add `case size==1 → return 4` (LDPSW reads two 4-byte sign-extended words; keep the trap only for size 3). In `classify_mte` gate the 0x68/0x69 STGP match on `bit22==0`. STGP always has bit22=0; LDPSW always bit22=1 → every STGP form (incl. negative immediates) stays MTE_STGP, LDPSW falls to MTE_NONE and the normal pair path handles it.
- Impact: CE-internal, CE rebuild only, no ABI. Fixing only `classify_mte` would convert the silent-wrong-trace bug into a CE SIGILL — hence atomicity.
- Risk: **low** (with atomic landing).

### Tests
**F-TEST-1 — C test suites and aarch64 type-checking are not wired into CI; CE encoding/ABI/MTE/PAC regressions pass CI unnoticed (P1).**
- Files: `tests/runtests.sh:12` & `24-36`, `contract_executor/Makefile:44`, `executor/Makefile:62`.
- Evidence: `runtests.sh` runs mypy only on `src/*.py src/x86/*.py` — the entire `src/aarch64/*.py` tree (the heaviest ctypes/struct ABI mirrors) is never type-checked, and no C test is compiled/run in CI. The dim-1/dim-3 surface (CE encodings, input-init parser, MTE tagmem, ABI sizes) has zero CI coverage.
- Fix (**corrected** — the proposal's file list is incomplete): (1) the real CI is `.github/workflows/python-lint-and-test.yaml` (ubuntu x86_64), which **never invokes `runtests.sh`** and re-implements the steps inline — it **must also be edited**, else editing `runtests.sh` is a no-op for CI. (2) **Drop the `contract_executor` prereq** from the `test:` target so only the host-portable pure-C binaries build on the x86 runner (the full `contract_executor` embeds AArch64 `base_hook.S` + python-embed and cannot link there). (3) Verify mypy on `src/aarch64/*.py` is clean before gating. Add `test_mte_tagmem` to `make test` (and to `clean:`). Keep `test_pac_auth_verify` out of CI (needs `/dev/executor`).
- Impact: gates the soundness fixes (F-CE-2/3/4, F-ABI-1/2) so regressions fail CI. Confirmed all four C binaries build+pass on this box (test_ce 1171/0, test_ce_integration 382/0, test_mte_tagmem 55/0, test_pmu pass).
- Risk: **medium** (CI-portability + must sequence after the suite is green).

**F-TEST-2 — NI/sealed safety-invariant tests trust the resolver's own `speculative` flag as ground truth; cannot catch F-SEAL-1/F-SEAL-2 (P1).**
- Files: `unit_ni_random.py:100-113`, `unit_regular_sealed.py:124-136`.
- Evidence: the central safety check derives `arch = not r.speculative` from the resolver entry and only flags a violation when `arch && forged`. If an architecturally-reachable slot is mis-classified speculative (exactly F-SEAL-1 MTE-over-placeholder and F-SEAL-2 zero-context AUT*), `arch=False` and a forge/wrong-tag on it is silently accepted; the decoy is compared against the same wrong flag. False confidence in the box-resetting "no forge lands on an arch slot" invariant.
- Fix: add an **independent oracle** — CE-trace over the `genuine()` fill, classify a slot architectural iff its offset is observed at `speculation_nesting==0`; assert this equals `r.speculative` for every entry. Route every `genuine()` AUT* through a **never-faulting** would-succeed check composed from `REVISOR_PAC_SIGN + REVISOR_PAC_XPAC` (`sign16(xpac(signed_ptr),ctx)==signed_ptr`) — **feed the GENUINE ptr/ctx, never route a forged sig through `REVISOR_PAC_AUTH`**. Gate behind the existing `/dev/executor` skip.
- Impact: test-only, no ABI. On HW boxes the new oracle **fails until F-SEAL-2/F-P0-1 land** — that failure is the intended regression signal. Co-commit with the seal fix (or land as the failing repro immediately preceding it) so the tree is never left red.
- Risk: **low** (test-only, box-safe).

**F-TEST-3 — `unit_ni_random` sets deterministic global PAC keys with no `tearDownClass`; leaks into `unit_pacga` (the 2 baseline failures, F-T1) (P1).**
- Files: `unit_ni_random.py:49-67`, `unit_regular_sealed.py:112-119`.
- Evidence: `setUpClass` calls `set_pac_keys(fixed)` with no teardown; the deterministic keys persist in the kernel and leak into `unit_pacga` (alphabetical order: ni_random idx 20 → pacga idx 23) → the documented 2 failures. `unit_pacga` passes 20/20 alone.
- Fix: add `tearDownClass` calling `set_pac_keys(None)` (the documented revert-to-live path, `aarch64_kernel.py:409-415`, done pac-ret-free under IRQ/preempt-off so it cannot FPAC-reset the box). For `unit_regular_sealed` capture `cls._last_ex` inside the `_executor()` helper. **Defense-in-depth:** also call `set_pac_keys(None)` at the **top** of `unit_pacga.setUpModule` (teardown is skipped if `setUpClass` raises `SkipTest` after pinning keys); guard both teardowns with `hasattr`.
- Impact: restores the 373/0/13 baseline that guards every later step. Test-only, no ABI. Does **not** touch the F-ABI-1 72-vs-80 mismatch (these pin keys via the correct 80B `PacKeys` ctypes ioctl, not an input-init section).
- Risk: **low.**

---

## 4. Cross-area findings (spanning python / CE / kernel / arch)

These span ≥2 boundaries and carry the rebuild-coupling that drives the plan ordering.
The **ABI rebuild flags** column states what must be recompiled/reloaded and whether the
dangerous "rebuild BOTH sides or the box resets on stale parse" coupling applies.

| Finding | Boundaries | Rebuild flags | Stale-parse box-reset coupling? |
|---|---|---|---|
| **F-ABI-1** (PAC_KEYS 72→80) | PY + CE + kernel | **PY+CE rebuilt together**; kernel already 80B, no rebuild | **Yes (latent)** — never ship 80B PY against 72B CE. Dormant today. |
| **F-ABI-2** (extra_data_size==0) | PY parser + CE | CE recompile (the `_Static_assert`); PY read-side only | No — read-side identical while field==0. |
| **F-ABI-3** (HTRACE_WIDTH/NUM_PFC dual-defined) | kernel + UAPI + PY | kernel module recompile only | No — ABI-neutral while widths stay 1/3; build *fails* on future divergence. |
| **F-ABI-4** (remote/userland NR16, ABI rot) | userland + PY | userland binary rebuild; mirror MTE struct into a userapi header | No — additive to an ioctl the kernel/PY already implement. |
| **F-CE-6** (trace truncation flag) | CE + PY | **CE rebuild + PY parser LOCKSTEP** (header 8B→16B) | No — CE→PY *response* blob only; can only misparse host-side, never forge an AUT*. |
| **F-ERR-1** (HW htrace error invariant) | PY + kernel | none (PY only) | No. |
| **F-ERR-2** (CE crash recoverable) | PY + CE-proc | none (PY only) | No. |
| **F-CE-1** (spec-window cap/barrier) | PY config + CE | CE rebuild (cap field already transmitted) | No — kernel never parses the CE envelope. |
| **F-CE-3 / F-CE-4** (LDPSW / LDNP / PRFM) | CE + generator | CE rebuild only | No. |
| **F-KERN-1** (TCF) | kernel + CE/PY model | kernel module insmod | No cross-side wire change; insmod itself is reset-class-if-stale. |
| **F-KERN-2** (stale sandbox tags) | kernel + CE model | kernel insmod | No — handles *absence* of the MTE_TAGS section; format untouched. |
| **F-SAFE-1** (forged-AUT* defense-in-depth) | CE + kernel + userland | CE rebuild; kernel comment; userland rebuild | No — CE stops *using* `REVISOR_PAC_AUTH`; no format change. |
| **F-SEAL-2** (zero-ctx AUT*) | PY resolver (+ CE trace) | none (PY only) | No — genuine AUT*s go through the kernel verify-then-auth net. |
| **F-FLOW-1 / F-FLOW-5** (executor/model seam) | shared `fuzzer.py` + x86 + aarch64 | none (PY only) | No. |
| **F-TEST-1** (CI wiring) | CI + Makefiles + mypy | host C-test recompile only | No — throwaway host binaries. |

Highlight: the **only** cross-side wire format actually touched in-plan (PAC_KEYS, F-ABI-1)
is dormant. Every other cross-boundary fix is single-side-rebuild or read-side-only — so
**none of the in-plan steps carry the live stale-parse box-reset hazard.**

---

## 5. Dimension-9 parity findings (accidental aarch64 ↔ x86 divergences)

These are places where AArch64 *accidentally* drifted from the x86 reference and should
converge (x86 is the oracle, never edited):

- **F-ERR-1** — x86 raises `HardwareTracingError` on `htrace==0`/malformed (`x86_executor.py:160-167`); AArch64 silently accepts zeros. **Direction: ARM catches up to x86.**
- **F-CE-1** — x86 `UnicornSpec` terminates a speculative window on a barrier mnemonic AND on `window>model_max_spec_window` (`model.py:874-879`); the CE does neither. **Direction: ARM converges on x86.** (`CONF.model_max_spec_window` must stay — x86 uses it.)
- **F-FLOW-1** — the AArch64 fused-executor 4-tuple/2-tuple return shapes leaked into shared `fuzzer.py`, so the generic loop AttributeErrors/mis-unpacks on x86 (which returns a bare `List[HTrace]` and has no taints method on the Executor). **This is parity *damage* — the fix repairs x86, edits no x86 source beyond the shared file + a stub test.**
- **F-FLOW-2** — common `fuzzer.py` hard-codes the AArch64 CE trace layout (0x2000 GPR region, `range(31)`, `x{n}` labels) and imports `src/aarch64` directly; on x86 the dump prints garbage. **Restore the common/arch boundary via no-op Executor hooks.** (Bonus: `range(31)` is an over-read even on aarch64 — only x0..x5+flags+sp are real.)
- **F-FLOW-6** — `Analyser.filter_violations` gained a `test_cases` param only in impl/caller, not the ABC; an alternative Analyser would be incompatible.
- **F-TRACE-1** — AArch64 `compute_ctrace` claims "like the x86 contract tracer" but records cache-set-only and never PC/control-flow, and ignores `CONF.contract_observation_clause` entirely. **Fix: validate the clause and reject the unimplemented ones; correct the misleading comment — do NOT append PC to "match x86 ct" (it would over-split under the data-only L1D htrace).**
- **F-IR-2** — common `cli.py` imports/calls the AArch64 `print_opcode_summary` (arch leak into the arch-agnostic CLI). The inline comment already says "should be removed."
- **F-IR-5** — the GPR-slot comment in `interfaces.py` (`slot 7 = unused`) is stale on **both** arches: slot 7 is SP (x86 RSP at `x86_model.py:482`, AArch64 sp). The corrected comment is more accurate for x86 too.

### Arch-NECESSARY divergences that are NOT problems (do not "fix" these)
- **NUM_PFC 3 (aarch64) vs 5 (x86)** — by design; the PY `UserMeasurement pfc*3` is correctly arch-specific. F-ABI-3 keeps this.
- **`compute_ctrace` = union bitmap, not x86's ordered `ct` (PC+mem)** — F-TRACE-2 makes this an *intentional* divergence: the AArch64 htrace is a single unordered union bitmap (`HTRACE_WIDTH=1`), so an order-sensitive ctrace has no observable counterpart. The union model is the *correct* one here; reinstate order only if a time-ordered htrace ever arrives.
- **No CE / Unicorn-model on AArch64** — AArch64 uses the contract executor; `factory.get_model` is x86-only and uncalled here. The dead-wiring removal (F-FLOW-5 core) is *not* a parity bug.
- **NI / sealing / branch-mistraining / `_arch_trace` / `trace_test_case_variants_hw`** — AArch64-only by design; x86 uses cross-input equivalence classes, not intra-input variant comparison. No x86 mirror is expected (F-FLOW-3, F-MIS-1/2, F-SEAL-2, F-CE-12).
- **PAC / MTE / IRG / SCTLR.TCF / `executor_userland`** — ARMv8.3/8.5 only, no x86 analog (F-CE-5, F-CE-10, F-KERN-1/2/8, F-SAFE-1, F-ABI-4).
- **AArch64 input-init section table vs x86 raw `/sys/x86_executor/inputs`** — different wire formats by design (F-TEST-5).
- **x86 FP categories (SSE/SSE2) genuinely supported** while F-GEN-3 removes BASE-FPSIMD/SVE/SME from the **AArch64** schema only — do not touch the x86 schema.

---

## 6. Themes (systemic)

1. **"Agree only by coincidence" ABI couplings.** Multiple boundaries are correct only because a value happens to match on both sides: PAC_KEYS (72B==72B but wrong, F-ABI-1), `extra_data_size` (always 0, F-ABI-2), `HTRACE_WIDTH`/`NUM_PFC` duplicated and equal (F-ABI-3), three independent hardcoded `MTE_INITIAL_TAG=6`s (F-KERN-2). The fix pattern is single-source + `_Static_assert` so divergence fails the build.

2. **The common/arch boundary leaks both ways.** AArch64 specifics bleed into shared `fuzzer.py`/`cli.py` (F-FLOW-1/2, F-IR-2) and the shared Executor/Analyser ABCs drift from their implementations (F-FLOW-6, F-FLOW-1). The Unicorn-model wiring is half-dead (`self.model`/`arch_*` declared, never assigned — F-FLOW-5). This is the structurally weakest area and the foundational refactor (step 6) gates many others.

3. **Soundness rests on undocumented, unenforced invariants.** Taint correctness depends on a hand-maintained capstone fixup denylist (F-TAINT-1/2), safe only because the unsound mnemonics are out of `supported_instructions`. The seal net depends on a per-slot `speculative` flag the resolver can mis-set (F-SEAL-2), and the guard tests trust that same flag (F-TEST-2). `kaddr2uaddr` relies on an out-of-CE sandbox clamp with no C-side assertion (F-CE-13). The flags slot is safe via three implicit layers, not one guard (F-MISC-1). Fix pattern: over-approximate (taint), add an independent oracle (seal), document+assert the invariant.

4. **Decode bugs masked by reachability gates, not correctness.** Sign-extending loads, LDPSW, LDNP/STNP, PRFM, LSE atomics, FP/SIMD all have decode/taint bugs that are currently unreachable only because the generator allow-list / blocklist excludes them (F-CE-2/3/4, F-TAINT-1/2, F-GEN-3). They are live the moment the allow-list widens, and the ASM/template path bypasses the allow-list entirely (F-GEN-3).

5. **No defense-in-depth for the box-reset hazards.** Forged-AUT* safety rests *solely* on the kernel verify-then-auth net (F-SAFE-1); MTE sync faulting would run unguarded at EL1 (F-KERN-1); a CE crash or kernel `-EIO` aborts the campaign (F-ERR-1/2). The respawn/skip machinery exists but is not wired to the callers.

6. **CI/test coverage is green-but-empty where it matters most.** C suites + aarch64 mypy not in CI (F-TEST-1); the highest-value seal/decoy/CE-correctness tests are HW-gated or orphaned (F-TEST-4); the seal guard tests are self-referential (F-TEST-2). The part most likely to harbor soundness bugs has the least automated coverage.

7. **Dead/cruft accretion.** Dead predictor `commit()` chain, `/tmp/ce_crash.log` probe, `stdout_print_hook` (F-DEAD-1); orphaned `src/bp/predictors.py` + `src/directed_fuzzing/` (F-DEAD-2); unreachable `_drain_stderr` + duplicated `_run_ce_on_tc` (F-DEAD-3); dead kernel JIT helpers with unbounded NOP-fill writes (F-KERN-4); dead state fields (F-KERN-5); default-args against the project style (F-STYLE-1).

---

## 7. P2 / cleanup (concise)

**ABI:** F-ABI-2 (assert `extra_data_size==0`, fixed 416B stride, `_Static_assert`), F-ABI-3 (single-source HTRACE_WIDTH/NUM_PFC + memcpy `_Static_assert`; drop the "tolerate WIDTH>1 in Python" scope), F-ABI-4 (userland NR16 MTE serve case + ADB `is_file_present`→False + NR-coverage test; defer RemoteHWExecutor revival).

**Common flow:** F-FLOW-2 (move arch dump into no-op Executor hooks; fix `range(31)` over-read), F-FLOW-3 (NI step-2.4 escalation shrinks instead of accumulates the sample — delete the two cumulative-bookkeeping lines), F-FLOW-4 (recompute artifact at detection nesting, not `model_max_nesting`; do NOT uncomment `self.model`), F-FLOW-5 core (delete uncalled `get_model` + never-assigned `arch_*` + dead `model:Model`), F-FLOW-6 (ABC `test_cases` param + indentation), F-FLOW-7 (dead imports; **keep `GPR_SUBREGION_SIZE`** — F-FLOW-2 now uses it).

**CE core/plugins:** F-CE-4 (LDNP/STNP model + PRFM non-access), F-CE-5 (PAC modifier reg-31 SP-vs-XZR — defer, unreachable), F-CE-6 (truncation flag, host-side), F-CE-7 (SWP/CAS after-value from Rs not Rt — cosmetic), F-CE-8 (literal PC-relative load → `is_mem=0`, not trap), F-CE-9 (trampoline callee-saved clobbers), F-CE-10 (drop IRG from MTE generator set), F-CE-11 (bpas guard `!mte_is_mem_tag_access`), F-CE-12 (NULL-instance guards fatal + determinism cold-flag), F-CE-13 (document the sandbox-clamp invariant; **no TBI mask, no hard [0,span) assert**).

**Taint/gen:** F-TAINT-1/2 (merged access==0 src-only fallback + explicit LDG RMW + base.json cross-check), F-GEN-1 (`B.al`/`B.nv` not conditional: `(enc&0xF)<0xE`), F-GEN-2 (load/store pool fallthrough robustness), F-GEN-3 (remove FP/SIMD/SVE/SME from the AArch64 schema + reject in asm_parser; defer the C V-bit work).

**Trace:** F-TRACE-1 (validate observation clause, fix comment), F-TRACE-2 (union-bitmap ctrace).

**Kernel:** F-KERN-2 (per-input tag reset else-branch), F-KERN-3 (check `execute_result` before `TRACED_STATE`), F-KERN-4 (delete dead JIT helpers / bound NOP-fill), F-KERN-5 (delete dead state fields), F-KERN-6 (reject `length&3`), F-KERN-7 (save/restore SSBS), F-KERN-8 (mask tag to 4 bits at `tag_ptr`), F-KERN-9 (drop unused cycle-counter enable), F-KERN-10 (BHB branch-count claim — defer, microarch-dependent low-confidence).

**Safety/mistraining/postproc:** F-SAFE-1 (non-faulting CE auth default + userland cmd-14 gate + kernel comment), F-MIS-1 (correct the false mistraining-disable comment — no polarity bug), F-MIS-2 (NI decoy re-trace guard as skip+STAT), F-POST-1 (warn+suppress fully-fenced output), F-POST-2 (doc-only: ARM spec-source is BR_MIS_PRED-only; `AddViolationCommentsPass` x86-only no-op).

**Common-IR/glue/style/dead:** F-IR-1 (config bounds), F-IR-3 (delete dead `constraints` parse), F-IR-4 (`actor_id*itemsize` 0x3000 not 0x4000), F-IR-5 (GPR-slot comment), F-IR-6 (delete `MOD2P64`, consolidate 2^64 const), F-IR-7 (`Input.load` size validation), F-GLUE-1 (assert `_sandbox_base` resolved), F-STYLE-1 (drop default args / `clear_pac_keys()`), F-MISC-1 part1 (NZCV flags assert; **defer part2 RNG skip-set**), F-DEAD-1/2/3, F-PERIPH-1 (`read_file` read-until-EOF).

---

## 8. CONSEQUENCE ANALYSIS — inter-fix conflicts and how reconciliation resolved them

**Hard merges (two findings are one change):**
- **F-FLOW-1 + F-FLOW-5 → step 6.** F-FLOW-1's proposed fix *is* the F-FLOW-5 wiring; separately they would edit `initialize_modules`/`is_architectural_mismatch` in conflicting ways. Also remove `model:Model` from the Fuzzer ABC (F-FLOW-5's own verify grep missed it).
- **F-ABI-2 + F-CE-6 → step 9.** Both edit `destroy_trace_log`/`log_sim_state` and the PY trace parser. Bump the `contract_trace_t` blob format **once** (8B→16B header) instead of twice.
- **F-ABI-3 + F-TRACE-2 → steps 17→18.** Both edit `measurement.h:5-11`; F-TRACE-2 reinterprets the htrace word as a union bitmap, F-ABI-3 single-sources the width. Sequence together, do not let both rewrite those lines independently.
- **F-TAINT-1 + F-TAINT-2 → step 16.** Both rewrite the same `decode_reg_accesses` access==0 fallback and the same test; implement one shared fallback + explicit LDG RMW + one base.json cross-check.
- **F-TRACE-2 + F-TRACE-1 → step 17.** Both rewrite `compute_ctrace` + `TestComputeCtrace`. F-TRACE-2 (union granularity) lands first; F-TRACE-1 layers observation-clause validation on top.

**Direct contradiction resolved:**
- **F-FLOW-2 vs F-FLOW-7 over `GPR_SUBREGION_SIZE`.** F-FLOW-7 lists it as a dead import to delete; F-FLOW-2 *starts using* it to derive the GPR dump offset/width. **Resolution: keep `GPR_SUBREGION_SIZE` imported; F-FLOW-7 drops only ChiSquaredAnalyser/InputFragment/EquivalenceClass/reference_htraces.**

**Sequencing dependencies (one fix's premise depends on another):**
- **F-CE-2 before F-CE-3/F-CE-4/F-CE-11.** F-CE-2 corrects the per-class `is_load`/`is_store` helpers the others build on; the naive global redefine would regress the pair path, so the class-aware decode must land first.
- **F-CE-8 before F-CE-13.** F-CE-8 defines the literal-PC-relative path as the documented *exception* to the sandbox-clamp invariant F-CE-13 narrates.
- **F-GLUE-1 before F-SEAL-2.** F-SEAL-2's genuine re-trace requires a resolved `_sandbox_base`; F-GLUE-1's assert is the prerequisite.
- **F-ERR-2 before F-CE-6's "truncated==1 is a hard error".** Without F-ERR-2's skip plumbing, treating truncation as a hard error would abort the campaign rather than discard one input.
- **Seam (step 6) before F-ERR-1/F-ERR-2 (step 7) and the flow cleanups (step 8).** They wrap/restructure the same call sites; applying them first forces a re-wrap over the new helper shape.
- **F-DEAD-1 before F-CE-12.** F-DEAD-1 removes the dead `commit()` chain; F-CE-12 then hardens only the surviving checkpoint/rollback NULL guards.

**Scope reductions (a "tempting" variant rejected as harmful):**
- **F-CE-2** "replace is_load/is_store accordingly" → restricted to the regular path (the global redefine misclassifies pre-indexed STP as a load).
- **F-CE-1** "delete the unused arg" → keep `CONF.model_max_spec_window` (x86 uses it); wire the field through instead.
- **F-CE-13** TBI mask + hard bounds assert → **rejected** (kbase top byte is 0xFF; the assert false-trips legitimate sub-kbase pre-index). Documentation-only.
- **F-CE-10** option 1 (program GCR_EL1.Exclude=0, define lowest-tag as the contract) → **rejected as physically unrealizable** (HW IRG stays RNG-random regardless of Exclude). Adopt option 2: drop IRG from the generator set.
- **F-CE-8** `__builtin_trap` → `is_mem=0` (a trap aborts the campaign until F-ERR-2).
- **F-SAFE-1** "re-issue REVISOR_PAC_AUTH when resigned==ptr" → use the always-return-canonical `debug_auth_no_fault` (provably contract-identical, no real EL1 AUT*).
- **F-MIS-2** hard `raise` → skip-input + STAT (else it recreates F-ERR-2's campaign-abort and fights F-FLOW-3's escalation loop).
- **F-MISC-1** part2 (RNG skip-set) → **deferred** (a literal rng-draw skip shifts the shared stream, changes slot-7/SP bytes, breaks input reproducibility and x86 parity).
- **F-ABI-3** "tolerate WIDTH>1 in Python" → **dropped** (downstream consumes a single 64-bit int; true WIDTH>1 is an F-TRACE-2-sized change).

**Safety/remit deferrals (cannot be validated on this box / out of remit):**
- **F-KERN-1 sync enablement** — DEFERRED behind the MTE-fidelity fixes + a new EL1 tag-fault recovery net (the bit-position fix lands; activation does not).
- **F-FLOW-5 x86 deletion tail** (X86ArchitecturalFuzzer/X86ArchDiffFuzzer/ArchitecturalFuzzer + factory routes) — DEFERRED; edits x86 source, removes two config fuzzer modes, breaks `acceptance.bats`, undetectable locally.
- **F-ABI-4 parts 2 & 4** (RemoteHWExecutor revival, `contents()`) — DEFERRED; never-instantiated dead code, needs a project-direction decision and HW.
- **F-CE-5** — DEFERRED; reg-31-as-PAC-modifier is unreachable (SP blocklisted from the generator pool).
- **F-KERN-10, F-MIS-1 enablement, F-CE-1 barrier-class-matching, F-CE-6 realloc, F-POST-2 arch-trace-diff** — DEFERRED pending HW data or to avoid re-adding `_arch_trace` coupling to shared code.

---

## 9. RECONCILED FIX PLAN (dependency-ordered)

`rebuild` = a kernel insmod and/or CE/userland recompile is required for the fix to take
effect. **The dangerous "rebuild BOTH sides or stale-parse resets the box" coupling
applies to NONE of these steps** (PAC_KEYS, the only cross-side wire change, is dormant;
all others are single-side or read-side-only). `∥` = files disjoint from other in-flight
steps, develop concurrently.

| Step | Fix ids | Depends on | Key files | rebuild | ∥ |
|---|---|---|---|---|---|
| 1 | F-TEST-3 | — | unit_ni_random / unit_regular_sealed / unit_pacga | no | yes |
| 2 | F-GLUE-1 | — | aarch64_executor.py, aarch64_contract_executor.py | no | yes |
| 3 | F-ABI-1 | 1 | input_wire.py, simulation_input.h, pac.h, test_ce.c, unit_input_wire.py | **CE** | no |
| 4 | F-SEAL-2 (+F-P0-1) | 2 | aarch64_sealer.py, aarch64_executor.py | no | no |
| 5 | F-TEST-2 | 4 | unit_ni_random, unit_regular_sealed | no | no |
| 6 | **F-FLOW-1 + F-FLOW-5** (core) | — | fuzzer.py, interfaces.py, factory.py, aarch64_fuzzer.py, unit_fuzzer.py | no | no |
| 7 | F-ERR-1 + F-ERR-2 | 6 | aarch64_executor.py, aarch64_kernel.py, aarch64_contract_executor.py, fuzzer.py | no | no |
| 8 | F-FLOW-2/4/3/6/7 | 6 | fuzzer.py, interfaces.py, analyser.py, aarch64_executor.py, x86_executor.py | no | no |
| 9 | **F-ABI-2 + F-CE-6** | 7 | simulation_output.{c,h}, aarch64_trace.py, aarch64_contract_executor.py | **CE** | no |
| 10 | F-CE-2 | — | simulation_output.c, simulation_hook.c, test_ce.c | **CE** | no |
| 11 | F-CE-3 + F-CE-4 + F-CE-11 | 10 | simulation_output.c, instruction_encodings.c, mte_tag_plugin.c, execution_clause_bpas.c, test_ce.c, test_mte_tagmem.c | **CE** | no |
| 12 | F-CE-8 + F-CE-13 + F-CE-7 | 11 | simulation_output.c, simulation_hook.c, aarch64_generator.py, test_ce.c | **CE** | no |
| 13 | F-CE-1 (a now / b after seal) | 2,4,7 | simulation_execution_clause_hook.{c,h}, instruction_encodings.c, aarch64_executor.py, test_ce_integration.c | **CE** | no |
| 14 | F-DEAD-1 → F-CE-12 → F-CE-9 | — | main.c, tage_py.c, neoverse_n3_bpu.c, simulation_hook.c, branch_predictor.h | **CE** | no |
| 15 | F-CE-10, F-GEN-3, F-GEN-2, F-GEN-1 | — | aarch64_config.py, aarch64_asm_parser.py, generator.py, aarch64_disasm.py | no | yes |
| 16 | **F-TAINT-2 + F-TAINT-1** | — | aarch64_disasm.py, unit_disasm_reg_access.py, unit_taint.py | no | yes |
| 17 | F-TRACE-2 → F-TRACE-1 | — | aarch64_trace.py, aarch64_config.py, config.py, unit_taint.py | no | no |
| 18 | F-ABI-3 | 17 | measurement.h, executor_user_api.h, chardevice.c, aarch64_kernel.py | **kernel** | no |
| 19 | F-KERN-2 + F-KERN-8 | — | measurement.c, mte.c, aarch64_mte.py | **kernel** | no |
| 20 | F-KERN-3/6/7/9/4/5 | — | chardevice.c, measurement.c, pmu.c, jit.c, executor.h, main.c | **kernel** | no |
| 21 | F-SAFE-1 + F-ABI-4 (1/3/5) | 4,14 | pac_sign_plugin.c, measurement.c, executor_userland.c, aarch64_connection.py, executor_ioctl_nr.h | **CE+userland** | no |
| 22 | F-MIS-1 → F-TEST-6 → F-MIS-2 | 4,7 | aarch64_executor.py, aarch64_config.py, unit_mistraining_gate.py, fuzzer.py | no | no |
| 23 | F-POST-1 + F-POST-2 | 13,8 | postprocessor.py, aarch64_fuzzer.py | no | no |
| 24 | F-IR-1/3/4/5/6/7/2 | 8 | config.py, isa_loader.py, postprocessor.py, interfaces.py, util.py, cli.py | no | yes |
| 25 | F-STYLE-1, F-MISC-1(p1), F-DEAD-2, F-DEAD-3 | 2,7,13 | aarch64_kernel.py, aarch64_executor.py, aarch64_input_layout.py, src/bp/predictors.py, aarch64_contract_executor.py | no | no |
| 26 | F-TEST-4 + F-TEST-5 + F-PERIPH-1 | 3 | unit_regular_sealed, unit_pac_mistraining, unit_input_wire, test_ce.c, executor_userland.c | **CE** | no |
| 27 | F-TEST-1 | 1,22,26 | runtests.sh, .github/workflows/python-lint-and-test.yaml, both Makefiles | **host C** | no |

**Global ordering rationale.** P0 security/correctness first (steps 1–5: green baseline,
glue prereq, PAC-keys ABI, zero-context AUT* soundness, its oracle). Two hard dependencies
pull structure earlier than its own severity: F-GLUE-1 (P2) precedes the P0 F-SEAL-2, and
the `FuzzerGeneric` seam refactor (step 6) precedes the P1 error-handling (step 7) and the
flow cleanups (step 8) because they wrap/restructure the same call sites. Then ABI/fidelity
(steps 9, 17–18), the strictly-serial CE-decoder chain (10→11→12→13, shared C files),
CE cleanup/hardening (14), kernel MTE-fidelity + IO (19–20), safety/remote (21), parity
(17), taint/gen (15–16), coupling/structure/style/dead-code (8/24/25), and **tests + CI
wiring last** (26–27) so CI gates a green suite.

### Composition with the agreed seal-subsystem plan (`REVIEW_FINDINGS.md`)
The user already chose "everything, in order" for the seal plan; it is a **strict prefix**
of this plan:
- seal **F-T1 = F-TEST-3 = step 1**; seal **F-P0-3 = F-ABI-1 = step 3**;
- seal **F-P0-1** (MTE spec classified over the genuine-tag re-trace) is **not a separate id** in the surviving set — it is implemented **together with F-SEAL-2 in step 4** (both restructure `MtePacSealedTestCase.resolve` staging); seal **F-P0-2 = F-SEAL-2 = step 4**;
- seal **E2 = F-GLUE-1 = step 2**; the regression gate is **F-TEST-2 = step 5**;
- seal `[T1]=F-ABI-2` (step 9), `[E3]/[E4]/[L1]` aarch64_log delete and `[PA1]` (steps 14/25), and the `[T2]` trace split + `seal/` subpackage + renames belong with the **dedicated mechanical-move step** (fold into step 25's structure work or run as a guarded follow-on, after all seal correctness fixes per the seal plan's stated order).
- **Net: run seal-plan steps 1–5 here exactly as `REVIEW_FINDINGS.md` orders them, then continue with the broader engine fixes.** Step 4 must re-verify against `simulation_hook.c` first (the REVIEW_FINDINGS C cross-check flags F-P0-1/F-P0-2 as possible over-flags).

---

## 10. Deferred items + coverage gaps

**Deferred (cannot be validated on this box or out of remit):**
- F-KERN-1 synchronous enablement (needs an EL1 tag-fault recovery net + the MTE-fidelity fixes).
- F-FLOW-5 x86 fuzzer-mode deletion tail (edits x86, breaks `acceptance.bats`).
- F-ABI-4 parts 2 & 4 (RemoteHWExecutor revival, `contents()` — dead code + project-direction call).
- F-CE-5 (reg-31 PAC modifier — unreachable; SP blocklisted).
- F-KERN-10 (BHB branch-count — microarch-dependent, low confidence, HW-only).
- F-MISC-1 part 2 (RNG skip-set — shifts the shared stream, breaks reproducibility).
- F-POST-2 optional CE-arch-trace-diff analysis (would re-add `_arch_trace` coupling to shared code).
- F-MIS-1 enablement (comment ships; flipping the flag needs HW efficacy confirmation).
- F-CE-1 barrier-class-matching variant & F-CE-6 realloc-on-demand (need HW data / unnecessary once the cap+flag land).

**Coverage gaps (verification we could not perform under the safety constraint):**
- No HW measurement, MSR pokes, forged AUT*, or `insmod` were run — every step was reasoned/verified statically. `REVISOR_TRACE` → ENODEV on this box; HW measurement is unavailable.
- The seal/decoy/CE-correctness/mistraining tests are HW-gated or orphaned (F-TEST-4): on a kernel-less box the suite is green-but-empty for exactly the soundness-critical paths. F-TEST-4 proposes a recorded-fixture `PacSigner` to run them offline.
- The C test suites are not in CI (F-TEST-1) — the CE decoder fixes (steps 10–14) must be validated by running `test_ce`/`test_mte_tagmem` **manually** until step 27 lands.
- mypy on `src/aarch64/*.py` has never been run (mypy not installed on this box); it may surface pre-existing type errors that must be cleared before it can gate CI.
- F-KERN-1's correctness can only be confirmed via the non-destructive `printk` SCTLR readback; the actual fault behavior cannot be exercised without resetting the VM.
