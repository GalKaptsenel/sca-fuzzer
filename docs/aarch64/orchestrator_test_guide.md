# AArch64 Revizor — Orchestrator VM: Test & Investigation Guide

**Audience:** the Claude session on the **orchestrator VM** — a Neoverse-N3 *virtual machine* that
generates test cases + inputs and runs the **architectural** half of Revizor.

**The one constraint that defines this machine:** it is a VM, so **microarchitectural state cannot be
measured** (no meaningful PMU counters, no cache Prime+Probe / Flush+Reload, no leak htraces). But it is
a *real N3*, so **everything architectural is faithful**: instruction semantics, the contract executor
(CE), the assembler, PAC/MTE *architectural* effects, the generator, and the whole Python model.

➡ **Therefore: the orchestrator's job is to prove every change we made is _architecturally correct_.**
The actual leak measurement (contract-vs-hardware comparison) happens later, on a bare-metal N3 — see
`server_verification_guide.md` for that tier; do not attempt it here.

> Branch: `review/engine-fixes`. All paths are repo-relative.

---

## 0. TL;DR — run this

```bash
cd src/aarch64
./install_revizor_env.sh --build                 # deps + venv + build the 4 C artifacts
# Tier A — the architectural suite (everything below MUST pass on the VM):
python3 tests/unit_saturating_bp.py                                  # BPU model (pure Python)
python3 -m unittest discover -s tests/aarch64_tests -p "unit_*.py" -t tests/aarch64_tests -v
python3 -m unittest discover tests -p "unit_*.py" -v                 # core/top-level
( cd src/aarch64/contract_executor && make test )                    # test_ce + test_ce_integration
( cd src/aarch64/asm_to_bytes && make && ./asm_to_bytes --help )     # in-memory assembler
```

If any **Tier A** check fails, that is a real regression — investigate (see §4). HW-gated tests
self-skip when `/dev/executor` is absent; that is expected on the VM, not a failure.

---

## 1. Why the CE runs here without the kernel module

The CE used to `open("/dev/executor")` and `abort()` at startup, which would have blocked the entire
CE layer on a module-less VM. That open is now **lazy** (`pac_sign_plugin.c: pac_executor_fd`): the
device is opened only when a test case *actually executes a PAC instruction*, and aborts loudly only
then. So:

- Test cases with **no PAC** → CE runs fully **module-free**. This covers `test_ce_integration` and
  ordinary generated test cases.
- A test case that **does** execute PAC needs the kernel's EL1 keys → it will abort if the module is
  absent. PAC is *architectural*, not µarch — so it runs on the VM **once the module is loaded** (see
  §3c), not only on bare metal.

---

## 2. Tier A — runs on the VM (the sound, complete architectural suite)

| What | Command | Pass criteria | Oracle |
|---|---|---|---|
| **BPU model** (TAGE/PHR, bimodal, N3 specifics) | `python3 tests/unit_saturating_bp.py` | all tests pass | TAGE algorithm + N3 RE facts (tests are tagged `(TAGE)` vs `(RE)`) |
| **CE unit tests** | `cd src/aarch64/contract_executor && make test` → `./test_ce` | `0 failed` | hand-derived encodings / ARM ARM |
| **CE integration** (black-box, fork+exec CE) | same `make test` → `./test_ce_integration` | `0 failed` | architectural semantics |
| **Python logic suite** | `unittest discover` (both lines in §0) | pass; HW-gated self-skip | spec/logic |
| **NI generation** (PAC/MTE non-interference test-case generation) | `python3 -m unittest -v unit_ni_factory unit_pac_generation unit_pac_mistraining unit_mte_generator unit_mte_taint` (from `tests/aarch64_tests`) | all pass | generation logic |
| **asm_to_bytes** | `cd src/aarch64/asm_to_bytes && make` | assembles; empty `.text` → nonzero exit | cross-`as` output |

**Pure-Python tests that must pass anywhere** (no HW): `unit_saturating_bp`, and under
`tests/aarch64_tests/`: `unit_isa_downloader`, `unit_isa_extractor`, `unit_minimizer`,
`unit_mistraining_gate`, `unit_ni_factory`, `unit_nzcv`, `unit_pac_generation`, `unit_pac_mistraining`,
`unit_sandbox`, `unit_taint`, `unit_executor`, `unit_asm_parser`,
`unit_asm_parser_cache`, `unit_mte_executor`, `unit_kernel_retry`, `unit_kernel_local`,
`unit_patch_pass`, `unit_mte_generator`, `unit_mte_taint`, `unit_connection`, `unit_target_desc`,
`unit_disasm_reg_access`, `unit_stream_ipc`; plus top-level `unit_arch_isolation`, `unit_config`,
`unit_postprocessor`, `unit_docs`.

**Needs the built CE but no module** (now runnable on the VM thanks to §1): `unit_ce_mem_model.py`,
`unit_sandbox_ce.py` (need cross-`as`/`objcopy`). `unit_pac_generator.py` exercises the PAC pipeline —
its PAC-executing parts need the kernel module, so expect partial skips on the VM.

---

## 3. The other two tiers

### 3a. Needs the kernel module, but only architectural — runs on the VM once the module is loaded
PAC sign/auth/strip use the EL1 key registers, so they need `revizor-executor.ko` — **not** µarch. Load
it on the VM (§3c) and these run there:
- CE PAC tests: `test_integration_pac_pacga`, `_sign_then_strip`, `_sign_then_auth` (they SKIP without
  `/dev/executor`; run once it exists).
- `tests/aarch64_tests/unit_pacga.py` — PACGA ioctl correctness/determinism/key-isolation.
- The PAC-executing parts of `unit_pac_generator.py`.

### 3b. Needs real µarch — bare metal only (DO NOT expect these on the VM)
PMU + cache measurement; on a VM they self-skip or are meaningless. Hand to the bare-metal session
(`server_verification_guide.md`):
- `unit_kernel_module.py`, `unit_mte_random.py`, `unit_code_base.py`.
- Prime+Probe / Flush+Reload measurement, the PMU oracle, the contract-vs-hardware leak campaign.
- The disabled *mismatched-key* AUT* tests (a failing auth is fatal on FEAT-FPAC).

### 3c. Enable & check the complete PAC system on the VM
PAC needs FEAT_PAuth (a real N3 has it) + the module loaded:
```bash
grep -oE 'paca|pacg' /proc/cpuinfo                 # paca = address auth, pacg = generic (PACGA)
sudo insmod src/aarch64/executor/revizor-executor.ko
sudo chmod 777 /dev/executor                        # exposes ioctls incl. 11-16 (PAC/MTE)
```

**Check PAC works** — three independent ways:
1. **Through the CE** (the contract path):
   ```bash
   ( cd src/aarch64/contract_executor && make test )   # the 3 PAC tests now run instead of SKIP
   ```
   Expect `test_integration_pac_pacga / _sign_then_strip / _sign_then_auth` to pass. `_sign_then_auth`
   is the real sign→authenticate round-trip; if the VM *resets* during it, the auth failed (the flagged
   `pac.c` key path) — recoverable here, report it.
2. **By hand** via the `executor_userland` tool (now drives the full PAC interface, ioctls 11-15):
   ```bash
   executor_userland /dev/executor 12                                   # GET keys
   executor_userland /dev/executor 13 pacia 0xffff000080000000 0x0      # PAC_SIGN  → prints signed ptr
   executor_userland /dev/executor 15 xpaci <signed-ptr>                # PAC_XPAC  → back to original
   executor_userland /dev/executor 14 autia <signed-ptr> 0x0           # PAC_AUTH  (matched → recovers)
   ```
3. **Complete PAC e2e generation pipeline** — generate a PAC test case, run it through the CE to capture
   the real signatures, then apply stage-2 instrumentation:
   ```bash
   ( cd tests/aarch64_tests && python3 -m unittest -v unit_pac_generator )   # stage1 → CE → stage2
   ```
   `unit_pac_generation` covers the generation layer (no CE); `unit_pac_generator` is the **full
   pipeline** and needs the built CE + the loaded module — the closest thing to an end-to-end PAC run
   the VM can do (the hardware leak-measurement step stays on bare metal).

**Two key modes:**
- **Default (recommended first):** if `SET_PAC_KEYS` (11) is never called, the module signs/auths with
  the kernel's *live* keys — sign and auth stay consistent, a matched round-trip authenticates, and
  **no key swap happens** (the flagged `pac.c` swap path is not even triggered).
- **Deterministic keys (complete system):** `executor_userland /dev/executor 11 <10 hex words>` pins a
  known key set, making signatures reproducible. This makes each op save→load-your-keys→run→restore,
  which **does** exercise the flagged swap path. `executor_userland /dev/executor 11` (no args) reverts
  to live keys.

---

## 4. Per-change verification matrix (what each change we made should prove here)

| Change (subsystem) | Run | Expect | If it fails — investigate |
|---|---|---|---|
| **CE speculative-PHR + BPU driver** | `./test_ce` (Group 42 `cond_branch_is_taken`) + `./test_ce_integration` (BPU input-selected / traps-without-predictor) | pass | direction must come from the condition, not `target==pc+4`; check `branch_speculation.c`, `execution_clause_bpu.c` |
| **BimodalBP + PHR fold** | `python3 tests/unit_saturating_bp.py` | pass | `saturating_bp.py`: predict purity, fold width, tag mask |
| **MTE emulation** (SUBPS/SUBP/ADDG/SUBG) | `./test_ce_integration` (MTE group) | NZCV + results match ARM ARM | `mte_tag_plugin.c` classify/emulate; compare to DDI0487 pseudocode |
| **`cond_branch_is_taken`** | `./test_ce` Group 42 | pass | `cond_branch_is_taken()` resolves direction from the register/condition test |
| **instruction_encodings UB fix** | `./test_ce` (sign-extension groups) | pass | `imm19*4` / `imm14*4`, signed offsets |
| **PAC lazy-open** (this work) | `./contract_executor` on a no-PAC TC, **module absent** → starts; `make test` passes | CE does not abort without the module | `pac_sign_plugin.c: pac_executor_fd` is only reached by a PAC instruction |
| **PAC sign/auth/strip** (needs module) | `make test` PAC group + `executor_userland … 12/13/14/15` | pacga deterministic; sign→strip & sign→auth recover the pointer | `pac_sign_plugin.c` + kernel `pac.c`; a VM reset = the flagged key-swap path |
| **PAC/NI test generation** | `unit_pac_generation` + `unit_pac_generator` (e2e) + `unit_ni_factory` | pass | `aarch64_generator.py` PACInstrumentation; `unit_pac_generator` needs CE+module |
| **generator: B.al/B.nv, Q-reg, capstone under-report, MTE-taint** | `unit_target_desc`, `unit_disasm_reg_access`, `unit_patch_pass`, `unit_mte_taint` | pass | the matching Python module |
| **asm_to_bytes cross-prefix + empty `.text`** | `make` + assemble a sample; empty `.text` | bytes match Python path; empty → nonzero exit + stderr | must exec `aarch64-linux-gnu-as`, not host `as` |

---

## 5. Investigation playbook (when a Tier-A test fails)

1. **Reproduce in isolation.** Run the single test (`./test_ce` / `./test_ce_integration` print the
   failing `FAIL func:line cond`; for Python, run the one `unittest` case `-v`).
2. **Confirm it is real, not environmental.** Missing `aarch64-linux-gnu-as`/`objcopy`,
   missing `python3-dev`, or a stale build cause spurious failures — rebuild clean (`make clean && make`).
3. **Check the oracle, not the code.** For a CE integration failure, re-derive the expected value from
   the ARM ARM (DDI0487) pseudocode — do not "fix" the test to match the CE. A mismatch means the CE is
   wrong, the test is wrong, or the encoding is wrong — decide which against the spec.
4. **Bisect to a commit.** `git log main..review/engine-fixes` lists the changes; the matrix in §4 maps
   each to its test. `git stash` / checkout to localize.
5. **Loud failure, never a workaround.** Per project methodology: never paper over a failure with a
   fallback. If a test exposes a real defect, file it (and add a regression test) rather than silencing it.
6. **Verify by debugging.** Print CE state (it dumps native + simulated state on crash to
   `/tmp/ce_crash.log`); add a focused `EXPECT` rather than guessing.

---

## 6. Soundness — what each oracle actually proves

- **CE integration MTE tests** — oracle is the **ARM ARM (DDI0487)**: the expected NZCV/results are
  derived from the architecture, so a wrong emulation is caught (they are *not* re-derived from the CE
  code). They cover only deterministic instructions; IRG (random tag) and tag-stores are CE modelling
  reductions and are intentionally not asserted.
- **BPU model tests** — oracle is the TAGE algorithm + N3 reverse-engineering; each test is tagged so
  you can tell a hardware fact `(RE)` from a generic property `(TAGE)`. Note the PHR **fold/tag widths
  are placeholders, not yet RE'd** — do not treat those constants as ground truth.
- **Python logic tests** — oracle is the spec / intended behaviour; many are regressions for specific
  audited defects.

The honest gap: these prove *architectural correctness*. They do **not** prove the tool detects real
leaks — that is the bare-metal measurement tier.

---

## 7. After verifying — report back

Summarize: which Tier-A checks passed, any failures with the exact `FAIL` line + your spec-based
diagnosis, and anything environmental (missing toolchain). Keep the bare-metal items (§3) explicitly
listed as *not attempted here* so the next tier knows what is still open.
