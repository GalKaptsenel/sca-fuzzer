# AArch64 Revizor — Server Verification Guide (branch `review/engine-fixes`)

**Audience:** the Claude session running on the Linux **ARM (Neoverse N3)** box.
**Why this exists:** the recent audit + fixes (and the new BPU speculative-PHR feature) were made
**logic-only on a Windows dev box** that cannot compile or run any of the C / kernel / contract-executor
/ asm code. The Python BPU *model* is unit-tested on the dev box; the C driver that uses it, the kernel
module, and everything Arm-specific are **build-and-HW pending**. This guide lets you reproduce every
check and confirm the work actually builds and behaves on real hardware.

> The branch `review/engine-fixes` is **72 commits ahead of `main`** (2026‑06‑18 → 2026‑06‑22) and is
> **not pushed** — make sure you are on it. All paths are repo‑relative.
>
> **Trust but verify:** some file:line references below come from audit notes that are a few days old —
> re-read the current code before asserting a line number. Where a sub-agent claimed a defect, confirm
> it against the live tree (two such claims this session — a "TAGE-wiped-mid-run" bug and a
> "malformed `/*` comment" — turned out to be false positives).

---

## How to use this guide
1. **§1 Build everything.** If any artifact fails to build, stop and report — most checks depend on it.
2. **§2 Run the test suites** (pure-Python → C tests → CE-driven → kernel-module-on-HW).
3. **§3 Functional probes** for fixes that have no unit test (MTE/PAC/scheduling/JIT/chardevice/…).
4. **§4 Per-subsystem verification checklist** (the server-verify commits, what each must prove).
5. **§5 Known-flagged (NOT fixed)** — be aware; do not expect these to pass.
6. **§6 Crash suspects & priorities**, **§7 ioctl/struct appendix**.

---

## 1. Build everything

The single source of truth is the installer/test-runner **`src/aarch64/install_revizor_env.sh`**
(guards `uname -m == aarch64`). It installs deps, makes a venv at `~/revizor/revizor-venv`, and can
build all four binaries.

```bash
cd src/aarch64
./install_revizor_env.sh --build          # deps + venv + build all 4 artifacts + module
# (then, to run the full suite incl. HW tests:)
./install_revizor_env.sh --test aarch64-all
```

apt deps it installs: `gcc make libc6-dev libelf-dev python3-dev` (build), `python3 python3-venv`
(runtime), `linux-headers-$(uname -r)` if available. **Not** installed but needed by two tools:
GNU cross-binutils **`aarch64-linux-gnu-as` / `aarch64-linux-gnu-objcopy`** (see asm_to_bytes /
executor_userland) — install `binutils-aarch64-linux-gnu` if missing.

Python env (pinned `requirements.txt`): `capstone==5.0.6`, `numpy==1.26.4`, `scipy==1.16.2`,
`PyYAML==6.0.3`, `unicorn==1.0.3`, `paramiko==4.0.0`, `pure_python_adb==0.3.0.dev0`, `pyelftools`,
`xxhash`, `joblib`, `matplotlib`, `cachetools`.

| Artifact | Directory | Build | Output / run |
|---|---|---|---|
| **Kernel module** | `src/aarch64/executor/` | `make -C src/aarch64/executor CC=<native-gcc> KDIR=/lib/modules/$(uname -r)/build` | `revizor-executor.ko`. `sudo insmod …`; `sudo chmod 777 /dev/executor`. Self-creates `/dev/executor` + `/sys/executor/`. MTE auto-detected from `/proc/cpuinfo` (`HW_SUPPORTS_MTE=y\|n` to override). **Use a native `CC` that matches the running kernel** (installer's `detect_compiler`), not the Makefile's default `aarch64-linux-gnu-`. |
| **Contract executor (CE)** | `src/aarch64/contract_executor/` | `make` (and `make test`) | `contract_executor`. **Embeds CPython** (`python3-config --embed`, `-march=native`) — needs `python3-dev`, must be built **on** the box. At runtime loads `bootstrap_director.py` + `saturating_bp.py` from its own dir (resolved via `/proc/self/exe`). |
| **executor_userland** | `src/executor_userland/` | `make -C src/executor_userland CC=<gcc>` | `executor_userland` (static). Manual `/dev/executor` driver (ioctls + `w file` / `r file`); also the remote on-device driver. Needs userapi headers + `aarch64-linux-gnu-gcc`. |
| **asm_to_bytes** | `src/aarch64/asm_to_bytes/` | `make -C src/aarch64/asm_to_bytes` | `asm_to_bytes`. In-memory assembler; **execs `aarch64-linux-gnu-as` at runtime** (must be on PATH). |

**Build gate:** if the kernel module or CE fails to build, stop here — almost everything downstream
depends on them.

---

## 2. Test suites

Invocation (from repo root; the aarch64 dir is a namespace package — both `-s` and `-t` needed):
```bash
# AArch64 Python unit tests (HW-gated ones self-skip when /dev/executor or /sys/executor is absent):
python3 -m unittest discover -s tests/aarch64_tests -p "unit_*.py" -t tests/aarch64_tests -v
# Core/top-level unit tests (arch-agnostic + aarch64 config/isolation):
python3 -m unittest discover tests -p "unit_*.py" -v
# The BPU model (pure Python, no scipy/no src):
python3 tests/unit_saturating_bp.py
```

### 2a. Pure-Python — must PASS anywhere (no HW)
`tests/unit_saturating_bp.py` (BPU model — **63 tests**, the new speculative-PHR + bimodal + N3 specifics),
and under `tests/aarch64_tests/`: `unit_isa_downloader`, `unit_isa_extractor`, `unit_minimizer`,
`unit_mistraining_gate`, `unit_ni_factory`, `unit_nzcv`, `unit_pac_generation`, `unit_pac_mistraining`,
`unit_sandbox`, `unit_taint`, `unit_executor`, `unit_asm_parser`,
`unit_asm_parser_cache`, `unit_mte_executor`, `unit_kernel_retry`, `unit_kernel_local`,
`unit_patch_pass`, `unit_mte_generator`, `unit_mte_taint`, **`unit_connection`** (SSH/ADB mocked),
**`unit_target_desc`** (branch/call classification, B.al/B.nv), **`unit_disasm_reg_access`** (needs
`capstone`), **`unit_stream_ipc`** (logic class). Plus top-level `tests/unit_arch_isolation.py`
(an aarch64 run must load **no** x86/Unicorn modules — important guard), `tests/unit_config.py`,
`tests/unit_postprocessor.py`, `tests/unit_docs.py`.

### 2b. CE C tests — `make test` (needs only the C toolchain + python3-dev for the CE binary)
```bash
cd src/aarch64/contract_executor && make test     # builds + runs test_ce and test_ce_integration
```
- `test_ce.c` — 42 unit groups, **no subprocess, no /dev/executor**: branch classify/target/condition
  (exhaustive NZCV), sign-extension, **wire-input header/bounds validation**, `simulation_input_load_fd`
  over a real `pipe()`, struct/ABI layout, fixup roundtrip, and **Group 42** (the new
  `cond_branch_is_taken` regression — direction must come from the condition, not `target==pc+4`).
- `test_ce_integration.c` — black-box: `fork+exec` the real `./contract_executor`, drive it over a pipe.
  ~55 tests: LDR/STR/LDP/STP addressing, NZCV, ALWAYS_MISPREDICT (cond clause), **BPAS store-bypass**,
  cond/BPAS composition + nesting caps, BPU input-selected-predictor / traps-without-predictor,
  unsupported-clauses rejection. (PAC integration tests are **intentionally disabled** — a faulting
  `AUT*` at EL1 is fatal on FEAT_FPAC HW; see the comment block. Consider re-enabling carefully.)

### 2c. Needs the **built CE** (+ kernel module on HW; self-skip if `/dev/executor` absent)
`unit_ce_mem_model.py` (LDP/STP both-elements; dest==base aliasing determinism — apply_fixups),
`unit_sandbox_ce.py` (execution-based sandbox bound; needs cross-`as`/`objcopy`),
`unit_pac_generator.py` (PAC stage1→exec→stage2 pipeline + fixed-key injection).

### 2d. Needs the **kernel module on real ARM HW** (`insmod` + `chmod 777 /dev/executor`)
- **`unit_kernel_module.py`** — the headline HW suite: sysfs store-handler validation, ioctl automata,
  **short-input write → -EINVAL**, **MTE_TAG_REGION bounds (accept full span / reject OOB)**, full
  load→trace→measure lifecycle.
- **`unit_pacga.py`** — PACGA ioctl correctness/determinism/key-isolation.
- **`unit_mte_random.py`** — MTE instrumentation promises over 100+ random test cases.
- `unit_code_base.py` — `print_code_base` JIT-offset correctness.

---

## 3. Functional probes (fixes with no automated test)

Run these by hand (or script them); each lists the expected result. They cover the audited fixes that
unit tests don't reach. **Build with `CONFIG_DEBUG_ATOMIC_SLEEP=y` and `CONFIG_DEBUG_PREEMPT=y`** for
the scheduling probes.

### MTE
- **SCTLR_EL1 / sync tag checks** (`executor/mte.c`): on a measurement CPU confirm `TCF=01` (sync),
  `ATA=1`, `WXN` unchanged; STG-tag a granule then a **mismatched-tag** access → **synchronous EL1
  Data Abort**; matching tag → no fault; the box still runs code (WXN not set).
- **Tagged sandbox + load-time self-check** (`executor/{mte,pagetable}.c`): on an MTE-active kernel the
  module **loads**; if MTE is inactive on an MTE build, the module must **REFUSE to load (`-EINVAL`)**,
  loudly; non-MTE build → stub no-op, still loads. Sandbox base PTE must be **Normal-Tagged**.
- **TCMA1 per-CPU** (`executor/sandbox.c`,`mte.c`): run a measurement on a CPU **other than** the one
  that loaded the module → must **not** oops with a tag fault (TCMA1 must be set on every CPU, not just
  init). Confirm TCMA1 is restored after unload.

### PAC
- **PACGA encoding** (`contract_executor/pac_sign_plugin.c`): a real `pacga x0,x1,x2` is recognized and
  signs with EL1 keys; the base opcode is **`0x9AC03000`** (2-source), **not** `0xDAC03000` (which
  aliases `AUTIZA`). Re-enable + run the disabled `AUT*` round-trip in `test_ce_integration.c` and
  confirm `AUTIZA` is no longer misclassified. **FEAT_FPAC fatality:** the old wrong constant could
  authenticate a corrupted pointer → reset the box; verify on the real FPAC core.

### Measurement / scheduling  ⚠ prime crash suspect
- **No scheduler/affinity in atomic context** (`executor/measurement.c`, `run_experiments`): with
  `CONFIG_DEBUG_ATOMIC_SLEEP=y`, run measurements under load **and** with an explicit remote
  `pinned_cpu_id` (the IPI path) → **no "scheduling while atomic" / BUG / oops**. Confirm no
  `set_cpus_allowed_ptr` remains in the atomic region. *(This is the ledger's prime suspect for the
  PAC-related box resets.)*
- **No-inputs error** (`measurement.c`/`chardevice.c`): MEASUREMENT with zero inputs → fails (`-EINVAL`),
  not "success".

### JIT
- **Overflow → `-ENOSPC`** (`executor/jit.c`): drive a maximal (~1024-instr) test case so the harness
  exceeds `MAX_MEASUREMENT_CODE_SIZE` → build returns **`-ENOSPC`** and refuses to run, never executes a
  **truncated** harness. Unknown measurement template → loud failure, not a zero-length harness.

### Chardevice / ioctl (`executor/chardevice.c`)
- CHECKOUT_INPUT (#4) / FREE_INPUT (#6) with a **bad user pointer** → **`-EFAULT`**, not `0`.
- `read()` with a bad user pointer → **`-EFAULT`**, does not advance `f_pos`, never an over-long count.
- Re-write TC/input in TRACED_STATE → stale measurements **invalidated**, not returned.
- Short input write (`< USER_CONTROLLED_INPUT_LENGTH`) → **`-EINVAL`** (covered by `unit_kernel_module`).

### Module load / lifecycle
- **`load_globals`** (`executor/main.c`): on kallsyms-lookup failure it must short-circuit (no call
  through a NULL fn-ptr → no panic).
- **Branch-training stub safety (C3)** (`executor/bpu.c`): a stub training **TAKEN** toward a
  backward/out-of-range target must **not** hang/oops (a RET is planted or it falls back to +4 / skips).
  ⚠ **status ambiguous in the audit notes — confirm in the current `bpu.c` whether this fix landed.**
- **Clean unload** (`executor/main.c` `__exit`): induce a `tracing_error`, then `rmmod` → no leak of the
  RWX vmap / device nodes (check `/proc/vmallocinfo` across load/unload cycles).
- **sysfs hardening** (`executor/sysfs.c`): `/sys/executor/*` no longer world-writable; `_show`
  handlers don't overflow PAGE_SIZE.

### Contract executor (userland)
- **Wire-input bounds** (`contract_executor/main.c`, `simulation_input.c`): a request with `HAS_REGS`
  clear (or `regs_size < 64`) that reads registers → **rejected**, no OOB/crash; sizes chosen to wrap
  the 64-bit length sum → **rejected**; a frame with `0 < length < 120` (header size) → rejected before
  the header copy. (Run the CE under a userspace ASan build if you can.)
- **TAGE error propagation** (`tage_py.c`, `neoverse_n3_bpu.c`): a predictor error must trap/propagate,
  never silently become "taken". Confirm the embedded CPython loads `bootstrap_director` →
  `Aarch64NeoverseN3BPU` from the CE's own directory.

### asm_to_bytes
- Confirm it invokes **`aarch64-linux-gnu-as`** (not host `as`); an input with no allocatable `.text` →
  **nonzero exit + stderr**, not empty output. Cross-check its bytes against the Python on-disk
  assembly path for an identical test case — they must match.

---

## 4. Per-subsystem verification checklist (server-verify commits)

> Goal: confirm each commit **builds** and **behaves**. Hashes are from `git log main..review/engine-fixes`.

### 4.1 Kernel module (`src/aarch64/executor/`) — ~20 commits
- MTE: `7443fe7` (SCTLR_EL1 sync TCF/ATA, was WXN bug), `d2cc4ed` (tagged sandbox allocator,
  `.`→`->`), `4236332` (tag overflow regions), `bc200ac` (load-time Normal-Tagged self-check),
  `e3024ca` (TCMA1 per-CPU), `d1d7107`/`972e8a3` (per-CPU enable + short-write + clean unload).
- PAC linkage: `ac8e91c` (kprobe return, full config defaults, PAC C99-inline link).
- chardevice/ioctl: `f337840` (CHECKOUT/FREE_INPUT errno), `0cd1ffd` (read `-EFAULT` sentinel),
  `f44fc73` (invalidate measurements on re-write), `cc62f22` (surface trace/alloc failures),
  `cce1bbf` (doc).
- measurement/sched: `0c6a9f9` (drop illegal sched/affinity from atomic), `b2bbd46` (loud
  `load_globals` + preempt-safe `execute_on_pinned_cpu`).
- jit/sysfs/templates: `761214b` (OOB guards + AND N-bit), `5beb3a5` (overflow → refuse truncated),
  `7b0c3e9` (sysfs perms 0644 + bounded buffers), `697a935` (loud bad-template), `d0ce907` (C3 BPU
  training-stub target).

### 4.2 Contract executor C + BPU (`src/aarch64/contract_executor/`) — ~14 commits
- **BPU speculative-PHR (headline):** `f7fb52f` (activate driver + bridges + bimodal base +
  `cond_branch_is_taken` fix + Group-42 regression), `bf4a79b` (explicit fold width), `53559f0`
  (shared-file UB + comment), `3cacb7b` (train on architectural path only).
  → **verify via `make test`** (test_ce Group 42 + test_ce_integration BPU cases) and the CE-load probe.
- TAGE robustness: `4a92e26` (loud init/reset), `b1bd383` (abort on update() exception).
- output/decoding: `544b349` (reg31=XZR for transfer/index), `539d579` (reg31=SP for SP-capable),
  `3ad1b5f` (don't swallow predict/alloc/transmit failures), `9efeace` (STGP post-index + SUBPS NZCV),
  `77766ad` (**PACGA_BASE bit30** → 0x9AC03000).
- speculation engine: `091a2e4` (bound nesting + trap on checkpoint OOM).
- wire protocol: `e383ae9` (input bounds C1/C2 + leak), `4678787` (framing).

### 4.3 asm_to_bytes — `df8acba`
Cross-prefix toolchain (`aarch64-linux-gnu-as`/`objcopy`) + reject empty output. Verify cross-binutils
present and a sample assembles; empty `.text` → nonzero exit.

### 4.4 HW-gated test additions (verify they pass on HW)
`3e8a06e` (short-input-write rejection) and `f64c0f8` (MTE_TAG_REGION bounds), both in
`tests/aarch64_tests/unit_kernel_module.py`.

*(The remaining ~37 commits are pure-Python with passing dev-box unit tests — re-run §2a to confirm,
but they are not the focus of HW verification.)*

---

## 5. Known-FLAGGED issues — NOT fixed (be aware; do not expect to pass)

These were deliberately left for a future design pass; the server should **know about them**, not try
to verify a fix:
- **PAC live-key swap (RISKY)** — `executor/pac.c` `pac_load_keys` swaps live `APIAKey*_EL1` (same key
  the kernel uses for pac-ret). Only safe if the compiler **inlines** the swap into `run_experiments`;
  otherwise prologue/epilogue PAC mismatch → fault. Test empirically with a kernel built
  `CONFIG_ARM64_PTR_AUTH_KERNEL`. Fix direction: `__attribute__((target("branch-protection=none")))`.
- **pagetable.c BBM (RISKY)** — `disable_mte_for_region` writes a **live** kernel PTE with no
  break-before-make → CONSTRAINED UNPREDICTABLE; can silently corrupt Prime+Probe results (not a loud
  failure). Validate MTE-eviction results cautiously.
- **`HardwareTracingError` model is dead** (`aarch64_executor.py`) — the local path raises `OSError`,
  so the fuzzer's `except HardwareTracingError` is dead → a transient ioctl error **aborts the whole
  campaign** instead of skipping one test case.
- **MTE stage-2 NI rework** not implemented (design only, see memory `kb-mte-design`); the MTE
  non-interference fuzzer is **not yet a complete leak oracle** (fixed tag 6, LDG/LDGM return tag 0,
  SUBPS no NZCV, STGP post-index unrecognized).
- **CE/BPU model-fidelity decisions** (deliberately deferred, your call): the kernel `bpu.c`/`jit.c` PHR
  model (300-bit / shift-4 / 11-bit) is confirmed vs the RE, but the **TAGE fold + tag widths** in
  `saturating_bp.py` are **placeholders, NOT reverse-engineered** (need the shift/fold microbench), and
  **u-bit / altpred are not modelled**.

---

## 6. Crash suspects & priority order
1. **`measurement.c` scheduler-in-atomic (`0c6a9f9`)** — the prime suspect for the PAC-related box
   resets. Verify first with `CONFIG_DEBUG_ATOMIC_SLEEP=y` under load + remote-CPU (IPI) path.
2. **PACGA encoding (`77766ad`)** — the box-reset-on-FEAT-FPAC suspect (old constant aliased `AUTIZA`
   → authenticated a bad pointer). Verify the encoding + the re-enabled `AUT*` round-trip on the FPAC core.
3. **Branch-training stub safety / C3 (`d0ce907`, `executor/bpu.c`)** — status ambiguous in the notes;
   confirm the fix is present, then exercise a backward/out-of-range taken-training stub (no hang/oops).
4. Then the MTE load/TCMA1 set (oops on non-init CPU), JIT overflow, chardevice errno paths.

---

## 7. Appendix — ioctl / wire reference (for building probes)
- **ioctls 1–13** (magic `'r'`): 1 CHECKOUT_TEST, 2 UNLOAD_TEST, 3 GET_NUMBER_OF_INPUTS,
  4 CHECKOUT_INPUT, 5 ALLOCATE_INPUT, 6 FREE_INPUT, 7 MEASUREMENT, 8 TRACE, 9 CLEAR_ALL_INPUTS,
  10 GET_TEST_LENGTH, 11 PAC_SIGN, 12 PAC_AUTH, 13 PAC_XPAC. PAC keys travel per-request in
  `pac_sign_req`; MTE tags travel in the input's `MTE_TAGS` section — neither has a keys/tag ioctl.
- `UserMeasurement = htrace:u64 + pfc:u64×3`.
- CE wire: inner header 120 bytes; outer frame `<II>` = (length:u32, type:u32), payload magic `RVZR`.
  Section flags `HAS_CODE=1 / HAS_REGS=2 / HAS_MEMORY=4`. Execution clauses `SEQ=0 / COND=1 / BPAS=2 /
  BPU=4`; `BranchPredictor NONE=0 / NEOVERSE_N3=1`; clause string `bpu_neoverse_n3`.
- Device: `/dev/executor` (ioctl + bulk read/write after a checkout); sysfs `/sys/executor/`
  (`measurement_mode`, `warmups`, `enable_pre_run_flush`, `pin_to_core`, `enable_branch_training`,
  `branch_training_config`, `print_sandbox_base`, `print_code_base`).
