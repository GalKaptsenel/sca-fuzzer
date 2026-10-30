# TikTag-v1 detection on the Pixel 8 (Cortex-X3) — results

Status snapshot 2026-10-30. This records the TikTag-v1 find, its reproducibility, the genuineness
triage, and the two transport/environment fixes that unblocked it. It is a results log, not the
paper mechanism writeup.

## Recipe (the one that detects)

- **Device:** Pixel 8, Tensor G3, pinned **Cortex-X3 = core 8** (the TikTag-v1 / Gadget-1
  speculation-shrinkage core; A715 cores 4–7 carry Gadget-2 / v2 store-to-load-forwarding instead).
- **Template:** `docs/tiktag_mte_ni/general_detection/user_mte_skeleton.asm` — a planted Spectre-v1
  window (SDIV dependency chain → `ORR` → `CBNZ`) with one sealed tagged `LDR [x5]` in the
  not-architecturally-taken shadow block, random instructions before/after.
- **Config:** `docs/tiktag_mte_ni/general_detection/s1_mte_hw_hard_noMT.yml` — `fuzzer:
  speculative-exception`, `speculative_exception_mode: strict`, `contract_execution_clause: [cond]`
  (conditional-branch window, **not** bpas), `contract_observation_clause: l1d`, `executor_mode:
  F+R`, `enable_pre_run_flush: true`, `executor_sample_sizes: [2000]`, `enable_branch_mistraining:
  false`, remote adb to the Pixel.
- **Command:** `revizor.py tfuzz -s base.json -c <that config> -t <that template> -n 150 -i 10`.

## The find

`violation-261030-140431` (test case #9), **time to detection ≈ 171 s**, 40-input equivalence class,
0 hardware tracing errors.

## Reproducibility — CONFIRMED

`revizor.py reproduce -s base.json -c <violation>/reproduce.yaml -t <violation>/generated.asm -i
<all 40 input_*.reif>` re-ran the saved test case and inputs through CE + hardware and reported
`Violations: 1`, `Effective Cls: 40.0`, `Hardware Tracing Errors: 0`. The reproduce path is
deterministic (the sealer salt is derived from the test-case bytes, not the global RNG stream), so
the reconstructed variants are byte-identical to the fuzzing run.

## dmesg — clean (no MTE exception written)

Throughout the run and the reproduce, the Pixel kernel log shows **0** `TFSR tag fault` lines (the
executor's async-TCF report, `src/aarch64/executor/measurement.c:404`) and 0 other
MTE/ESR/BUG/panic/KASAN lines. The genuine/architectural path raises no MTE exception; the decoy
mismatch is confined to the speculative path.

## Genuineness triage

The triage tools were migrated to the GlobalValueTable model (see the commit): a variant is now its
GVT image, not a spliced relocation plan, so "tag-only" is a GVT-image diff and a tag sweep rewrites
the applied-tag lane while the re-converge lane stays genuine.

- **Gate 1 — tag-only: PASS.** Genuine vs decoy GVT images differ only in MteSealing applied-tag
  lanes (`accounting.py`).
- **Gate 2 — speculative-only leaking set: PASS.** Of the hardware-diverging sets, the leak-relevant
  ones are CE speculative accesses, not architectural demands: set 49 *is* a speculative demand
  (base x3), set 6 is a prefetch stream of a speculative demand (base x4). Architectural-demand
  divergences (sets 25, 32) are prefetcher/batch contamination and are excluded.
- **Gate 3 — 16-tag pointer sweep:** running (`triage_sweep.py`, 2000 reps).
- **Gate 4 — memory-tag tracking control:** pending (`tag_track.py`).

A GENUINE verdict requires all four gates AND the memory-tag tracking moving the detected match tag;
gates 3–4 are not yet complete, so no genuine claim is made here yet.

## Environment/transport fixes that unblocked this

1. **Device-exclusivity race (root cause of the "malformed input" / Hardware Tracing Errors).** A
   leftover autonomous campaign (`collect_daemon.sh` → `revizor.py fuzz s24_mte_random.yml
   --nonstop`, resurrected every 5 min by a `campaign_boot.sh` cron) was driving the same
   `/dev/executor`. Two processes corrupt the module-global `checkout_region`, so a test-case write
   was validated on the input path → "Malformed input initialization". Stopped the daemon + fuzz and
   disabled the cron. After that, 90 TikTag test cases ran with 0 tracing errors.
2. **stderr isolation (commit `10df5412`).** `ADBConnection.run` drove the batch over adb's `exec:`
   service, which merges the remote command's stderr into stdout, so a device/unit error polluted the
   binary response and surfaced as an opaque "bad magic". It now uses adb shell-protocol v2 (raw, no
   PTY): stdout, stderr and the exit code are framed separately; a nonzero exit raises `IOError`
   carrying stderr, mapped to `HardwareTracingError` (skip the test case). Test:
   `tests/aarch64_tests/unit_adb_shell_v2.py`.
