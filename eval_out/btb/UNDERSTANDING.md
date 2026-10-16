# Cross-input canonicality-over-BTB leak — full understanding (2026-10-16)

## What it is
A leaker input speculatively trains a BTB entry for an indirect call to a (canonicality-sealed) target;
a LATER, contract-equal prober input then speculatively jumps to that mistrained target -> the prober's
L1D trace reveals the leaker's sealed-target canonicality. Caught by GENERALIZED (cross-input) priming,
missed by standard priming. Detector: enable_cross_input_priming + cross_input_leaks_only (strictly non-self).

## What had broken it — TWO regressions (both fixed + committed 2c3b18e3)
1. DETECTION-REGIME (aarch64_fuzzer.py). The no-flush BTB leftover regime (view_rotation=0, phr_flush=0,
   pre_run_flush=0, ssbs=1, unpinned) was applied ONLY around localization (_priming), NOT the initial
   candidate-DETECTION measurement. view_rotation/phr_flush default to PRE_RUN_FLUSH_DEFAULT=1 (executor.c)
   and are NOT written from CONF at init -> detection flushed the BTB per input -> leftover wiped BEFORE any
   candidate -> _priming never ran -> 0 violations. FIX: `fuzzing_round` wraps the WHOLE round (detection +
   localization) in executor_regime(_CROSS_INPUT_PRIMING_REGIME). (Matches the historical killer bug in
   memory project_btb_leftover_template_automatic: old _maybe_scan_leftovers wrapped the whole scan.)
2. GENERATOR CRASH (sealer.py SandboxWalk). The sandbox-clamp dataflow elided re-clamping a reused base
   register, but did NOT invalidate that across a BL/BLR: a callee clobbers caller regs, so a base clamped
   before the call and clobbered by the callee was reused UNCLAMPED -> architectural out-of-sandbox access
   -> CE fail-loud "[CE FATAL] kaddr2uaddr ... spec_nesting=0". FIX: after a call, drop in-region facts
   (`if inst.is_call: curr = frozenset()`). Test: tests/aarch64_tests/unit_sandbox_call_reclamp.py.

## The config insight — fully-random detection rate
The cross-input gadget (speculative SEALED INDIRECT-CALL TARGET trained by an earlier input) is the ONLY
cross-input carrier (ordinary-load canonicality has NO cross-input carrier: 0 leaks in 1300 tc). In random
programs this gadget is RARE with few targets / few inputs:
  - 2 dispatch targets + -i 8  -> seed-dependent, often 0 in 100+ tc (randC seed 799067 was a lucky tc-5 hit).
  - >=3 dispatch targets + -i 50 -> RELIABLE: 2/2 runs detected distinct random gadgets (pair 24->27 in 8tc/126s;
    pair 23->24 in 2tc/90s). More BTB candidates + more boosted inputs raise the per-test-case detection chance.
RECIPE for fully-random (no template) auto-detection: min_functions_per_test_case>=4 (entry + >=3 targets),
indirect_call_probability=1, dispatch_call_probability=1, enable_branch_target_sealing=1, SSB window on,
enable_cross_input_priming + cross_input_leaks_only, enable_pre_run_flush=0, -i ~50.  (eval_configs/canon_btb_3t.yml)

## Ruled out (NOT the cause)
- Contract restrictiveness: [seq] no better than [cond,bpas] in random (both ~0); not a contract issue.
- Regime re-flushing: LIVE sysfs dump during fuzzing = view_rotation=0/phr_flush=0/ssbs=1/unpinned (correct).
- SSB-only / v1-vs-v4: not window-specific. "Fixed-PC no-mispredict" applies only to the TEMPLATE (fixed PCs);
  random has diverse PCs. Pinning: unpinned (pinning raises the non-canonical residual).

## Reliable demonstrators (reproducible)
- Template templates/canon_oracle_ssb.py (+ eval_out/btb/tmpl_ssbs.yml, -i 8): CONFIRMED ~6s, reproducible (x2 leaker2->detector3).
- Gadget seed program_generator_seed=799067 input_gen_seed=79352157 (canon_btb.yml, -i 8): deterministic 9s, leaker0->detector3.
- Fully-random eval_configs/canon_btb_3t.yml (-i 50): 2/2 distinct random gadgets.

## Required executor regime for the cross-input BTB leftover (must hold during detection AND localization)
enable_ssbs=1, enable_pre_run_flush=0, enable_phr_flush=0, enable_view_rotation=0, pin_to_core=-1 (unpinned).
