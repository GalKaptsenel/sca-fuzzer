# SESSION HANDOFF — AArch64 PAC/MTE non-interference sealing (branch review/engine-fixes)

Standalone catch-up for a fresh session. Read this + the memory files in
`~/.claude/projects/-home-gal-k-1-1998-sca-fuzzer/memory/` (especially `[[pac-mte-context-tag-mismatch]]`,
`[[kernel-pac-auth-fpac-resets-vm]]`, `[[fix-at-source-not-downstream]]`, `[[code-style-no-default-args]]`).

## Environment / safety (READ FIRST)
- Run Python with `/home/gal_k_1_1998/revizor/revizor-venv/bin/python` (system python lacks deps).
- After the module is (re)loaded: `sudo chmod -R a+rwX /sys/executor; sudo chmod a+rw /dev/executor`.
- The VM **reboots spontaneously** (infra) — `/tmp` is wiped on reboot, `/home` survives. Put scratch in
  `/home`. After a reboot: `cd src/aarch64/executor && sudo insmod revizor-executor.ko` + the chmod.
- A wire-format / kernel change must REBUILD+RELOAD the kernel module (`make` in `src/aarch64/executor`,
  then rmmod/insmod) AND rebuild the CE (`make contract_executor` in `src/aarch64/contract_executor`).
- **Box-safety is now handled in the kernel:** `handle_pac_auth` -> `pac_auth_with_keys` (pac.c) verifies
  a signature with XPAC+re-sign and only runs the real AUT* if it would succeed, else `pr_warn(
  "revizor: PAC AUTH would FPAC ...")` and returns the canonical pointer. So a wrong sig NEVER resets the
  box from any caller. `CE_DEBUG_PAC_AUTH_NO_FAULT` in `pac_sign_plugin.c` is `0` (real AUT*, kernel-safe).
  Debug a sealing FPAC by running the suite/harness and `sudo dmesg | grep "would FPAC"`.

## What is DONE this session
- Kernel-side AUTH safety net (`pac_auth_with_keys`, pac.c + chardevice.c) — built, loaded, verified.
- Uniform MTE initial tag: `MTE_INITIAL_TAG=6` (aarch64_mte.py); model seeds `MteTagState(6)`; executor
  feeds `[6]*MTE_TAG_COUNT` via `_mte_tags_for` when "mte" in `_primitives` (tasks #26/#29 done). Deleted
  dead `_resolve_mte_staged`.
- **Self-context `AUT* Xn,Xn` FIXED in the generator** (the proven primary regular-sealed would-FPAC):
  `aarch64_generator.py` `_AUTH_CTX` + `Aarch64PatchUndefinedLoadsStoresPass._patch_auth_context_collision`
  forces ctx!=ptr. Silent fallthroughs at aarch64_pac.py:308/442 now `for/else: raise`. Verified: harness
  trials 0-84 -> 0 would-FPAC (trial 0 used to fault on `AUTIA x3,x3`).

## IMMEDIATE NEXT TASK — zero-context PAC would-FPAC (the remaining bug)
- `AUT*ZA/ZB` (ctx=0) still would-FPAC (repro: `investigate_seal_fpac.py`, hits trial 85). DIFFERENT
  mechanism from self-context: the seal resolves the sig over the register's value in the PLACEHOLDER
  trace ([NOP,XPAC] PAC slots), but the register holds a different (wild, un-clamped) value at the GENUINE
  auth — e.g. signed_ptr=0xf700aaaaffd20f57 vs auth ptr=0xf71daaab00227397 (whole low address differs).
  General "resolve-over-placeholder != genuine register state" divergence.
- Likely fix: iterative PAC resolution (resolve sigs in program order, applying each genuine sig into the
  TC before resolving the next so each sig sees the genuine register state), OR a genuine-PAC re-trace
  pass. See `_resolve_pac` / `MtePacSealedTestCase.resolve` in aarch64_sealer.py.
- After fixing, re-run `investigate_seal_fpac.py` until many trials are clean, then run the full suite.
- REMOVE the debug `[SEAL-PAC]` print left in `aarch64_sealer.py:_resolve_pac`.

## Tooling
- Durable repro harness: `/home/gal_k_1_1998/sca-fuzzer/investigate_seal_fpac.py` (deterministic seeds;
  wraps `_seal_trace`, dumps the faulting TC + correlates seal sig <-> auth ptr). Last report:
  `/home/gal_k_1_1998/seal_fpac_report.txt`.
- Python suite: `revizor-venv/bin/python -m unittest discover -s tests/aarch64_tests -p 'unit_*.py'`
  (`unit_regular_sealed` is the PAC×MTE sealing test that triggers the would-FPACs).
- C tests: in `src/aarch64/contract_executor`: `make test_ce test_ce_integration test_mte_tagmem`;
  in `src/aarch64/executor`: `test_pac_auth_verify`.

## After the sealing is clean (broader e2e goals)
- Run the full suite periodically; watch `uptime` + `dmesg | grep "would FPAC"`.
- TME completeness: set kernel `GCR_EL1.Exclude=0` (guarded by MTE support; untestable on this no-MTE box).
- Full e2e of NI / regular-sealed PAC×MTE on the box once the suite is green.
- Pending tasks: #18 remote batching, #19 PAC alt_sig pool, #21 multi-agent review.
