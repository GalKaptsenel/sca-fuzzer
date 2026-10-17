# Independent verification / audit log (integrity record)

## v2 (Spectre-v2 / branch-target) genuineness — CE-only, independent agent — 2026-10-16
**VERDICT: GENUINE branch-target (v2-class) leaks.** Device not touched (campaign held it); CE + static only.

- Config eval_configs/v2_l1d.yml: multi-target indirect dispatch (entry + 2 callee targets), NO BASE-BRANCH,
  contract [cond,bpas], obs l1d, nesting 1..30. v1(cond)/v4(bpas) are in-contract by construction.
- Gadget (static, both runs): `ADR table; LDRSW; ADD; BLR Xd` -> function_1 (has input-dependent load) OR
  function_2 (leaf, no mem). No conditional branch carries the leak. Indirect-BLR target gadget.
- CE contract ladder (base validated against the saved report hash): seq/cond/bpas/cond-bpas ALL AGREE
  (divergent-sets=[]). So the two counterexample inputs are genuinely contract-equal through [cond,bpas];
  the HW difference is beyond-contract (true positive, not a modeling artifact).
- Decisive match:
  - run1 (#22 vs #72): only initial x1 differs; on the MISPREDICTED target function_1, `LDR x1,[x1]`
    maps x1=0x9247->L1D set 9, x1=0xf0c9->set 3 == HW diff {3,9}. x1 clobbered only AFTER the BLR,
    so input-dependent x1 reaches a load ONLY on a mispredicted dispatch. ~65% HW freq = real intermittent
    BTB target mispredict. UNAMBIGUOUS Spectre-v2.
  - run2 (#9 vs #59): initial x2 differs; computing the target load with initial x2 gives set 12 / set 40
    == HW diff {12,40}; reachable only by a target mispredict skipping the prologue clobber. v2-class.
- Rules out: v1 (cond AGREE, no cond branch), v4 (bpas AGREE, pointer-chased load not bypassed store),
  noise (exact set match in two independent test cases), modeling-artifact (inputs truly contract-equal).

### Caveats / follow-ups (need HW, do after campaign frees the device)
1. run2: CE cannot separate pure BTB indirect-target mispredict from inner RET/RSB (ret2spec) — both
   v2/branch-target class. Pin via HW controlled-swap (swap only the leaking slot; confirm divergence persists).
2. Absolute set numbers: CE common-mode set differs from HW always-on sets (htrace->set offset / P+P
   eviction geometry / prefetcher). The DIFFERENCE sets match exactly (what matters); reconcile absolute
   offset on HW before quoting absolute set indices. [[project_hw_ce_l1d_alignment]]

### TOOLING BUG FOUND (integrity)
- `tools/triage.py` SIGBUSes on dispatch/multi-target test cases: it omits `data_size` so the 8-byte
  dispatch table is parsed as code and faults at the BLR. Workaround added: `tools/v2_ce_ladder.py`
  (passes data_size=8; base reproduces the saved contract hash exactly). Fix triage.py to honor data_size.

## REMOTE BLOCKER (Pixel module ABI) — 2026-10-16
- Pixel /data/local/tmp/revizor-executor.ko (Sep build) reports abi_version=1; Oct codebase expects
  EXECUTOR_ABI_VERSION=2 (src/aarch64/executor/userapi/executor_user_api.h). Remote fuzzer _check_abi_version
  would raise "device executor ABI 1 != expected 2; reload the module". So remote A4/A5 is BLOCKED until an
  ABI-2 .ko is built for the Pixel kernel (vermagic 5.15.137-android14-11-gbc062a78e195).
- Repo _LOCAL_MODULE is N3-only (vermagic 6.12.111). build_pixel8.sh needs ~/pixel-kernel/out +
  ~/toolchains/llvm-bin (on laptop box B), absent on N3.
- Userland OK: _LOCAL_USERLAND is static aarch64 (runs on Android), ABI-2, fuzzer pushes it fresh.
- ACTION: attempt to build an ABI-2 Pixel .ko on N3 (download matching GKI android14-5.15.137 + LLVM);
  if infeasible (exact Module.symvers/vermagic match needs the Pixel kernel tree), remote stays prepped
  and documented, to run once an ABI-2 .ko exists. DO NOT fake remote with mismatched ABI (would be garbage).

## A6 localization — integrity note (2026-10-16)
- exp_search (cross_input.py) docstring claims "Cost O(log d)... never worse than a plain bisection". MEASURED:
  for a distant boundary (d~k) it does ~2*log2(k) probes (k=12: up to 9 vs pure-bisection 4), i.e. WORSE than a
  plain bisection over the full range. It IS O(log d) and beats bisection for RECENT (near-hi) boundaries (~4,
  const in k). The "never worse than plain bisection" claim holds only for recent/close boundaries; flag for doc fix.
- Measured via eval_scripts/a6_localization_cost.py importing the REAL localizers (no reimplementation),
  synthetic single-boundary oracle, counting distinct memoized probes. Device-independent (localizer cost is
  determined by k and boundary position, not the HW).

## CROSS-INPUT BTB REGRESSION — ROOT CAUSE (2026-10-16)
Symptom: cross-input BTB leftover gives 0 divergence / 0 violations (template canon_oracle_btb.py and
random canon_btb) even with ssbs=1, pre_run_flush=0. Seal/boost/priming plumbing verified correct.
ROOT CAUSE: the leftover regime (view_rotation=0, phr_flush=0) is applied ONLY inside _priming
(aarch64_fuzzer.py:128, the LOCALIZATION step via executor_regime(_CROSS_INPUT_PRIMING_REGIME)), NOT
around the INITIAL candidate-detection measurement (aarch64_fuzzer.py:275 main trace loop). view_rotation
& phr_flush default to PRE_RUN_FLUSH_DEFAULT=1 (executor.c:94-96) and are NOT written from CONF at init
(aarch64_kernel.py init writes only pin, pre_run_flush, ssbs). So the detection measurement runs with
view_rotation=1 -> BTB invalidated per input -> cross-input leftover WIPED before a candidate is flagged
-> _priming never runs -> 0 violations. (Matches memory project_btb_leftover_template_automatic killer bug:
old _maybe_scan_leftovers wrapped the whole scan in the regime; new cross_input_priming wraps only
localization.) Independent of v1/v4 window and of pinning.
FIX: apply _CROSS_INPUT_PRIMING_REGIME around the detection measurement loop when enable_cross_input_priming
is set (or add CONF enable_view_rotation/enable_phr_flush driven at init). Then template should surface the
leftover; then fix the generator kaddr2uaddr crash for real random fuzzing.

## CROSS-INPUT BTB — RESOLVED (2026-10-16 19:39)
CONFIRMED cross-input leak via templates/canon_oracle_ssb.py + eval_out/btb/tmpl_ssbs.yml + -i 8:
  "cross-input priming: CONFIRMED cross-input leak, leaking pair 0 -> detecting pair 2" (Violations:1).
  Genuine cross-input: leaker(0) != detector(2); cross_input_leaks_only=true (strictly non-self).
  Violation: eval_out/btb/ssb_i8/violation-261016-193925.
TWO compounding regressions:
 (1) detection ran under flush regime: _CROSS_INPUT_PRIMING_REGIME (view_rotation=0/phr_flush=0) was
     applied only around localization (_priming), not the initial detection measurement; view_rotation
     defaults to 1 and is not CONF-driven at init -> BTB flushed per input -> leftover wiped. FIX:
     aarch64_fuzzer.py:117 wraps the detection path in the regime too (subagent).
 (2) -i too large (100/400) diluted the leaker->prober class structure; original used -i 8 (small).
 (+) generator crash fix (sealer.py): re-clamp base registers after a BL/BLR (callee clobbers them).
NEXT: real random fuzzing (no template) with small -i to auto-detect it (user's end goal).

## CROSS-INPUT BTB — REPRODUCIBILITY CONFIRMED (2026-10-16 20:08)
Replayed randC's exact seed (program_generator_seed=799067, input_gen_seed=79352157) via
eval_out/btb/randC/rerun.yaml -> IDENTICAL: "CONFIRMED cross-input leak, leaking pair 0 -> detecting
pair 3", tc 5, 9s (eval_out/btb/reseed/violation-261016-200835). DETERMINISTIC + HW-reproducible.
The earlier "0 in ~250 tc" fully-random runs were genuine SEED VARIANCE (randC's seed forms a
cross-input gadget at tc5; most seeds don't for hundreds of tc). Regime verified live-correct
(view_rotation=0/phr_flush=0/ssbs=1/unpinned during detection) -> NOT a flush regression. Honest:
the leak is real + reproducible; its per-random-seed occurrence is low (gadget rarity).

## CROSS-INPUT BTB — v2 + no-SSB (v1) fully-random (2026-10-16 20:10)
- v2_btb.yml (Spectre-v2, no canonicality sealing, SSB on, cross_input_leaks_only) fully-random: 0 cross-input
  leaks in 44 tc/98s (candidates localized, all non-cross-input -> false positive). Rare, like canon random.
- canon_btb_nossb.yml (v1 window, SSB mitigation present) fully-random: 0 confirmed in ~600s. Rarer (v1 needs
  a mispredicting branch AND a cross-input gadget; fixed-PC doesn't mispredict on N3, random diverse PCs do but
  the combination is uncommon).
CONCLUSION: cross-input canonicality-over-BTB leak is REAL + REPRODUCIBLE (template ssb_i8 + gadget seed
799067 deterministic 9s). Fully-random auto-detection across arbitrary seeds is rare (gadget rarity), v1 rarer.
v2/BTB uses the same carrier; a clean v2 demo needs a gadget seed/template (folded into the audit).

## CROSS-INPUT BTB — restrictiveness hypothesis TESTED & RULED OUT (2026-10-16 20:3x)
cr-seq (contract [seq], less restrictive) fully-random, -i 8:
  BTB-carrier+seq : rep1 131tc->0, rep2 118tc->0
  ordinary-canon+seq: rep1 1283tc->0, rep2 (see summary)->0
CONCLUSION: the narrow contract [cond,bpas] was NOT the cause — [seq] also yields 0 in random. Ordinary-load
canonicality has NO cross-input carrier (ordinary loads don't persist state across inputs) so 0 is expected
even at 1283 tc; the BTB (indirect sealed TARGET) is the only cross-input carrier, and that gadget is RARE in
random programs. Not a config/contract/regime bug (regime live-verified view_rotation=0/phr_flush=0; [seq]
ruled out restrictiveness). The leak is REAL + REPRODUCIBLE on a gadget seed (799067, 9s) and RELIABLE via the
template (canon_oracle_ssb, 6s x2 CONFIRMED leaker 2->detector 3); fully-random auto-detection across arbitrary
seeds is seed-dependent/rare because the specific speculative-sealed-indirect-target gadget forms uncommonly.

## v2/BTB vs Canonicality/BTB — honest status (2026-10-16 22:28)
- Canonicality/BTB: SOLID. Working template canon_oracle_ssb.py (CONFIRMED cross-input every run: pairs
  0->2, 2->3, 4->6) + fully-random 12/20 (60%) + 2 triage-GENUINE. Leaked axis = target canonicality.
- Pure Spectre-v2/BTB (leaked bit = which target, NO canonicality sealing): NOT cleanly demonstrated.
  Fully-random v2_btb_3t = 0/3. Template canon_oracle_btb.py does NOT detect (0 violations at -i8 AND -i50,
  49 classes) -> that template is non-working; canon_oracle_ssb.py is the working BTB template but its axis
  is canonicality. The BTB which-target CARRIER is demonstrated (shared by both rows); the pure-v2 CONTENT
  variant was not independently surfaced on N3. (Fold a dedicated pure-v2 template into the audit if needed.)

## SUPERSEDED (2026-10-17) — pure Spectre-v2/BTB now CONFIRMED via CT-SEQ
The "NOT cleanly demonstrated" verdict above predates the CT-SEQ breakthrough and is SUPERSEDED. Pure
Spectre-v2/BTB (which-target, NO sealing, NO PAC) IS confirmed on the template templates/spectre_v2_btb.py
under CT-SEQ: forcing contract [seq] makes the speculative dispatch/target beyond-contract so the
target-selecting bits are untainted and boosting creates the within-class which-target toggle. tfuzz -i100
= 1 viol/57s, cross-input leaking 0->detecting 11, deterministic reproduce 3/3. See eval_out/v2btb/FINDINGS.md.
Fully-random (no-template) detection remains structurally rare (plain-fuzzer ct-equal boosting cannot toggle
an ARCHITECTURAL dispatch) and was NOT achieved; per user decision we add no seal. canon_oracle_btb.py
non-working note above still stands.

## REMOTE BLOCKER RESOLVED (2026-10-17) — supersedes the 2026-10-16 "BLOCKED" note above
The ABI-2 Pixel module WAS built and installed (from the GKI tree, cross-compiled; artifact
~/pixel-mod-build/revizor-executor-abi2.ko, pushed to /data/local/tmp/revizor-executor.ko — same path, ABI-2
content replacing the Sep ABI-1 build). VERIFIED on device 2026-10-17: `adb shell cat
/sys/executor/system/abi_version` = 2, matching the repo's EXECUTOR_ABI_VERSION=2, so _check_abi_version does
NOT raise and the remote runs executed against a matching ABI (NOT the "garbage" mismatch the note warned
against). All remote rows (eval_out/remote/*, eval_out/remote_a510_x3/*) are therefore valid. The "BLOCKED"
section above is stale and superseded.

## REMOTE A715 5/5 — reconciliation (2026-10-17)
eval_out/remote/summary.csv shows v1_l1d/a715 det=0 (missed at 107 tc / 311 s, the 300 s cap). The "A715 5/5"
headline rests on a RECHECK with a larger budget: v1/a715 detected at 137 tc / 374 s (campaign.log
"v1/A715 recheck: det viol=1 tc=137"). Honest statement: A715 = 4/5 within the first 300 s cap; the 5th (v1)
reproduces on recheck at 137 tc — v1 on the A715 is marginal (core-dependent BPU; A715 needs more test cases
than X3), consistent with [[project_n3_fixed_branch_no_mispredict]]-style per-core behavior. summary.csv was
not re-folded; treat A715 v1 as "reproduced on recheck", not first-cap.
