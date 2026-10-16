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
