# Autonomous execution plan (Machine A) — order, contention rules, commands, recovery

## Contention / timing rules (INTEGRITY)
- N3 TIMING-sensitive phases run with a QUIET host (no other fuzzing/building): TTD, throughput, exploits.
- Remote (Pixel) work runs in N3-IDLE windows (its host-side gen+CE would bias N3 timing).
- Pixel keepalive (adb ping /2.5min) runs throughout (negligible load).
- Before every launch: check device idle (ps for revizor/contract_executor) + correct PID; before every kill: use the exact PID (never pkill-by-name). Validate config + CE-model locally before committing a long run.

## ORDER
1. [running] TTD 5 rows (PID 18069). On DONE: compute per-row mean±sd wall, mean tc-to-viol, inputs-to-viol, detected/R -> Table 7.2 N3 TTD + N3 repro column.
2. v2 variants (device): probe v2_l1d_1bb.yml (single BB) and v2_btb.yml (cross-input, cross_input_leaks_only). Compare TTD to v2_l1d. Record.
3. Canon/BTB TTD: config_canon_cip_ni.yml (cross-input, unpinned). Vary input_gen_seed for R reps (edit per-run or pass). Record TTD + repro.
4. A1 throughput (3-way, --nonstop --timeout 3600, 2 reps each; run_throughput.sh):
   a. local:        config.yml-equiv pinned (contract seq) -> tc/h, exec/s, traces/s
   b. remote-loopback: same config + executor_remote over adb to a LOOPBACK N3 executor (isolates protocol/orchestration overhead). [verify how to target loopback-N3 over adb; may need a local adbd or an 'ssh/local' transport -> if not feasible, measure transport overhead via per-tc round-trip timing instead and SAY SO]
   c. remote-Pixel:  executor_remote pinned core 4 (A715) and core 8 (X3) -> device-limited rate
   OVERHEAD = local - loopback (protocol); device-speed = loopback - pixel. Also log per-tc adb round-trip time.
5. Remote A4 (A715 core4) + A5 (X3 core8) reproduction: 7 rows. L1D/PAC rows via make_remote_config.sh (standard priming). BTB rows: cross-input is LOCAL-ONLY -> remote BTB verdict needs a non-cross-input path or comes from the cross_boundary exploit; DECIDE at run time, report honestly. X3 Canon/BTB expected cross.
6. A6 localization: instrument Localizer (cross_input.py) measure() call count over boundaries for k=12/64/256; mean HW-exec = calls x reps x 2 dirs; compare analytic ceil(log2(k+1))=4/7/9.
7. A7 exploits (eval_out/exploit_plan.md): build common/raw_pmu.ko + each PoC; run on isolated core 90, quiet host. OpenSSL ~5.5h (LAST, long pole). Capture accuracy%, bit/s, bit counts, wall-clock.
8. Multi-model audit: Fable + others independently verify ALL violations/results (CE triage + HW controlled-swap where needed; run HW swaps only when device idle). Investigate >=1 of EACH type. Append to eval_out/VERIFICATION.md.
9. Fill EVALUATION_RESULTS.md §5 tables (mean±sd, R); git commit all.

## Recovery (if a device wedges)
- N3: (reboot only if truly wedged) else rmmod -f revizor_executor; insmod ...; chmod 666 /dev/executor; chmod -R a+rw /sys/executor.
- Pixel: adb -s 47061FDJH001K3 shell 'su -c "rmmod revizor_executor; insmod /data/local/tmp/revizor-executor.ko; chmod 666 /dev/executor"'. Re-verify dmesg.

## Open items to resolve at run time (think first, don't guess blind)
- Remote device-binary (Sep build) vs Oct fuzzer protocol: SMOKE -n1 first; rebuild/repush if mismatch.
- Remote BTB rows detection method (cross-input local-only).
- Loopback-N3-over-adb feasibility for overhead isolation.
- triage.py data_size bug (use tools/v2_ce_ladder.py pattern; fix triage.py in cleanup).

## CROSS-INPUT BTB rows (v2_btb, canon_btb) — REFINED (user directive)
WHAT WE SHOW: a genuine CROSS-INPUT BTB leak — earlier input trains a BTB target; a later
contract-equal input's speculation changes; caught by generalized (cross-input) priming, missed by
standard priming. Distinct from the same-input v2 (mispredicted target does a load -> L1D).
CONFIG for the cross-input BTB gadget:
  - >=2 (try 3) dispatch targets.
  - targets DO NOT access memory (so NO same-input v2 path; the only leak is cross-input BTB state).
    Mechanism to enforce: tiny targets (function_size_shrink small so targets have no room for a load)
    and/or structure so only the entry/prober holds the observable; VERIFY by disassembly that targets
    contain no LDR/STR.
  - enable_cross_input_priming: true ; cross_input_leaks_only: true (strictly non-self) ; unpinned.
  - full nesting.
VERIFY (auditor/integrity agent, CE + leak-flow, HW controlled-swap when device idle):
  - the leaking PAIR are DIFFERENT inputs (leaker != prober) -> truly cross-input.
  - the divergence tracks which BTB target was predicted (trained by the leaker), not a same-input load.
  - NOT the same-input v2, NOT noise. Investigate >=1 detected violation per BTB row by hand.
SMART: experiment with target count (2 vs 3) and target emptiness; iterate until the detected violation
is unambiguously cross-input-BTB; only then record the row.

## CROSS-INPUT BTB — experiment matrix (user directive, control experiment)
Run THREE configs and compare (demonstrates cross_input_leaks_only isolates cross-input from self):
  1. v2_btb_nomem      : targets MEMORY-FREE, cross_input_leaks_only=true -> pure cross-input BTB.
  2. v2_btb_memtargets : targets WITH memory, cross_input_leaks_only=true -> the same-input v2 path
       EXISTS but must be FILTERED as self; so EITHER nothing (self correctly suppressed) OR, if a
       violation appears, it is a GENUINE cross-input leak (stronger positive). 
  3. control = v2_l1d (standard priming, memory targets) -> same-input v2 IS shown (already done).
Integrity check for 1&2: for any violation, confirm the leaking PAIR are DIFFERENT inputs (leaker!=prober)
via the cross_input report / leak-flow; that is the proof it is cross-input, not self.
Interpretation: (2) showing nothing while (3) shows v2 => flag correctly removes self-inflicted.
(2) or (1) showing a cross-input pair => genuine cross-input BTB training leak (the generalized-priming win).

## CROSS-INPUT BTB — flag-reason nuance (user directive)
A program may have BOTH a self-inflicted (same-input v2) leak AND a cross-input leak. With
cross_input_leaks_only=true, the violation is kept only when its LOCALIZED boundary is non-self
(leaker != prober). So the integrity check is NOT "no self leak present" — it is:
  THE FLAGGED/LOCALIZED BOUNDARY IS CROSS-INPUT (the two members are DIFFERENT inputs).
Verify per flagged violation: read the cross_input localizer's returned pair (indices) and confirm they
are distinct inputs; that distinct-input boundary is the proven reason for the flag, even if a same-input
leak also exists in the same test case. This applies to v2_btb_memtargets especially (targets access
memory -> self leak may coexist, but the flag must be the cross-input boundary).
