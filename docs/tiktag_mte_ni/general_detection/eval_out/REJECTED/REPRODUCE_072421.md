# violation-261023-072421 — master reproduction log

Genuine speculative MTE pointer-vs-memory tag-check leak, found by the fully-random Revizor
non-interference fuzzer on a Pixel-8 Cortex-X3. GAIN / present-on-mismatch polarity. This file is the
single index to reproduce it end to end: paths, device setup, every experiment with its command/result,
time-to-detection, the module build recipe, and the full artifact inventory. All paths are under the repo
root /home/gal_k_1_1998/revizor/sca-fuzzer unless noted.

## Verdict
GENUINE, device-confirmed, all controls passed except one openly-inferential sub-point (prefetcher's role
in burst contiguity, Proof 1b, not completed). Transmit = a speculative access burst (NOT a quiet HW
prefetch stream — PMU-refuted). Strictly tag-gated, speculative (never retires), two-sided tracking,
tag-check-off collapses it.

## Time to detection
- Stable continuous fully-random campaign start (daemon 77242): 2026-10-22 22:02:40 UTC
- violation-261023-072421 collected:                           2026-10-23 07:24:21 UTC
- wall-clock from stable start: 9h 22m ; effective fuzzing (minus 24m logged triage pauses): ~8h 57m
- from the first (crashed/restructured) launch 2026-10-22 17:55 UTC: ~13h 28m
- config: s24_mte_random.yml (fully random). Prior 8 fully-random violations were all NOISE.

## Key paths
- Violation (persistent snapshot): docs/tiktag_mte_ni/general_detection/eval_out/VERIFIED_GENUINE/fuzzer_072421/
  (generated.asm, input_0000..0031.reif, input_*_trace_log.txt, enacted_test0001/0009/0017/0025.bin, report.txt)
  Original collected dir: docs/tiktag_mte_ni/general_detection/eval_out/collect/session_20261023_070340/violation-261023-072421/
- Config: docs/tiktag_mte_ni/general_detection/s24_mte_random.yml (fuzzer non-interference, cond, l1d, F+R,
  superbatch_size 1, pinned core 8, view_rotation on via enable_pre_run_flush, min/max_bb 6/10)
- Writeups (this dir): VERDICT_072421.md (verdict + full chain), DEEPDIVE_072421.md (all seals/windows/
  combinations), DEVICE_CONTROLS_072421.md (confirmation reps), MECHANISM_PROOF_072421.md (PMU + proofs),
  TCF_OFF_AB_072421.txt + TCF_OFF_WORKLOG.txt (tag-check-off), GD_CHECK/POS/CTRL_STORE_TRACK/GRANSCAN/
  TRACK2D/SHAM_AND_RECONFIRM/072421_followup.txt (measurement logs).
- Analysis/measurement scripts (repo, persistent): docs/tiktag_mte_ni/general_detection/{gd_check.py,
  ctrl_pos_tag.py (fixed-prefix swap-only-suspect; --decoy), ctrl_store_track.py (position-controlled gate
  sweep + granule retag; --gate-off), granule_scan.py, track_pin.py (single-seal tracking, other seals
  pinned TCMA-unchecked), ce_set_class.py (CE arch/spec set classifier), tag_track_rc.py, tag_track_2d.py,
  pmu_diff.py (match-vs-mismatch PMU + residency)}.
- Session scratchpad snapshot (agent drivers/controls, persisted here): eval_out/session_scratchpad_snapshot/
  (combo_matrix.py, confirm3.py, window_probe.py, ce_trace.py, byhand.py, decode_enacted.py, ctl_*.asm/.bin/
  .reif hand-built controls, disasm_000*.txt, proof2_set3_sweep.txt, item2_track.log, CE/proof drafts).
- Pixel-8 GKI module build env (home, persistent across restart; NOT in git): ~/pixel-mod-build/
  (rebuild_abi2.sh; kernel-src + out/ prepared GKI tree; config.gz; gki_artifacts/Module.symvers). Test
  variant tree: ~/pixel-mod-build/executor-src-pftest. Canonical module md5 5c8dfec2646f075c0328a6de2fa77602,
  vermagic 5.15.137-android14-11-gbc062a78e195, device path /data/local/tmp/revizor-executor.ko.

## Device setup
- Serial 47061FDJH001K3 (zuma / Cortex-X3), adb over reverse-SSH tunnel to 127.0.0.1:5037. Pinned core 8.
- Module: insmod via `su 0` (NEVER bare `su -c`); `chmod 777 /dev/executor; chmod -R a+rw /sys/executor`.
- Single-access invariant: pause the collect campaign (touch eval_out/collect/PAUSE) and stop/ SIGSTOP the
  daemon before any manual device run; never two device ops at once.
- venv: /home/gal_k_1_1998/revizor/revizor-venv/bin/python ; PYTHONPATH=/home/gal_k_1_1998/revizor/sca-fuzzer.

## Mechanism (proven)
Program exits via TBZ x0,#58 not-taken; bb_0.7/0.8 never retire (purely speculative). One active window W2
holds three tag-sealed speculative accesses: A (LDR[x4], granule 0, demand set 0), B (STR[x1], granule 0,
demand set 0), D (LDP[x2], granule 305, demand set 12). Data chain A->D (A's loaded value feeds D's pointer).
On a tag MATCH each access proceeds, window short, footprint narrow (demand sets only). On a MISMATCH the
check fails and the window becomes a large speculative access burst (MEM_ACCESS 12 -> ~54,000; L2D refills
only ~10x) that fills a contiguous L1D block (A/B -> {1,4,5,6,8,9,10,11}; D -> {4..28}). Read at a probe set
inside the burst (A set 1, B set 4, D set 9) since the demand set is lit on match too. Set 3 (arch demand,
granule 15) is evicted on mismatch as collateral cache pressure. Match delta uimm4=7 (ptr tag (15+7)%16=6=
memtag); only tag 15 TCMA-unchecked (TCMA1).

## Experiment log (all position-controlled: 14 genuine-variant prefix, suspect #1 swapped at tail; UTC)
- 2026-10-23 ~05:40-06:00  gd_check / early exam (REJECTED 021502 path; see eval_out/REJECTED/)
- ~07:24  072421 collected by the fuzzer
- ~08:00  gd_check gate-independent on 072421: decoys+noncanon gain {1,4,5,6,8,9,10,11} (GD_072421.txt)
- ~08:0x  ctrl_pos_tag fixed-slot vs noncanon + random decoy: Δtag +0.6..+1.0, Δpos ~0 (POS_072421.txt,
          072421_followup.txt) -> tag, not batch position
- ~08:0x  ctrl_store_track --gate-off 0x120 granule 305, tags 3/6/9/12: match delta tracks (7+T-6)%16, 4/4
          (TRACK2D_072421_set9_gran305.txt); granule_scan: only granule 305 tracks (GRANSCAN_072421.txt)
- ~08:xx  6-model paper audit (3 GEN/1 ART/2 Q) -> device experiments resolved the dissents
- ~09:00  DEVICE_CONTROLS: 155905 positive control opposite polarity (3 reps); granule-305 tracking 5 reps
          total; non-MTE control assessed; tag-check-off assessed (DEVICE_CONTROLS_072421.md)
- ~10:48  TAG-CHECK-OFF (kernel TCO=1 test module, built via rebuild_abi2.sh + runtime param tco_force,
          loaded su 0, ORIGINAL restored md5 5c8dfec): checks ON Δtag +1.0 -> checks OFF Δtag ~0, 3+3 reps
          (TCF_OFF_AB_072421.txt, TCF_OFF_WORKLOG.txt) -> tag check IS the cause
- ~11:00-14:00  DEEPDIVE (disassembly + CE + by-hand + HW matrix: per-seal isolation, anchor-move,
          combinations, W1/W2) (DEEPDIVE_072421.md)
- ~14:00-15:00  MECHANISM_PROOF: PMU differential (MEM_ACCESS floods ~4500x, L2D ~10x -> speculative access
          burst, NOT quiet prefetcher); window-dependence refuted (tag-gated, not view_rotation); set-3
          collateral; Proof 1b (CPUACTLR6_EL1[41] prefetcher-disable) NOT completed -> test module wedged on
          unload -> adb reboot (MECHANISM_PROOF_072421.md). Device offline after; auto-recovery cron armed.

## Reproduce each measurement (device reachable, campaign paused, sole user)
PY=/home/gal_k_1_1998/revizor/revizor-venv/bin/python ; export PYTHONPATH=/home/gal_k_1_1998/revizor/sca-fuzzer
D=docs/tiktag_mte_ni/general_detection ; VD=$D/eval_out/VERIFIED_GENUINE/fuzzer_072421
- gate-independent:  $PY $D/gd_check.py      $D/s24_mte_random.yml $VD 1 --reps 4000
- position vs tag:   $PY $D/ctrl_pos_tag.py  $D/s24_mte_random.yml $VD 1 "1,4,8,9" --reps 3000 --prefix 14 --decoy noncanon
- two-sided track:   $PY $D/ctrl_store_track.py $D/s24_mte_random.yml $VD 1 9 305 --gate-off 0x120 --tags 3,6,9,12 --reps 2000 --prefix 14
- granule scan:      $PY $D/granule_scan.py  $D/s24_mte_random.yml $VD 1 9 0x120 "287,305,189,...,255" --T 9 --reps 2000 --prefix 14
- per-seal pinned:   $PY $D/track_pin.py ... (seal A/B with the other pinned to TCMA-unchecked)
- PMU differential:  $PY $D/pmu_diff.py ...  (counts L2D_CACHE_REFILL vs MEM_ACCESS on match vs mismatch)
- tag-check-off:     build the tco_force test module (edit measurement.c per-window disable_TCO_bit ->
                     if(tco_force) enable_TCO_bit(); rebuild via ~/pixel-mod-build/rebuild_abi2.sh; su 0
                     rmmod+insmod; echo 1 > /sys/module/revizor_executor/parameters/tco_force; sweep; restore
                     the canonical .ko md5 5c8dfec), see TCF_OFF_WORKLOG.txt.

## Artifact inventory for complete reproduction (all persistent, in repo except the home build env)
1. Violation: eval_out/VERIFIED_GENUINE/fuzzer_072421/ (asm + inputs + enacted bins + report).
2. Config: s24_mte_random.yml.
3. Scripts: the measurement tools above + session_scratchpad_snapshot/ drivers.
4. Module: canonical .ko (md5 5c8dfec) on device + build recipe ~/pixel-mod-build/rebuild_abi2.sh.
5. Writeups + measurement logs: this directory.

## Open / inferential
- Proof 1b (direct prefetcher-disable via CPUACTLR6_EL1[41]) not completed; the prefetcher's role in making
  the burst contiguous is inferential (PMU ratio argues it is mostly the burst itself).
- W1 (CBNZ window) is CE-modeled but produces no HW stream without branch mistraining.
