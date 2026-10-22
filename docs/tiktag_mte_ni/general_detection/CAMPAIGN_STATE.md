# Autonomous TikTag campaign — state & standing orders

GOAL: collect real TikTag-v1 (and other MTE tag) leaks via fully-random MTE non-interference fuzzing on
the Pixel-8 Cortex-X3, triage each with 100% integrity, keep only GENUINE ones, disassemble and root-cause
them, and loop forever until the user explicitly says stop. No fallbacks, no lying, re-check everything,
verify no noise, use controlled "games" (the all-tags sweep) for PoCs. Think outside the box.

## Timezone / phases
- Server clock is UTC. User local = UTC+3. User token reset = 10:20 PM local = **19:20 UTC**.
- COLLECT phase (now .. 19:20 UTC): only collect violations (cheap). Keep my own token use minimal.
- TRIAGE phase (>= 19:20 UTC): triage collected violations, disassemble GENUINE ones, then keep
  collecting + triaging in a loop, forever.

## Moving parts (all under docs/tiktag_mte_ni/general_detection/)
- `collect_daemon.sh` -> eval_out/collect/ : self-healing fuzz --nonstop collector + Pixel keepalive +
  crash-restart + device module-reload recovery. Honors eval_out/collect/PAUSE. PID in
  eval_out/collect/daemon.pid; current fuzz PID in eval_out/collect/fuzz.pid. Log: eval_out/collect/collect.log.
- `triage_batch.sh` : set PAUSE, kill fuzz by tracked PID (PID-safe), wait device free, run auto_triage.py
  on every violation-* lacking AUTO_TRIAGE.md, persist GENUINE to eval_out/collect/GENUINE/, remove PAUSE.
  Log: eval_out/collect/triage.log. Ledger: eval_out/collect/TRIAGE_LEDGER.tsv.
- `auto_triage.py` : tag-only diff + CE spec-only gate + all-tags device sweep -> GENUINE/QUESTIONABLE/NOISE.
  GENUINE requires: tag-only; a diverging set that is CE spec-only (or prefetch of a speculative tag-sealed
  access); and the matching CHECKED tag dominating every checked mismatch (TCMA tags 0/15 excluded).
- Config: `s24_mte_random.yml` (fully random, cond, view_rotation ON via enable_pre_run_flush:true,
  mistraining OFF, inputs_per_class 4, sample 2000, pinned X3 core 8).

## What each 30-min cron tick must do (standing orders)
1. Re-read THIS file first (anti-drift). 
2. HEALTH (verify REAL progress, not just liveness — a process can be alive but hung):
   - daemon alive? `kill -0 $(cat eval_out/collect/daemon.pid)`. If dead, relaunch:
     `setsid nohup bash docs/tiktag_mte_ni/general_detection/collect_daemon.sh >/dev/null 2>&1 &`.
   - heartbeat FRESH? the last `heartbeat` line in collect.log must be within ~2 cycles (<~120s old). If
     stale, the daemon is stuck -> kill its PID (verified) and relaunch.
   - fuzz PROGRESSING? across the recent heartbeats, `cpu=` seconds AND `loglines=` must be INCREASING
     (not just fuzz_pid present). The daemon auto-restarts on `stall>=4`; if you see a hang it did not
     catch, force it: kill the tracked fuzz PID (verified python revizor) and let the daemon restart.
   - device reachable? `timeout 15 adb -s 47061FDJH001K3 shell true`. If not, the daemon's touch logs
     "touch FAIL"; try a module reload (recover_dev) only if the device is truly stuck.
   - PID-safety: never pattern-kill; only kill PIDs read from daemon.pid/fuzz.pid and verified via
     /proc/<pid>/exe (python) and cmdline (revizor.py fuzz); exclude bash/pgrep/grep lines.
   The daemon now runs SLEEP=45s with a hang detector (restart if fuzz.log stops growing for 4 cycles).
3. If UTC < 19:20: stop here (token-light). Do NOT triage.
4. If UTC >= 19:20 (TRIAGE phase):
   a. Run `bash triage_batch.sh` (pauses collection, triages all new violations, resumes).
   b. For each NEW violation marked GENUINE in the ledger: disassemble the gadget
      (objdump the enacted_test / walk generated.asm), identify the leaking speculative tag-checked load
      and the downstream probe, and write a per-violation ROOTCAUSE note in its dir explaining the exact
      mechanism (which branch mispredicts, which sealed reg, which set, why the tag check gates it).
      Re-verify genuineness is not noise (controlled all-tags sweep already in AUTO_TRIAGE.md; sanity-check
      the match-vs-checked-mismatch separation by eye).
   c. Update the "## Progress log" below with counts + any genuine finds.
5. NEVER stop the campaign on my own. Only the user stops it.
6. AGENTS / MULTI-MODEL: only AFTER 19:20 UTC (token reset) may I spawn subagents or consult multiple
   models — for triaging batches in parallel, adversarially re-verifying a GENUINE find, root-causing by
   disassembly, or hunting novel tag-leak variants. Before 19:20, stay token-light (health only).
   Think outside the box: new gadget shapes, other tag preconditions, other observation channels.

## Integrity rules (hard)
- 100% genuine only. A violation is GENUINE only if it passes the full pipeline AND the leaking set is
  spec-only AND the checked-tag separation is clean. Prefetcher/batch/arch-demand divergences = NOISE.
- Don't trust a single measurement; the all-tags sweep (16 deltas) IS the recheck. Re-run if marginal.
- Log before and after every experiment; persist every artifact; never fabricate a result.
- Destructive ops (kill) must target explicit PIDs verified as python revizor procs; never broad patterns.

## Pending verification asks (do in TRIAGE phase, first)
1. GRANULE-TAG TRACKING CONTROL (gold-standard genuineness) on the genuine ones (155905, 160128) AND
   every future GENUINE find: besides the pointer-tag all-tags sweep, change the MEMORY (granule) tag of
   the guess load and confirm the detected match delta MOVES with it. `_mte_tags_for` seeds ALL granules
   with one uniform default tag; the sealer's genuine pointer tag matches it. Patch ONLY the guess load's
   granule (index = (CE effective addr of the speculative tag-sealed load - sandbox_base) >> 4,
   MTE_GRANULE=16) to T2, re-run the pointer-delta sweep, verify the peak shifts so pointer==T2. Single
   granule (not global) so architectural sealed accesses still match. Build tag_track.py; make it a
   required GENUINE confirmation.
2. TEMPLATE STRUCTURE CONFIRMED (user_mte_skeleton.asm): random_instructions prior+post the SDIV window,
   prior+post the sealed tagged LDR [x5], and post the entire if. Genuine violations 155905/160128 come
   from this structure.

## Progress log
- [init] Collector launching; genuine baseline already established from the USER TEMPLATE
  (user_mte_skeleton): eval_out/VERIFIED_GENUINE/userskel_155905 (set 9 spec-only, sweep 7:1.00) and
  userskel_160128 (set 39 spec-only, 7:0.58) — both independently confirmed GENUINE TikTag-v1.
- Fully-random so far = NOISE (run24h violation-164234). This campaign tests random at scale.
- [tag-track] GRANULE-TAG TRACKING CONTROL built (tag_track.py) and run on the genuine ones:
  userskel_160128 GENUINE — detection TRACKS the memory tag (T=3/9/12 -> match delta 4/10/13, d0
  collapses); userskel_155905 control INVALID (finder returns granule 0, wrong; d0 unchanged — needs a
  better guess-granule finder: locate the CHECKED load after the gating ADDG whose base==Xd). tag_track.py
  excludes TCMA deltas 0/15 from the tracked peak and has a validity gate (d0 must collapse). Evidence:
  eval_out/VERIFIED_GENUINE/TAG_TRACK.txt. Apply tag_track as a genuine-confirmation on future finds;
  if its finder returns granule 0 / VALIDITY FAIL, fix the finder before trusting the verdict.
