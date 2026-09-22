
## Campaign (re)established 2026-09-02 21:00 UTC

Full autonomous MTE/TikTag campaign on the Pixel 8 X3 (cpu8), superbatch=100, F+R, cond+bpas,
categories MTE + MTE-TAG-MEM. Config: config_mte_pixel_tiktag.yml.

Runs on the fixed pipeline (this session):
- GCR_EL1.Exclude cleared per run (f6a8af6) + pre-TC reset (843047e) — the architectural ADDG tag fault.
- upper_overflow boundary granule tagged (718e719) — the spill fault.
- empty-boosted TCs skipped in the window (0673289) — the 0-input-unit batch failure.
- windowed path catches HardwareTracingError + falls back to per-TC (aa595d2) — resilience.
- decoy tags 0/15 skipped, TCMA-unchecked (97c88d2) — decoy effectiveness.
- live super-batch progress in the status line (ddf7d6c).

Persistent (non-tmpfs) root: /home/gal_k_1_1998/mte_campaign  (wd/ = workdir+violations, campaign.log,
crash_debug.log, health.log). Runner auto-restarts on a hard crash; health monitor keyed on runner PID +
output. Verified pre-launch: n=250 superbatch=100 completed with 0 tracing errors, both windows crossed.
Triage of any violation is done LOCALLY (not on the Pixel), verifying the divergence is an MTE-tag
difference (per verify-mte-tag-leak).

### Monitors (4 independent, each notifying)
- mon_violations.sh — new violation-* artifact → triage locally
- mon_crashes.sh — new HARD EXIT in crash_debug.log → inspect traceback
- mon_dmesg.sh — device dmesg KASAN/BUG/abort/panic/stall/would-FPAC (re-baselines on reload) → investigate
- mon_health.sh — runner-death / 40-min stall / 30-min heartbeat → confirm progress
All in /home/gal_k_1_1998/mte_campaign/. Robust checks only (saved PID, output content, comm=python).
2026-09-02 21:14 UTC VIOLATION violation-260902-211022 (TC3): source=MTE-tag confirmed (x5 ADDG deltas 7->11,9->5), same-input; divergence scattered (sets 11/48/39, one per decoy) -> leans noise; queued for HW controlled tag-sweep.
2026-09-02 21:20 UTC VIOLATION violation-260902-211745 (TC?): source=MTE-tag confirmed (x5 ADDG 7->0xc), same-input; htrace SEVERELY noisy (~200 distinct patterns/500 reps, divergence scattered across all sets) -> strongly leans NOISE; the F+R measurement itself is unstable for MTE-heavy TCs on X3.
