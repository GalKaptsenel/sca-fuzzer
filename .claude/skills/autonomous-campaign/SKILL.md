---
name: autonomous-campaign
description: Run a long unattended fuzzing+triage campaign on the Pixel-8 (or any single exclusive device) the way Gal demands — collect violations continuously, triage each to a 100%-integrity verdict, root-cause the genuine ones, loop forever until told to stop, and survive reboots/crashes/session loss. Use when asked to "run the campaign autonomously", keep a hunt alive across hours/days, or set up self-healing collection + scheduled triage. Covers the collect/triage phase model, the self-healing daemon, monitors, reboot survival, PID-safety, anti-drift state, and the hard integrity standard. Pairs with mte-tiktag-triage.
---

# Autonomous campaign operation (Gal's standard)

Goal: never give up. Relentlessly collect and triage until the user says stop, finding genuine leaks
(TikTag and other tag-leak variants) with zero integrity compromise. Be pessimistic but goal-oriented;
think outside the box (new gadget shapes, other preconditions/channels); be proactive; stay positive.

## Phase model (respect token windows)
- **COLLECT** (cheap, no tokens): a background daemon fuzzes `--nonstop` and accumulates violation dirs.
  Do this during any window where heavy work must be conserved (e.g., before a token reset).
- **TRIAGE** (expensive, needs the Claude session): after the reset / when tokens allow, triage every new
  violation with the `mte-tiktag-triage` pipeline, disassemble + root-cause the genuine ones, run the
  memory-tag tracking control, then resume collecting. Loop.
- The two never share the device simultaneously — triage PAUSES collection first.

## The self-healing collection daemon (`collect_daemon.sh`)
Runs `revizor.py fuzz --nonstop` with the VERIFIED working config (view_rotation ON via
`enable_pre_run_flush: true`), and every cycle: touches the device (keepalive), restarts fuzz if it DIED,
and restarts it if it HUNG. Honors a `PAUSE` flag so triage can borrow the device.
- **Hang detection must use log BYTE size and/or CPU-time advance — NOT line count.** The fuzzer's
  progress line rewrites in place with `\r`, so `wc -l` barely grows while it is working fine; a
  line-count detector false-restarts a healthy run. Use `stat -c %s` on the fuzz log and/or utime+stime
  from `/proc/<pid>/stat`; declare hung only if NEITHER advances for several cycles.
- Keep the cycle short (~45s) so PAUSE is honored quickly; for an immediate pause, kill the tracked fuzz
  PID directly (don't wait a cycle) AND set PAUSE so the daemon won't restart it.

## Monitors (verify REAL progress, not just liveness)
A process can be alive but hung. Each periodic health tick (a ~30-min cron re-reading the state file)
must confirm: daemon PID alive; last heartbeat FRESH (< ~2 cycles old); fuzz `cpu=` AND log bytes
INCREASING across recent heartbeats; device reachable (`adb shell true`). Relaunch the daemon if dead;
force-restart fuzz (PID-safe) if truly hung. Report one line; do NOT triage before the allowed window.

## Reboot / crash / session-loss survival
- A **host crontab** makes collection outlive this Claude session and reboots:
  `@reboot /bin/bash <campaign_boot.sh>` and `*/5 * * * * /bin/bash <campaign_boot.sh>` (a watchdog).
  `campaign_boot.sh` is idempotent — it (re)launches the daemon only if its tracked PID is dead.
- Collected violations persist on disk and are committed periodically, so triage can always catch up after
  a reboot; triage itself needs a live session (it requires reasoning), so it resumes when one is alive.
- The external adb tunnel (laptop → Pixel) is out of our control; after a box reboot the daemon logs
  `touch FAIL` and keeps trying until the tunnel returns. State this honestly; don't claim what we can't.
- The in-session cron (CronCreate) is session-only — re-arm it when a session resumes.

## PID-SAFETY (destructive ops) — you WILL match yourself if careless
- NEVER pattern-kill (`pkill -f fuzz`, broad `grep`). Your own `bash -c` wrapper and your `grep/pgrep`
  line contain the very strings you search for and will be matched/killed. This bit us.
- Kill only PIDs read from `daemon.pid`/`fuzz.pid`, and VERIFY each via `/proc/<pid>/exe` (python) and
  `/proc/<pid>/cmdline` (contains `revizor.py fuzz`). Exclude anything whose cmdline has `bash`, `pgrep`,
  `grep`, or `cmdline`. List-then-kill; never kill from a pattern match directly.

## Anti-drift / anti-hallucination
Keep a persistent `CAMPAIGN_STATE.md`: goal, phases, standing orders, integrity rules, pending asks,
progress log. RE-READ it first on every wake — it is the anchor against drift when context is summarized.
Log every step before and after. Convert relative times to absolute (UTC + the user's local offset).
Write results to the state file and the violation dirs, never only to chat.

## Integrity standard (hard — this is the whole point)
100% genuine only. No fallbacks, no quiet returns, no lying, ever. Don't trust a single measurement —
recheck, verify no noise, run the controlled "games" (fixed-prefix swaps, all-tags sweeps, memory-tag
tracking) that a PoC demands. Declare invalid controls inconclusive, not results. Only the user stops the
campaign — never give up on your own. If something is uncertain, say so and keep working to resolve it.

## After the reset you MAY spawn agents / consult multiple models
For parallel triage of a backlog, adversarial re-verification of a GENUINE find, root-causing by
disassembly, or hunting novel variants. Before the reset, stay token-light (health only).
