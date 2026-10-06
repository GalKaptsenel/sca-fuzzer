---
name: pac-ni-campaign
description: Run and autonomously watch a long PAC non-interference fuzzing campaign on the Neoverse N3, and triage every violation it saves — classifying it as a GENUINE PAC-sourced speculative leak vs hardware noise by reproducing it under all three priming strategies (regular / cross-input both / cross-input only) and root-causing it in the contract executor. Use when asked to hunt for a PAC leak with cross-input priming, keep such a campaign alive, detect hangs/zombies, or analyze its violation-* dirs. Covers the hard device-exclusivity, PID-safety, and launch pitfalls. Pairs with revizor-violation-triage / revizor-leak-flow for deeper per-violation analysis.
---

# PAC non-interference campaign + autonomous watchdog/triage

Goal: find a leak whose **source is PAC** — a speculative `AUT*` on a forged (decoy) signature that
evicts a cache set the genuine path does not. Driven by the non-interference fuzzer with cross-input
priming (the new galloping+bisection localizer), keeping **both** self-dependent and cross-input leaks.

Tools (all in `tools/campaign/`, repo-relative, device-safe):
- `launch_campaign.sh <config.yml> ['key: val' ...]` — create a detached campaign + its `run.sh`.
- `monitor.sh <campaign_dir>` — health/progress check; restarts a dead/zombie/stuck campaign.
- `analyze_violation.sh <campaign_dir>` — exclusive-device triage of every un-analyzed `violation-*`.
- `tools/analyze_violation.py <violation_dir> mode <regular|new_both|new_xinput> | rootcause`.

## Launch

```
tools/campaign/launch_campaign.sh config_pac_xinput.yml 'cross_input_leaks_only: false'
```

`config_pac_xinput.yml` = PAC category, `fuzzer: non-interference`, `enable_cross_input_priming: true`,
P+P, `inputs_per_class: 2`. `cross_input_leaks_only: false` keeps self + cross-input. The campaign dir
is `~/revizor/campaigns/<cfg>_<ts>/`; find the current one with `ls -d ~/revizor/campaigns/*_* | tail -1`.

## Watch (every ~30 min)

`monitor.sh <D>` prints a verdict: `OK` | `RESTARTED:*` | `STUCK_RESTARTED:*` | `RESTART_FAILED:*` |
`NEW_VIOLATIONS:n`. On `NEW_VIOLATIONS` run `analyze_violation.sh <D>`; on a restart/failure, notify.
A session-only cron can drive this; it dies with the session, so re-create it per session.

## Triage a violation — what each priming strategy says + the root cause

`analyze_violation.sh <D>` reproduces each violation with the **full input sequence, same order, same
config** (regenerated from `input_gen_seed`; byte-identical) through the real `fuzzing_round`, under:
- **regular** priming (`enable_cross_input_priming=false`) — the swap test,
- **new both** (`cross_input_priming`, `leaks_only=false`) — self + cross,
- **new cross-input only** (`leaks_only=true`).

Then it root-causes in the CE: the detecting input's genuine vs decoy (and the strongest
`forced_noncanon`) L1D sets. A **decoy-only speculative set = the PAC channel = a genuine PAC leak**.
No decoy-only set = not PAC (noise/other). Result is written to `<violation>/analysis.md`.

## CRITICAL pitfalls (learned the hard way)

- **PID-ONLY, never by name.** Select the campaign only by the PID in `<D>/pid` (written by `run.sh`
  as `$$`, preserved across `exec`), confirmed by reading `/proc/<pid>/cmdline` for the config path.
  `pkill`/`pgrep -f` by name once killed the controlling shell itself. The monitor/analyzer never
  name-match a process.
- **The device is a single shared resource — never two experiments at once.** `/dev/executor` serves
  one measurer. The analyzer holds `flock <D>/device.lock`, TERMINATES the campaign, verifies no other
  `/proc/*/fd` points at `/dev/executor`, analyzes, then RELAUNCHES.
- **Do NOT pause (SIGSTOP) the campaign for analysis.** A stop mid-batch leaves input ids allocated on
  the device; the analyzer's fresh executor calls `discard_all_inputs`, so on resume the campaign's
  `checkout_input` hits `EINVAL` and dies. Terminate + relaunch instead (a fresh seed keeps exploring).
- **Close the lock fd when launching.** `setsid nohup run.sh ... 9>&-` — otherwise the campaign inherits
  the `device.lock` flock and the monitor/analyzer block forever on `BUSY`.
- **After `insmod`:** `sudo chmod 777 /dev/executor` AND `sudo chmod -R a+rw /sys/executor` (the fuzzer
  writes sysfs knobs at startup). The module exports raw `tcr_el1`/`id_aa64isar1_el1`/`id_aa64isar2_el1`;
  the PAC profile is decoded from them (or from the config if fully specified) — see the PAC model skills.
- **Reproducing from the per-TC `Program seed:` does NOT regenerate the TC.** Use the saved
  `generated.asm` + regenerate inputs from `input_gen_seed` (what the analyzer does).
- **Be skeptical:** re-verify progress and any "confirmed PAC leak" before acting; HW noise can pass
  run-time priming once yet be rejected on reproduction by all three strategies (seen on the first hit).
