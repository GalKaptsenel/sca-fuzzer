#!/bin/bash
# Analyze every not-yet-analyzed violation of a campaign, with EXCLUSIVE device access.
#
# The device is a single shared resource and the analysis resets its state, so we must not measure while
# the campaign measures. Pausing is unsafe (a SIGSTOP mid-batch leaves device-side input ids the analysis
# then wipes -> the campaign crashes on resume). So: hold the lock, TERMINATE the campaign, verify no
# other process holds the device, analyze, then RELAUNCH the campaign (fresh seed -> keeps exploring).
#
# Reproduction is faithful: same program (generated.asm), the full input sequence regenerated in the same
# order from input_gen_seed, same config; fuzzing_round boosts + runs each priming strategy.
#
# Usage: analyze_violation.sh <campaign_dir>
set -u
D="${1:?campaign dir required}"
D="$(cd "$D" 2>/dev/null && pwd)" || { echo "bad campaign dir"; exit 2; }
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (tools/campaign/..)
CFG="$D/config.yml"; PIDFILE="$D/pid"; RUN="$D/run.sh"; LOG="$D/fuzz.log"; LOCK="$D/device.lock"

exec 9>"$LOCK"; flock -w 300 9 || { echo "BUSY: lock held"; exit 3; }
pid=$(cat "$PIDFILE" 2>/dev/null)

is_campaign() { local p="$1"; [[ "$p" =~ ^[0-9]+$ && "$p" -gt 1 && -d "/proc/$p" ]] || return 1
    [[ "$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)" == "Z" ]] && return 1
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "$CFG"; }

device_holders() {  # PIDs (one per line) with /dev/executor open, excluding $1
    local excl="$1" q
    for f in /proc/[0-9]*/fd/*; do [[ "$(readlink "$f" 2>/dev/null)" == "/dev/executor" ]] || continue
        q=$(echo "$f" | cut -d/ -f3); [[ "$q" != "$excl" && "$q" != "$$" ]] && echo "$q"; done | sort -u
}

relaunch() { : > "$PIDFILE"; setsid nohup "$RUN" >> "$LOG" 2>&1 < /dev/null 9>&- &
    for _ in $(seq 1 20); do sleep 1; pid=$(cat "$PIDFILE" 2>/dev/null); is_campaign "$pid" && break; done; }

# 1. terminate the campaign (whole group) so the device is ours alone
if is_campaign "$pid"; then kill -TERM "-$pid" 2>/dev/null; sleep 3; is_campaign "$pid" && kill -KILL "-$pid" 2>/dev/null; fi
sleep 1
# 2. verify exclusivity — abort rather than measure against interference
h=$(device_holders "$pid")
if [[ -n "$h" ]]; then echo "ABORT: /dev/executor still held by:$h — not analyzing"; relaunch; echo "relaunched pid=$pid"; exit 4; fi

# 3. analyze each violation that has no analysis.md yet
cd "$R" || { echo "no repo"; exit 1; }
source venv/bin/activate
analyzed=0
for V in $(ls -d "$D"/violation-* 2>/dev/null); do
    [[ -f "$V/analysis.md" ]] && continue
    {
        echo "# Violation analysis: $(basename "$V")"
        echo; echo "Reproduced with the full input sequence (same order, same config); each priming"
        echo "strategy run on the reproduced violation."; echo
        echo '## Priming comparison'
        for m in regular new_both new_xinput; do
            timeout 600 python tools/analyze_violation.py "$V" mode "$m" 2>/dev/null | grep -E "^MODE|cross-input priming:"
        done
        echo; echo '## Root cause (CE model)'
        timeout 600 python tools/analyze_violation.py "$V" rootcause 2>/dev/null \
            | grep -E "PAC slots|genuine L1D|random-decoy|forced-noncanon|set [0-9]|VERDICT|not a PAC"
    } > "$V/analysis.md" 2>&1
    echo "analyzed $(basename "$V")"; grep -E "^MODE|VERDICT" "$V/analysis.md" | sed 's/^/    /'
    analyzed=$((analyzed + 1))
done

# 4. relaunch and record how many violations are now accounted for
relaunch
echo "$(ls -d "$D"/violation-* 2>/dev/null | wc -l)" > "$D/violations_seen"
echo "DONE: analyzed=$analyzed, campaign relaunched pid=$pid, violations_seen=$(cat "$D/violations_seen")"
