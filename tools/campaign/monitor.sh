#!/bin/bash
# Campaign health + progress checker. PID-ONLY: the campaign's identity is the PID written by run.sh
# ($$, preserved across exec); it is confirmed by reading THAT one PID's /proc/<pid>/cmdline. No name
# scan (pgrep/pkill) anywhere, so this can never collide with the caller's own shell. Holds an
# exclusive flock so it never overlaps another monitor run or an analysis.
# Usage: monitor.sh <campaign_dir>
# Verdicts: OK | RESTARTED:<reason> | STUCK_RESTARTED:<reason> | RESTART_FAILED:<reason> | NEW_VIOLATIONS:<n>
set -u
D="${1:?campaign dir required}"
D="$(cd "$D" 2>/dev/null && pwd)" || { echo "bad campaign dir"; exit 2; }
PIDFILE="$D/pid"; STATE="$D/monitor_state"; LOG="$D/fuzz.log"; RUN="$D/run.sh"; LOCK="$D/device.lock"
CFG="$D/config.yml"

exec 9>"$LOCK"
flock -w 60 9 || { echo "BUSY: another monitor/analysis holds the lock"; exit 0; }

now=$(date +%s)
pid=$(cat "$PIDFILE" 2>/dev/null)

# A pid IS our campaign iff that exact pid is alive, not a zombie, and its own cmdline names this config.
is_campaign() {
    local p="$1"
    [[ "$p" =~ ^[0-9]+$ && "$p" -gt 1 && -d "/proc/$p" ]] || return 1
    [[ "$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)" == "Z" ]] && return 1
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "$CFG"
}

alive=0; is_campaign "$pid" && alive=1
cur_tc=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE '[0-9]+ tc ' | grep -oE '^[0-9]+' | tail -1)
cur_tc=${cur_tc:-0}
cpu_ticks=0
if [[ "$alive" -eq 1 ]]; then
    for p in $(pstree -p "$pid" 2>/dev/null | grep -oE '\([0-9]+\)' | grep -oE '[0-9]+'); do
        t=$(awk '{print $14+$15}' "/proc/$p/stat" 2>/dev/null); cpu_ticks=$((cpu_ticks + ${t:-0}))
    done
fi

last_pid=0; last_tc=0; last_cpu=0; last_epoch=0
[[ -f "$STATE" ]] && read -r last_pid last_tc last_cpu last_epoch < "$STATE"
elapsed=$((now - last_epoch))
emit() { echo "$1"; printf 'pid=%s alive=%s tc=%s cpu_ticks=%s since_last=%ss violations=%s\n' \
         "$pid" "$alive" "$cur_tc" "$cpu_ticks" "$elapsed" "$(ls -d "$D"/violation-* 2>/dev/null | wc -l)"; }
save() { echo "$pid $cur_tc $cpu_ticks $now" > "$STATE"; }

restart() {   # kill the OLD campaign's process group (only if it is really ours), then relaunch
    if is_campaign "$pid"; then kill -TERM "-$pid" 2>/dev/null; sleep 3; is_campaign "$pid" && kill -KILL "-$pid" 2>/dev/null; fi
    : > "$PIDFILE"                                   # run.sh will write the authoritative new pid
    setsid nohup "$RUN" >> "$LOG" 2>&1 < /dev/null 9>&- &
    for _ in $(seq 1 20); do sleep 1; pid=$(cat "$PIDFILE" 2>/dev/null); is_campaign "$pid" && break; done
    cur_tc=0; cpu_ticks=0
    is_campaign "$pid"
}

nv=$(ls -d "$D"/violation-* 2>/dev/null | wc -l)
seen=$(cat "$D/violations_seen" 2>/dev/null || echo 0)

if [[ "$alive" -eq 0 ]]; then
    if restart; then emit "RESTARTED:process_gone_or_zombie"; else emit "RESTART_FAILED:process_gone"; fi
    save; exit 0
fi
# Stuck = alive but neither tc nor CPU time advanced for >= 20 min (deadlock / silent wedge; a slow but
# working run still burns CPU). A violation's priming/reproduce phase burns CPU, so it is not flagged.
if [[ "$last_pid" == "$pid" && "$last_epoch" -ne 0 && "$elapsed" -ge 1200 && "$cur_tc" -le "$last_tc" && "$cpu_ticks" -le "$last_cpu" ]]; then
    if restart; then emit "STUCK_RESTARTED:no_progress_${elapsed}s"; else emit "RESTART_FAILED:stuck"; fi
    save; exit 0
fi
save
[[ "$nv" -gt "$seen" ]] && { emit "NEW_VIOLATIONS:$((nv - seen))"; exit 0; }
emit "OK"
