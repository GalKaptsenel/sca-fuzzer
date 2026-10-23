#!/bin/bash
# Self-healing fully-random MTE/NI violation collector + Pixel-8 keepalive + hang detector.
# - runs `revizor.py fuzz --nonstop` (view_rotation ON via enable_pre_run_flush:true) so one session
#   accumulates many violation-* dirs;
# - restarts fuzz if it DIES or HANGS (fuzz.log stops growing) -> device module-reload recovery;
# - touches the device every cycle so the adb tunnel does not drop;
# - honors a PAUSE flag (fast: checked every cycle) so triage can borrow the device, then resumes;
# - logs CPU seconds + fuzz.log growth each heartbeat so "really running" is verifiable.
# PID-safe: tracks the fuzz child by its own PID; never pattern-kills.
set -u
REPO=/home/gal_k_1_1998/revizor/sca-fuzzer
PY=/home/gal_k_1_1998/revizor/revizor-venv/bin/python
CFG=$REPO/docs/tiktag_mte_ni/general_detection/s24_mte_random.yml
BASE=$REPO/docs/tiktag_mte_ni/general_detection/eval_out/collect
LOG=$BASE/collect.log
PAUSE=$BASE/PAUSE
SERIAL=47061FDJH001K3
SLEEP=45
STALL_LIMIT=4          # heartbeats with no fuzz.log growth -> treat as hung
mkdir -p "$BASE"
# SINGLE-INSTANCE LOCK: only one daemon may ever run (prevents the watchdog/@reboot from spawning a
# second daemon -> two fuzz -> parallel /dev/executor access). A second launch fails flock and exits.
exec 9>"$BASE/daemon.lock"
if ! flock -n 9; then
  echo "[$(date -u +%FT%TZ)] another collect_daemon holds the lock; exiting (no duplicate)" >> "$LOG"
  exit 0
fi
echo $$ > "$BASE/daemon.pid"
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
FUZZ_PID=""
CUR_LOG=""
prev_lines=0
stall=0

cpu_secs(){ # utime+stime (ticks) -> seconds for $1
  local s; s=$(awk '{print $14+$15}' /proc/$1/stat 2>/dev/null); [ -n "$s" ] && echo $((s/100)) || echo "?"; }

start_fuzz(){
  local WD="$BASE/session_$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "$WD"
  CUR_LOG="$WD/fuzz.log"
  PYTHONPATH=$REPO setsid nohup "$PY" "$REPO/revizor.py" fuzz -s "$REPO/base.json" -c "$CFG" \
      -i 8 -n 2000000 --nonstop -w "$WD" >> "$CUR_LOG" 2>&1 &
  FUZZ_PID=$!
  echo "$FUZZ_PID" > "$BASE/fuzz.pid"
  prev_lines=0; prev_cpu=0; stall=0
  log "started fuzz pid=$FUZZ_PID wd=$(basename "$WD")"
}
stop_fuzz(){
  if [ -n "$FUZZ_PID" ] && kill -0 "$FUZZ_PID" 2>/dev/null; then
    kill -TERM "-$FUZZ_PID" 2>/dev/null || kill -TERM "$FUZZ_PID" 2>/dev/null
    sleep 3
    kill -9 "-$FUZZ_PID" 2>/dev/null || kill -9 "$FUZZ_PID" 2>/dev/null
    log "stopped fuzz pid=$FUZZ_PID"
  fi
  FUZZ_PID=""; : > "$BASE/fuzz.pid"
}
recover_dev(){
  # su 0 -c (NOT bare `su -c`, which under magisk runs with ZERO effective caps -> insmod EPERM).
  # No shell keywords (if/then/fi): adb re-tokenizes the string so the device sh would see `su 0 -c if`
  # and choke on the bare `then`. An unconditional insmod is safe — if the module is already loaded it
  # fails harmlessly with EEXIST and the LIVE module is left untouched (we NEVER rmmod a working device,
  # which bit us once). Then (re)chmod the nodes.
  timeout 45 adb -s "$SERIAL" shell su 0 -c \
    'insmod /data/local/tmp/revizor-executor.ko 2>/dev/null; chmod 777 /dev/executor 2>/dev/null; chmod -R a+rw /sys/executor 2>/dev/null; ls /dev/executor' >> "$LOG" 2>&1 \
    && log "device recover attempted (insmod-if-absent + chmod; ls result above)" || log "device recover cmd failed"
}

trap 'stop_fuzz; log "DAEMON EXIT"; exit 0' TERM INT
log "DAEMON START pid=$$ cfg=$(basename "$CFG") sleep=${SLEEP}s"
start_fuzz
while true; do
  if [ -f "$PAUSE" ]; then
    [ -n "$FUZZ_PID" ] && kill -0 "$FUZZ_PID" 2>/dev/null && stop_fuzz
    # keepalive ONLY while paused — during an active run the fuzzer's own adb traffic keeps the tunnel
    # warm, and an extra concurrent `adb shell` interleaves with its batch reads (bad-magic responses).
    timeout 20 adb -s "$SERIAL" shell true >/dev/null 2>&1 || log "touch FAIL (device unreachable, paused)"
    log "PAUSED (triage borrowing device); holding"
  else
    if [ -z "$FUZZ_PID" ] || ! kill -0 "$FUZZ_PID" 2>/dev/null; then
      log "fuzz not running -> recover + restart"; recover_dev; start_fuzz
    else
      # hang detection: use fuzz.log BYTE size (the progress line rewrites with \r, so line count
      # barely grows even while working) AND cpu-time advance; hung only if NEITHER moves.
      lines=$(stat -c %s "$CUR_LOG" 2>/dev/null || echo 0)
      cpu_now=$(cpu_secs "$FUZZ_PID"); [ "$cpu_now" = "?" ] && cpu_now=0
      if [ "$lines" -le "$prev_lines" ] && [ "$cpu_now" -le "${prev_cpu:-0}" ]; then
        stall=$((stall+1))
        if [ "$stall" -ge "$STALL_LIMIT" ]; then
          log "fuzz HUNG (no log-byte growth AND no cpu advance ${STALL_LIMIT}x, cpu=${cpu_now}s) -> restart"
          stop_fuzz; recover_dev; start_fuzz
        fi
      else
        stall=0
      fi
      prev_lines=$lines; prev_cpu=$cpu_now
    fi
  fi
  V=$(find "$BASE" -type d -name 'violation-*' 2>/dev/null | wc -l)
  log "heartbeat fuzz_pid=${FUZZ_PID:-none} cpu=$([ -n "$FUZZ_PID" ] && cpu_secs "$FUZZ_PID" || echo -)s logbytes=${prev_lines} stall=${stall} paused=$([ -f "$PAUSE" ] && echo 1 || echo 0) violations=$V"
  sleep "$SLEEP"
done
