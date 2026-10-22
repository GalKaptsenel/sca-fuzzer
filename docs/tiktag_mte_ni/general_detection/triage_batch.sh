#!/bin/bash
# Borrow the device from the collector, triage every not-yet-triaged violation, then resume collection.
# Device coordination: create PAUSE (so the daemon will not restart fuzz), kill the current fuzz child by
# its tracked PID (PID-safe, verified to be a python revizor process), wait until the device is free,
# run auto_triage.py on each new violation, persist GENUINE ones, remove PAUSE (daemon resumes).
set -u
REPO=/home/gal_k_1_1998/revizor/sca-fuzzer
PY=/home/gal_k_1_1998/revizor/revizor-venv/bin/python
BASE=$REPO/docs/tiktag_mte_ni/general_detection/eval_out/collect
PAUSE=$BASE/PAUSE
GEN=$BASE/GENUINE
LEDGER=$BASE/TRIAGE_LEDGER.tsv
LOG=$BASE/triage.log
AT=$REPO/docs/tiktag_mte_ni/general_detection/auto_triage.py
mkdir -p "$GEN"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

touch "$PAUSE"
log "PAUSE set; stopping collection fuzz by tracked PID"
if [ -f "$BASE/fuzz.pid" ]; then
  FP=$(cat "$BASE/fuzz.pid" 2>/dev/null)
  if [ -n "$FP" ] && kill -0 "$FP" 2>/dev/null; then
    exe=$(readlink /proc/$FP/exe 2>/dev/null)
    cmd=$(tr '\0' ' ' </proc/$FP/cmdline 2>/dev/null)
    if echo "$exe" | grep -q python && echo "$cmd" | grep -q 'revizor.py fuzz'; then
      kill -TERM "-$FP" 2>/dev/null || kill -TERM "$FP" 2>/dev/null; sleep 3
      kill -9 "-$FP" 2>/dev/null || kill -9 "$FP" 2>/dev/null
      log "killed fuzz PID $FP"
    else
      log "fuzz.pid $FP is not a python revizor fuzz proc (exe=$exe); NOT killing"
    fi
  fi
fi
# wait until the device is free (no python revizor fuzz anywhere), max ~90s
for i in $(seq 1 45); do
  busy=0
  for p in $(pgrep -f 'revizor.py fuzz' 2>/dev/null); do
    e=$(readlink /proc/$p/exe 2>/dev/null); echo "$e" | grep -q python && busy=1
  done
  [ "$busy" = 0 ] && break
  sleep 2
done
log "device free; beginning triage batch"

n=0; gen=0
for VD in $(find "$BASE" -type d -name 'violation-*' 2>/dev/null | sort); do
  [ -f "$VD/AUTO_TRIAGE.md" ] && continue   # already triaged
  [ -f "$VD/org-config.yaml" ] || continue
  n=$((n+1))
  log "triage[$n] $VD"
  PYTHONPATH=$REPO "$PY" "$AT" "$VD/org-config.yaml" "$VD" --reps 2000 --ledger "$LEDGER" \
      >> "$VD/auto_triage.stdout" 2>&1
  rc=$?
  V=$(grep -aoE 'VERDICT=[A-Z]+' "$VD/auto_triage.stdout" | tail -1)
  log "triage[$n] $(basename "$VD") $V rc=$rc"
  if [ "$rc" = 42 ]; then
    gen=$((gen+1))
    dst="$GEN/$(basename "$(dirname "$VD")")_$(basename "$VD")"
    cp -r "$VD" "$dst" 2>/dev/null
    log "*** GENUINE *** persisted -> $dst"
  fi
done
rm -f "$PAUSE"
log "batch done: triaged=$n genuine=$gen; PAUSE removed (collection resumes)"
