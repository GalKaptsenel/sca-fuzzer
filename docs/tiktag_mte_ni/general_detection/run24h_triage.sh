#!/bin/bash
# 24h random MTE/NI fuzzing with per-violation triage. Each fuzz invocation runs WITHOUT --nonstop, so it
# stops at the first violation (campaign halts) -> triage that violation with auto_triage.py (tag-only +
# CE spec-only + all-tags device sweep) -> record GENUINE/QUESTIONABLE/NOISE -> loop. Device is used by
# fuzz and triage strictly sequentially (never concurrently). Logs every step; persists genuine ones.
set -u
REPO=/home/gal_k_1_1998/revizor/sca-fuzzer
PY=/home/gal_k_1_1998/revizor/revizor-venv/bin/python
CFG=$REPO/docs/tiktag_mte_ni/general_detection/s24_mte_random.yml
BASE=$REPO/docs/tiktag_mte_ni/general_detection/eval_out/run24h_triage
LOG=$BASE/campaign.log
LEDGER=$BASE/TRIAGE_LEDGER.tsv
GEN=$BASE/GENUINE
DUR=${1:-86400}            # seconds (default 24h)
SERIAL=47061FDJH001K3
mkdir -p "$BASE" "$GEN"
: > "$LEDGER"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

recover_device(){
  log "WEDGE? attempting module reload"
  adb -s "$SERIAL" shell su -c 'rmmod revizor_executor 2>/dev/null; insmod /data/local/tmp/revizor-executor.ko; chmod 777 /dev/executor; chmod -R a+rw /sys/executor' 2>&1 | tee -a "$LOG"
}

END=$(( $(date +%s) + DUR ))
log "START 24h-triage campaign; deadline=$(date -u -d @$END +%H:%M:%S)Z cfg=$CFG"
i=0
while [ "$(date +%s)" -lt "$END" ]; do
  i=$((i+1))
  WD=$BASE/iter_$(printf '%04d' $i)
  rm -rf "$WD"; mkdir -p "$WD"
  log "iter $i: FUZZ (stops at first violation) -> $WD"
  PYTHONPATH=$REPO timeout 3600 $PY "$REPO/revizor.py" fuzz -s "$REPO/base.json" -c "$CFG" \
      -i 8 -n 100000 -w "$WD" > "$WD/fuzz.log" 2>&1
  rc=$?
  VD=$(find "$WD" -type d -name 'violation-*' | head -1)
  if [ -z "$VD" ]; then
    log "iter $i: no violation (rc=$rc); $(grep -aoE '[0-9]+ test cases|Cls:[0-9.]+' "$WD/fuzz.log" | tail -1)"
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then recover_device; fi
    rm -rf "$WD"      # keep only iters that produced a violation
    continue
  fi
  log "iter $i: VIOLATION $VD -> triage"
  PYTHONPATH=$REPO $PY "$REPO/docs/tiktag_mte_ni/general_detection/auto_triage.py" \
      "$VD/org-config.yaml" "$VD" --reps 2000 --ledger "$LEDGER" > "$VD/auto_triage.stdout" 2>&1
  trc=$?
  V=$(grep -aoE 'VERDICT=[A-Z]+' "$VD/auto_triage.stdout" | tail -1)
  log "iter $i: $V  ($VD)"
  if [ "$trc" -eq 42 ]; then
    log "iter $i: *** GENUINE *** persisting to GENUINE/"
    cp -r "$VD" "$GEN/$(basename "$WD")_$(basename "$VD")"
  fi
done
log "END campaign after $i iterations"
log "LEDGER:"; cat "$LEDGER" | tee -a "$LOG"
