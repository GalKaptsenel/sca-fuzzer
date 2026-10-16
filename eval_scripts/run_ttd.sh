#!/bin/bash
# Adaptive TTD campaign runner (Machine A / N3-local).
# Usage: run_ttd.sh <row:config:inputs> [<row:config:inputs> ...]
# Sequential (device-exclusive). R=20 reps/row; if run1 hits the cap undetected,
# reduce to 3 reps (cross candidate) so we never do 20x a very-long run.
# Writes eval_out/ttd/<row>/summary.csv and per-run logs; logs every step to campaign.log.
set -u
ROOT=/home/gal_k_1_1998/revizor/sca-fuzzer
cd "$ROOT" || exit 2
source venv/bin/activate
R=20; TMAX=3600
mkdir -p "$ROOT/eval_out/ttd"
echo $$ > "$ROOT/eval_out/ttd/campaign.pid"
LOG="$ROOT/eval_out/campaign.log"
log(){ echo "[$(date '+%F %T')] [ttd pid=$$] $*" >> "$LOG"; }
log "=== TTD campaign START rows: $* ==="
for spec in "$@"; do
  ROW="${spec%%:*}"; rest="${spec#*:}"; CFG="${rest%%:*}"; INP="${rest##*:}"
  OUT="$ROOT/eval_out/ttd/$ROW"; mkdir -p "$OUT"
  SUM="$OUT/summary.csv"
  echo "seed,detected,wall_s,internal_dur,test_cases,violations,prog_seed,input_seed,rc" > "$SUM"
  log "ROW $ROW cfg=$CFG inputs=$INP R=$R Tmax=$TMAX"
  R_eff=$R; s=0
  while [ $s -lt $R_eff ]; do
    s=$((s+1))
    rd="$OUT/run$s"; mkdir -p "$rd"
    start=$(date +%s.%N)
    timeout $((TMAX+180)) ./revizor.py fuzz -s base.json -c "$CFG" -n 100000000 \
        -i "$INP" -w "$rd" --timeout "$TMAX" --save-violations true > "$rd/run.log" 2>&1
    rc=$?
    end=$(date +%s.%N)
    wall=$(awk "BEGIN{printf \"%.1f\", $end-$start}")
    viol=$(grep -m1 '^Violations:' "$rd/run.log" | awk '{print $2}'); viol=${viol:-NA}
    tc=$(grep -m1 '^Test Cases:' "$rd/run.log" | awk '{print $3}'); tc=${tc:-NA}
    idur=$(grep -m1 '^Duration:' "$rd/run.log" | awk '{print $2}'); idur=${idur:-NA}
    seedline=$(grep -m1 'seeds' "$rd/run.log")
    pseed=$(printf '%s' "$seedline" | sed -nE 's/.*program:([0-9]+).*/\1/p'); pseed=${pseed:-NA}
    iseed=$(printf '%s' "$seedline" | sed -nE 's/.*input:([0-9]+).*/\1/p'); iseed=${iseed:-NA}
    det=0; if [ "$viol" != "NA" ] && [ "$viol" -ge 1 ] 2>/dev/null; then det=1; fi
    echo "$s,$det,$wall,$idur,$tc,$viol,$pseed,$iseed,$rc" >> "$SUM"
    log "  $ROW run$s rc=$rc det=$det wall=${wall}s tc=$tc viol=$viol"
    if [ $s -eq 1 ] && [ $det -eq 0 ]; then R_eff=3; log "  $ROW run1 undetected at cap -> cross candidate, reduce to 3 reps"; fi
  done
  log "ROW $ROW DONE ($s runs)"
done
log "=== TTD campaign DONE ==="
