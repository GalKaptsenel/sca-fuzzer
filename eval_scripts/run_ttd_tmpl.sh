#!/bin/bash
# Template TTD runner (tfuzz; device-exclusive, sequential). R=20 fresh-seed reps; each rep runs tfuzz
# (one fixed template program, fresh random inputs each TC) and stops at the first violation or the cap.
# Usage: run_ttd_tmpl.sh <row> <config.yml> <template.py> <inputs> [cap_s] [R]
set -u
ROOT=/home/gal_k_1_1998/revizor/sca-fuzzer; cd "$ROOT" || exit 2
source venv/bin/activate
ROW="$1"; CFG="$2"; TMPL="$3"; INP="$4"; CAP="${5:-600}"; R="${6:-20}"
OUT="$ROOT/eval_out/ttd/$ROW"; mkdir -p "$OUT"
SUM="$OUT/summary.csv"; echo "rep,detected,wall_s,test_cases,violations,prog_seed,input_seed,kind,leak_pair,det_pair,rc" > "$SUM"
LOG="$ROOT/eval_out/campaign.log"
log(){ echo "[$(date '+%F %T')] [ttd-tmpl $ROW] $*" >> "$LOG"; }
echo $$ > "$OUT/campaign.pid"
log "=== TTD-TMPL START cfg=$CFG tmpl=$TMPL inputs=$INP cap=$CAP R=$R ==="
for s in $(seq 1 "$R"); do
  rd="$OUT/run$s"; mkdir -p "$rd"
  start=$(date +%s.%N)
  timeout $((CAP+120)) ./revizor.py tfuzz -s base.json -c "$CFG" -t "$TMPL" -n 100000000 \
      -i "$INP" -w "$rd" --timeout "$CAP" --save-violations true > "$rd/run.log" 2>&1
  rc=$?
  wall=$(awk "BEGIN{printf \"%.1f\", $(date +%s.%N)-$start}")
  viol=$(grep -m1 '^Violations:' "$rd/run.log" | awk '{print $2}'); viol=${viol:-NA}
  tc=$(grep -m1 '^Test Cases:' "$rd/run.log" | awk '{print $3}'); tc=${tc:-NA}
  seedline=$(grep -m1 'seeds' "$rd/run.log")
  pseed=$(printf '%s' "$seedline" | sed -nE 's/.*program:([0-9]+).*/\1/p'); pseed=${pseed:-NA}
  iseed=$(printf '%s' "$seedline" | sed -nE 's/.*input:([0-9]+).*/\1/p'); iseed=${iseed:-NA}
  cline=$(grep -m1 'CONFIRMED' "$rd/run.log")
  kind=$(printf '%s' "$cline" | sed -nE 's/.*CONFIRMED ([a-z-]+) leak.*/\1/p'); kind=${kind:-none}
  lp=$(printf '%s' "$cline" | sed -nE 's/.*leaking pair ([0-9]+).*/\1/p'); lp=${lp:-NA}
  dp=$(printf '%s' "$cline" | sed -nE 's/.*detecting pair ([0-9]+).*/\1/p'); dp=${dp:-NA}
  det=0; if [ "$viol" != "NA" ] && [ "$viol" -ge 1 ] 2>/dev/null; then det=1; fi
  echo "$s,$det,$wall,$tc,$viol,$pseed,$iseed,$kind,$lp,$dp,$rc" >> "$SUM"
  log "  run$s rc=$rc det=$det wall=${wall}s tc=$tc viol=$viol kind=$kind $lp->$dp"
done
log "=== TTD-TMPL DONE ($R reps) ==="
