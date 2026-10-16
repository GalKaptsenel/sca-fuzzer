#!/bin/bash
# Bounded cross-input BTB campaign: R reps, each stop-at-first-violation with a per-run time CAP.
# Records TTD (wall to first CONFIRMED cross-input leak), detected?, leaking pair, violation dir.
# Usage: run_btb.sh <row> <config> <inputs> <R> <cap_s>
set -u
ROOT=/home/gal_k_1_1998/revizor/sca-fuzzer; cd "$ROOT" || exit 2
source venv/bin/activate
ROW=$1; CFG=$2; INP=$3; R=$4; CAP=$5
OUT="$ROOT/eval_out/btbcamp/$ROW"; mkdir -p "$OUT"
echo $$ > "$OUT/pid"
SUM="$OUT/summary.csv"; echo "rep,detected,wall_s,test_cases,leaking_pair,violation_dir" > "$SUM"
LOG="$ROOT/eval_out/campaign.log"
log(){ echo "[$(date '+%F %T')] [btb $ROW pid=$$] $*" >> "$LOG"; }
log "START row=$ROW cfg=$CFG inputs=$INP R=$R cap=${CAP}s"
for s in $(seq 1 "$R"); do
  rd="$OUT/run$s"; rm -rf "$rd"; mkdir -p "$rd"
  start=$(date +%s.%N)
  timeout $((CAP+60)) ./revizor.py fuzz -s base.json -c "$CFG" -n 100000000 -i "$INP" \
      -w "$rd" --timeout "$CAP" --save-violations true > "$rd/run.log" 2>&1
  end=$(date +%s.%N); wall=$(awk "BEGIN{printf \"%.1f\", $end-$start}")
  pair=$(grep -m1 'CONFIRMED cross-input leak' "$rd/run.log" | sed -nE 's/.*leaking pair ([0-9]+) -> detecting pair ([0-9]+).*/\1->\2/p'); pair=${pair:-none}
  viol=$(ls -d "$rd"/violation-* 2>/dev/null | head -1); viol=${viol:-none}
  tc=$(grep -m1 '^Test Cases:' "$rd/run.log" | awk '{print $3}'); tc=${tc:-NA}
  det=0; [ "$pair" != "none" ] && det=1
  echo "$s,$det,$wall,$tc,$pair,$viol" >> "$SUM"
  log "  $ROW run$s det=$det wall=${wall}s tc=$tc pair=$pair"
done
log "DONE row=$ROW"
