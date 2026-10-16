#!/bin/bash
# A1 throughput: fixed-duration --nonstop run; parse test cases + duration -> tc/hour, exec/s, traces/s.
# Usage: run_throughput.sh <config> <label> [reps=2] [dur=3600] [inputs=50]
set -u
ROOT=/home/gal_k_1_1998/revizor/sca-fuzzer; cd "$ROOT" || exit 2
source venv/bin/activate
CFG=$1; LABEL=$2; REPS=${3:-2}; DUR=${4:-3600}; INP=${5:-50}
OUT="$ROOT/eval_out/throughput/$LABEL"; mkdir -p "$OUT"
LOG="$ROOT/eval_out/campaign.log"
log(){ echo "[$(date '+%F %T')] [tp $LABEL pid=$$] $*" >> "$LOG"; }
echo $$ > "$ROOT/eval_out/throughput/$LABEL.pid"
SUM="$OUT/summary.csv"; echo "rep,test_cases,inputs_per_tc,duration_s,tc_per_hour,exec_per_s,traces_per_s" > "$SUM"
log "START cfg=$CFG reps=$REPS dur=$DUR inp=$INP"
for r in $(seq 1 "$REPS"); do
  rd="$OUT/rep$r"; mkdir -p "$rd"
  timeout $((DUR+180)) ./revizor.py fuzz -s base.json -c "$CFG" -n 100000000 -i "$INP" \
      -w "$rd" --timeout "$DUR" --nonstop --save-violations true > "$rd/run.log" 2>&1
  tc=$(grep -m1 '^Test Cases:' "$rd/run.log" | awk '{print $3}'); tc=${tc:-0}
  ipt=$(grep -m1 'Inputs per test case:' "$rd/run.log" | awk '{print $5}'); ipt=${ipt:-0}
  dur=$(grep -m1 '^Duration:' "$rd/run.log" | awk '{print $2}'); dur=${dur:-0}
  tps=$(grep -m1 '^Traces/s:' "$rd/run.log" | awk '{print $2}'); tps=${tps:-NA}
  tph=$(awk "BEGIN{if($dur>0)printf \"%.0f\", $tc/($dur/3600); else print \"NA\"}")
  eps=$(awk "BEGIN{if($dur>0)printf \"%.1f\", ($tc*$ipt)/$dur; else print \"NA\"}")
  echo "$r,$tc,$ipt,$dur,$tph,$eps,$tps" >> "$SUM"
  log "rep$r tc=$tc ipt=$ipt dur=${dur}s tc/h=$tph exec/s=$eps traces/s=$tps"
done
log "DONE"
