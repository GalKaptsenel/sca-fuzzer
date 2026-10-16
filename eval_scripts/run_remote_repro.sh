#!/bin/bash
# Remote reproduction A4 (Cortex-A715=core4) + A5 (Cortex-X3=core8) for the L1D/PAC rows.
# (BTB rows excluded: cross-input priming is local-only.) Verdict = detected (check) within 300s cap.
set -u
ROOT=/home/gal_k_1_1998/revizor/sca-fuzzer; cd "$ROOT" || exit 2
source venv/bin/activate
LOG="$ROOT/eval_out/campaign.log"
ROWS="v1_l1d v2_l1d v4_l1d canon_cachetlb aut_signext"
OUT="$ROOT/eval_out/remote"; mkdir -p "$OUT"
SUM="$OUT/summary.csv"; echo "row,core,detected,wall_s,test_cases,violations" > "$SUM"
echo "[$(date '+%F %T')] [remote] START A4(A715=4)+A5(X3=8) rows: $ROWS" >> "$LOG"
emit_core() { case $1 in a715) echo 4;; x3) echo 8;; esac; }
for corename in a715 x3; do
  c=$(emit_core "$corename")
  for row in $ROWS; do
    cfg="$OUT/${row}_${corename}.yml"
    bash eval_scripts/make_remote_config.sh "eval_configs/$row.yml" "$c" > "$cfg"
    rd="$OUT/${row}_${corename}"; mkdir -p "$rd"
    start=$(date +%s)
    timeout 360 ./revizor.py fuzz -s base.json -c "$cfg" -n 100000000 -i 50 \
        -w "$rd" --timeout 300 --save-violations true > "$rd/run.log" 2>&1
    wall=$(( $(date +%s) - start ))
    viol=$(grep -m1 '^Violations:' "$rd/run.log" | awk '{print $2}'); viol=${viol:-NA}
    tc=$(grep -m1 '^Test Cases:' "$rd/run.log" | awk '{print $3}'); tc=${tc:-NA}
    det=0; if [ "$viol" != "NA" ] && [ "$viol" -ge 1 ] 2>/dev/null; then det=1; fi
    echo "$row,$corename,$det,$wall,$tc,$viol" >> "$SUM"
    echo "[$(date '+%F %T')] [remote] $row/$corename det=$det wall=${wall}s tc=$tc viol=$viol" >> "$LOG"
  done
done
echo "[$(date '+%F %T')] [remote] DONE" >> "$LOG"
