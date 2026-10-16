#!/bin/bash
# cr-seq restrictiveness test: run canon cross-input configs under contract [seq] (less restrictive than
# [cond,bpas]) a couple of times each, fully-random, and record TTD/leaking-pair. No rm (fresh -w dirs via
# mkdir -p; run.log is overwritten by redirection).
set -u
ROOT=/home/gal_k_1_1998/revizor/sca-fuzzer; cd "$ROOT" || exit 2
source venv/bin/activate
SUM="$ROOT/eval_out/btb/crseq_summary.txt"; : > "$SUM"
for cfg in canon_crseq_btb canon_crseq; do
  for i in 1 2; do
    d="$ROOT/eval_out/btb/crseq_${cfg}_$i"; mkdir -p "$d"
    start=$(date +%s)
    timeout 150 ./revizor.py fuzz -s base.json -c "eval_configs/$cfg.yml" -n 100000000 -i 8 \
        -w "$d" --timeout 130 --save-violations true > "$d/run.log" 2>&1
    pair=$(grep -m1 'CONFIRMED cross-input leak' "$d/run.log" | sed -nE 's/.*leaking pair ([0-9]+) -> detecting pair ([0-9]+).*/\1->\2/p')
    tc=$(grep -m1 '^Test Cases:' "$d/run.log" | awk '{print $3}')
    nv=$(ls -d "$d"/violation-* 2>/dev/null | wc -l)
    echo "[$(date +%T)] crseq $cfg rep$i: wall=$(( $(date +%s) - start ))s tc=${tc:-NA} pair=${pair:-none} viol=$nv" >> "$SUM"
  done
done
echo "ALL DONE $(date +%T)" >> "$SUM"
