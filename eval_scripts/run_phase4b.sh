#!/bin/bash
# Phase 4b batch (device-exclusive, sequential): v2/BTB confirmation + contract-throughput.
set -u
ROOT=/home/gal_k_1_1998/revizor/sca-fuzzer; cd "$ROOT" || exit 2
LOG="$ROOT/eval_out/campaign.log"
echo "[$(date '+%F %T')] [phase4b] START (v2_btb_3t confirm + tp_cond + tp_condbpas)" >> "$LOG"
# 1) pure Spectre-v2 / BTB cross-input (>=3 targets, -i50, NO sealing): 3 bounded runs, 240s cap
bash eval_scripts/run_btb.sh v2_btb_3t eval_configs/v2_btb_3t.yml 50 3 240
# 2) contract-throughput (same generation/-i as seq; only contract differs): 2 reps x 400s
bash eval_scripts/run_throughput.sh eval_configs/tp_cond.yml cond 2 400 50
bash eval_scripts/run_throughput.sh eval_configs/tp_condbpas.yml condbpas 2 400 50
echo "[$(date '+%F %T')] [phase4b] DONE" >> "$LOG"
