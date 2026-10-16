#!/bin/bash
# Blind stack-canary: 20 launches (fresh random canary each), -w all (v1/v4/rsb), count full [55:0] recovery.
set -u
cd /home/gal_k_1_1998/arm-disclosure/pocs/stack_leak || exit 2
OUT=/home/gal_k_1_1998/revizor/sca-fuzzer/eval_out/exploit_canary; mkdir -p "$OUT"
: > "$OUT/all.log"
for i in $(seq 1 20); do
  echo "=== launch $i $(date +%T) ===" >> "$OUT/all.log"
  timeout 600 taskset -c 90 ./canary_leak_blind -w all -c 90 >> "$OUT/all.log" 2>&1
done
{ for ch in "[v1 ]" "[v4 ]" "[rsb]"; do
    tot=$(grep -cF "$ch" "$OUT/all.log")
    full=$(grep -F "$ch" "$OUT/all.log" | grep -c FULL)
    echo "$ch : $full/$tot FULL [55:0]"
  done ; } > "$OUT/summary.txt"
cat "$OUT/summary.txt" >> /home/gal_k_1_1998/revizor/sca-fuzzer/eval_out/campaign.log
echo "[$(date '+%F %T')] [exploit] CANARY DONE" >> /home/gal_k_1_1998/revizor/sca-fuzzer/eval_out/campaign.log
