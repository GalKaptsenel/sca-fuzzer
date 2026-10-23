#!/bin/bash
cd /home/gal_k_1_1998/revizor/sca-fuzzer
source /home/gal_k_1_1998/revizor/revizor-venv/bin/activate 2>/dev/null
export REVIZOR_LOG=0
GD=docs/tiktag_mte_ni/general_detection
for r in 1 2 3 4; do
  WD=$GD/eval_out/hard_r$r; rm -rf $WD; mkdir -p $WD
  V=$(python revizor.py tfuzz -s base.json -t $GD/s1_tiktag_hard.asm -c $GD/s1_mte_hw_hard.yml -n 1 -i 8 -w $WD 2>&1 | grep -iE "^Violations:" | awk '{print $2}')
  echo "run $r: Violations=$V"
done
