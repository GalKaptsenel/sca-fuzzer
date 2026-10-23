#!/bin/bash
cd /home/gal_k_1_1998/revizor/sca-fuzzer
source /home/gal_k_1_1998/revizor/revizor-venv/bin/activate 2>/dev/null
export REVIZOR_LOG=0
GD=docs/tiktag_mte_ni/general_detection
for t in user_mte_skeleton user_mte_skeleton3; do
  WD=$GD/eval_out/try_$t; rm -rf $WD; mkdir -p $WD
  OUT=$(python revizor.py tfuzz -s base.json -t $GD/$t.asm -c $GD/s1_mte_hw_hard_noMT.yml -n 40 -i 10 --nonstop -w $WD 2>&1)
  V=$(echo "$OUT" | grep -iE "^Violations:" | awk '{print $2}')
  TC=$(echo "$OUT" | grep -iE "^Test Cases:" | awk '{print $3}')
  echo "$t : Test Cases=$TC Violations=$V ; viol dirs=$(find $WD -type d -name '*viol*' 2>/dev/null | wc -l)"
done
