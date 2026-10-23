#!/bin/bash
cd /home/gal_k_1_1998/revizor/sca-fuzzer
source /home/gal_k_1_1998/revizor/revizor-venv/bin/activate 2>/dev/null
export REVIZOR_LOG=0
GD=docs/tiktag_mte_ni/general_detection
for t in general_mte_alu general_mte; do
  for r in 1 2 3; do
    WD=$GD/eval_out/rel_${t}_$r; rm -rf $WD; mkdir -p $WD
    V=$(python revizor.py tfuzz -s base.json -t $GD/$t.asm -c $GD/s1_mte_hw_hard_noMT.yml -n 3 -i 8 -w $WD 2>&1 | grep -iE "^Violations:" | awk '{print $2}')
    echo "$t run $r: Violations=$V"
  done
done
