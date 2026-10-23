#!/bin/bash
cd /home/gal_k_1_1998/revizor/sca-fuzzer
source /home/gal_k_1_1998/revizor/revizor-venv/bin/activate 2>/dev/null
export REVIZOR_LOG=0
GD=docs/tiktag_mte_ni/general_detection
declare -A S=( [160705]=32 [160128]=39 [155905]=9 )
for v in 160705 160128 155905; do
  VD=$GD/eval_out/try_user_mte_skeleton/violation-261022-$v
  echo "################ violation $v  divset=${S[$v]} ################"
  python $GD/triage_sweep.py $GD/s1_mte_hw_hard_noMT.yml "$VD" 5 ${S[$v]} --reps 1200 2>&1 | grep -ivE "WARNING|^\s*$"
  echo
done
