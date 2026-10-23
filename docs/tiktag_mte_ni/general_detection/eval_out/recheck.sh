#!/bin/bash
cd /home/gal_k_1_1998/revizor/sca-fuzzer
source /home/gal_k_1_1998/revizor/revizor-venv/bin/activate 2>/dev/null
export REVIZOR_LOG=0
GD=docs/tiktag_mte_ni/general_detection
declare -A S=( [155905]=9 [160128]=39 )
for pass in 1 2; do
for v in 155905 160128; do
  VD=$GD/eval_out/try_user_mte_skeleton/violation-261022-$v
  echo "### recheck pass $pass violation $v divset=${S[$v]} ###"
  python $GD/triage_sweep.py $GD/s1_mte_hw_hard_noMT.yml "$VD" 5 ${S[$v]} --reps 1500 2>&1 | grep -E "SWEEP|by delta|^   [0-9]" | tail -2
done
done
