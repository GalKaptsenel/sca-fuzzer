#!/bin/bash
cd /home/gal_k_1_1998/revizor/sca-fuzzer
source /home/gal_k_1_1998/revizor/revizor-venv/bin/activate 2>/dev/null
export REVIZOR_LOG=0
GD=docs/tiktag_mte_ni/general_detection
WD=$GD/eval_out/step4canon; rm -rf $WD; mkdir -p $WD
python revizor.py fuzz -s base.json -c $GD/s4_canon_random.yml -n 40 -i 8 --nonstop -w $WD 2>&1 | grep -iE "violations:|test cases:|effective cls|fast path|duration"
echo "viol dirs: $(find $WD -type d -name '*viol*' 2>/dev/null | wc -l)"
