#!/bin/bash
cd /home/gal_k_1_1998/revizor/sca-fuzzer
source /home/gal_k_1_1998/revizor/revizor-venv/bin/activate 2>/dev/null
export REVIZOR_LOG=0
GD=docs/tiktag_mte_ni/general_detection
WD=$GD/eval_out/run24h; rm -rf $WD; mkdir -p $WD
echo "START $(date -u +%FT%TZ)" > $WD/run24h.log
python revizor.py fuzz -s base.json -c $GD/s24_mte_random.yml -n 500000 -i 8 --nonstop -w $WD >> $WD/run24h.log 2>&1
echo "END $(date -u +%FT%TZ)" >> $WD/run24h.log
