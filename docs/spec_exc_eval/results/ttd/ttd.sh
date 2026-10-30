#!/bin/bash
# Measure time-to-detection (TTD) per cell: run the fuzzer to the FIRST violation, K trials (fresh
# random seed each), record the report's "Time to detection". NONE if no violation within the timeout.
cd /home/gal_k_1_1998/revizor/sca-fuzzer
source ../revizor-venv/bin/activate
sudo chmod 777 /dev/executor 2>/dev/null; sudo chmod -R a+rw /sys/executor 2>/dev/null
R=docs/spec_exc_eval/results/ttd
OUT=$R/TTD.txt
K=3; TO=180
run() {  # label  mode  config  [template]
  local label=$1 mode=$2 config=$3 tmpl=$4
  for k in $(seq 1 $K); do
    local W=$R/run/$label-$k; rm -rf "$W"; mkdir -p "$W"
    if [ "$mode" = tfuzz ]; then
      timeout $TO python revizor.py tfuzz -s base.json -c "$config" -t "$tmpl" -n 99999 -i 50 -w "$W" >"$W/log" 2>&1
    else
      timeout $TO python revizor.py fuzz  -s base.json -c "$config" -n 99999 -i 50 -w "$W" >"$W/log" 2>&1
    fi
    local ttd tc
    ttd=$(grep -haoP "Time to detection:\s*\K[0-9.]+" "$W"/violation*/report.txt 2>/dev/null | head -1)
    tc=$(grep -haoP "Test Cases:\s*\K[0-9]+" "$W"/violation*/report.txt 2>/dev/null | head -1)
    echo "$label trial$k: TTD=${ttd:-NONE(>${TO}s)} tc_at_detect=${tc:-NA}" >> "$OUT"
  done
}
echo "# TTD results (K=$K trials, timeout=${TO}s)  $(date)" > "$OUT"
run canon_branch_template tfuzz config_canon_cip_ni.yml templates/canon_oracle_ssb.py
run canon_branch_random   fuzz  "$R/config_canon_branch_notmpl.yml"
run canon_load_template   tfuzz config_canon.yml        templates/spec_mem_leak.py
run canon_load_random     fuzz  config_canon.yml
run pac_signext_template  tfuzz config_pac_signext.yml  templates/spec_mem_leak.py
run pac_signext_random    fuzz  config_pac_signext.yml
echo "# DONE $(date)" >> "$OUT"
