#!/bin/bash
# Idempotent launcher for the collect daemon. Used by the host crontab (@reboot + every-5-min watchdog)
# so fully-random MTE/NI collection survives reboots and crashes with no Claude session attached.
# It ONLY (re)starts collection; triage needs a Claude session and resumes when one is alive.
REPO=/home/gal_k_1_1998/revizor/sca-fuzzer
GD=$REPO/docs/tiktag_mte_ni/general_detection
COLLECT=$GD/eval_out/collect
mkdir -p "$COLLECT"
DP=$(cat "$COLLECT/daemon.pid" 2>/dev/null)
if [ -n "$DP" ] && kill -0 "$DP" 2>/dev/null && tr '\0' ' ' </proc/"$DP"/cmdline 2>/dev/null | grep -q collect_daemon.sh; then
  exit 0   # daemon already alive — nothing to do
fi
cd "$REPO" || exit 1
source "$REPO/../revizor-venv/bin/activate" 2>/dev/null
setsid nohup bash "$GD/collect_daemon.sh" >/dev/null 2>&1 &
echo "[$(date -u +%FT%TZ)] campaign_boot: (re)launched collect daemon" >> "$COLLECT/boot.log"
