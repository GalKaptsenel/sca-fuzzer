#!/bin/bash
# One-shot recovery: when the Pixel 8 is reachable again after the device reboot, reinstall the canonical
# executor module (md5 5c8dfec), restore the sysfs/dev permissions, resume the paused collect daemon, and
# remove the test module + this cron watcher. Idempotent and self-removing on success.
set -u
S=47061FDJH001K3
CANON=/data/local/tmp/revizor-executor.ko
CANON_MD5=5c8dfec2646f075c0328a6de2fa77602
DAEMON_PID=77242
LOG=/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/collect/restore_canonical.log
SELF=/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/restore_canonical.sh
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

adb devices 2>/dev/null | grep -q "^$S[[:space:]]*device" || { log "device not visible, retry later"; exit 0; }
bc=$(timeout 8 adb -s $S shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')
[ "$bc" = "1" ] || { log "device visible, boot_completed=$bc, retry later"; exit 0; }

# verify the canonical .ko on device is intact before loading it
dm=$(timeout 10 adb -s $S shell "md5sum $CANON 2>/dev/null" | awk '{print $1}')
[ "$dm" = "$CANON_MD5" ] || { log "canonical .ko md5 on device = '$dm' (expected $CANON_MD5); NOT loading"; exit 0; }

# insmod if absent (harmless EEXIST if already loaded), restore node perms
timeout 30 adb -s $S shell "su 0 -c 'insmod $CANON 2>/dev/null; chmod 777 /dev/executor 2>/dev/null; chmod -R a+rw /sys/executor 2>/dev/null; rm -f /data/local/tmp/revizor-executor-pftest.ko 2>/dev/null'"
loaded=$(timeout 10 adb -s $S shell "lsmod | grep -c revizor_executor" | tr -d '\r')
abi=$(timeout 10 adb -s $S shell "cat /sys/executor/system/abi_version 2>/dev/null" | tr -d '\r')
node=$(timeout 10 adb -s $S shell "ls /dev/executor 2>/dev/null" | tr -d '\r')
if [ "$loaded" = "1" ] && [ "$abi" = "2" ] && [ -n "$node" ]; then
  kill -CONT "$DAEMON_PID" 2>/dev/null && log "resumed collect daemon $DAEMON_PID (campaign stays paused via PAUSE flag)"
  log "CANONICAL RESTORED: loaded=$loaded abi=$abi node=$node. removing cron watcher."
  ( crontab -l 2>/dev/null | grep -v 'restore_canonical.sh' ) | crontab - 2>/dev/null
  exit 0
else
  log "restore attempt incomplete: loaded=$loaded abi=$abi node='$node'; will retry"
  exit 0
fi
