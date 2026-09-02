#!/bin/bash
# Build the GCR_EL1.Exclude / ADDG PoC on host B (WSL, has the Pixel GKI tree + LLVM), push it to the
# Pixel over the VM's adb, run it, and print the demonstration from the kernel log. The module refuses
# to stay loaded (init returns -EINVAL), so no rmmod is needed.
set -e
KEY=/home/gal_k_1_1998/.ssh/revizor_remote
SSH="ssh -i $KEY -p 2222 gal@localhost"
POC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/gal_k_1_1998/revizor/revizor-venv/bin/python

echo ">> syncing PoC sources to host B"
$SSH 'mkdir -p ~/poc_gcr'
scp -i "$KEY" -P 2222 "$POC/poc_gcr_addg.c" "$POC/Makefile" gal@localhost:~/poc_gcr/ >/dev/null

echo ">> building poc_gcr_addg.ko on B against the Pixel GKI tree"
$SSH 'cd ~/poc_gcr && PATH=$HOME/toolchains/llvm-bin:$PATH make -C $HOME/pixel-kernel/out M=$PWD \
        ARCH=arm64 LLVM=1 CC=clang CROSS_COMPILE= KCFLAGS="-Wno-error -std=gnu11" modules 2>&1 | tail -4; \
      ls -l ~/poc_gcr/poc_gcr_addg.ko'

echo ">> pulling the .ko to the VM"
scp -i "$KEY" -P 2222 gal@localhost:~/poc_gcr/poc_gcr_addg.ko "$POC/poc_gcr_addg.ko" >/dev/null

echo ">> running it on the Pixel (insmod prints the PoC to dmesg, then init fails on purpose)"
"$PY" - "$POC/poc_gcr_addg.ko" <<'PY'
import sys, re
from ppadb.client import Client
d = Client(host="127.0.0.1", port=5037).devices()[0]
d.push(sys.argv[1], "/data/local/tmp/poc_gcr_addg.ko")
d.shell("su -c 'dmesg -c >/dev/null 2>&1'")
d.shell("su -c 'insmod /data/local/tmp/poc_gcr_addg.ko 2>/dev/null'")
dm = [re.sub(r'\x1b\[[0-9;]*m', '', l) for l in d.shell("su -c 'dmesg'").splitlines()]
for l in dm:
    if "PoC" in l or "ADDG" in l or "GCR_EL1" in l or "BUG REPRODUCED" in l or "modular" in l or "granule" in l:
        print("   ", l.split("] ", 1)[-1] if "] " in l else l)
PY
echo ">> done"
