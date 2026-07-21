#!/usr/bin/env bash
# Canonical build of revizor-executor.ko for the Pixel 8 (Tensor G3), run ON the WSL box (B, gal@lap1092).
#
# WHY THIS SCRIPT EXISTS: a plain `make` CANNOT build this module on WSL, and the working command was
# never recorded anywhere except the auto-generated kbuild .<obj>.o.cmd files. The module Makefile's
# default target forces aarch64-linux-gnu-gcc (not installed on B; B has clang/LLVM only), autodetects
# MTE/PAC from the x86 build host (gets "n" -> wrong -march + CONFIG_ARM64_{MTE,PAC}_HW=0), and does not
# bump the C standard (clang then compiles as gnu89 and errors on C99 for-loop declarations). The four
# overrides below are all load-bearing and were verified byte-for-byte against the known-good .cmd
# record (-std=gnu11, MTE_HW=1, PAC_HW=1, -march=armv8.5-a+..., -Wno-error). Do NOT use the Makefile's
# `make`/`all` target (it forces gcc) -- invoke kbuild via M= as below.
#
# Usage (on B):   ./build_pixel8.sh [BUILD_DIR]     (default BUILD_DIR = this script's directory)
set -euo pipefail

BUILD_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
KDIR="${KDIR:-$HOME/pixel-kernel/out}"          # GKI tree matching the phone (kernel.release 5.15.137-android14-11)
LLVM_BIN="${LLVM_BIN:-$HOME/toolchains/llvm-bin}"

[ -f "$KDIR/Module.symvers" ] || { echo "ERROR: KDIR=$KDIR has no Module.symvers (wrong kernel tree)"; exit 1; }
[ -x "$LLVM_BIN/clang" ]      || { echo "ERROR: no clang at $LLVM_BIN (need the LLVM toolchain)"; exit 1; }

echo ">> building in $BUILD_DIR against $KDIR"
cd "$BUILD_DIR"
PATH="$LLVM_BIN:$PATH" make -C "$KDIR" M="$BUILD_DIR" \
    ARCH=arm64 LLVM=1 CC=clang CROSS_COMPILE= \
    HW_SUPPORTS_MTE=y HW_SUPPORTS_PAC=y \
    KCFLAGS="-Wno-error -std=gnu11" \
    modules

echo ">> done. vermagic:"
modinfo "$BUILD_DIR/revizor-executor.ko" | grep vermagic
