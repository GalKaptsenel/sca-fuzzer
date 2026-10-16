#!/bin/bash
# Emit a remote (Pixel) config from a local eval config, pinned to a given device core.
# Usage: make_remote_config.sh <local_config.yml> <core>  > <remote_config.yml>
# NOTE: cross-input priming is LOCAL-ONLY (aarch64_fuzzer asserts enable_cross_input_priming=False
# for remote). For the BTB rows on remote, strip enable_cross_input_priming / cross_input_* and rely
# on standard priming (handled by the caller), else the fuzzer refuses to run remote.
set -u
LOCAL=$1; CORE=$2
# copy local config minus pin and minus cross-input knobs (invalid on remote)
grep -vE '^(executor_pinned_core|enable_cross_input_priming|cross_input_priming_localizer|cross_input_leaks_only):' "$LOCAL"
cat <<EOF

# --- remote (Pixel) execution, pinned to device core $CORE ---
executor_remote: true
executor_remote_transport: adb
executor_remote_host: 127.0.0.1
executor_remote_port: 5037
executor_remote_device: /dev/executor
executor_remote_sysfs: /sys/executor
executor_remote_userland: /data/local/tmp/executor_userland
executor_remote_module: /data/local/tmp/revizor-executor.ko
executor_pinned_core: $CORE
EOF
