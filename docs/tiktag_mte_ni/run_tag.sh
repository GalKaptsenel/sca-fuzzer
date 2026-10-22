#!/system/bin/sh
# One on-device run of the loaded test case over a REIF input: N repeats of
# (TRACE; CHECKOUT; MEASUREMENT), printing each htrace. Keeps all executor_userland
# traffic on the device so the host issues one adb call per input, not one per repeat.
# Args: $1 = REIF path, $2 = repeat count.
EU="/data/local/tmp/executor_userland /dev/executor"
REIF="$1"
N="$2"
$EU 9 >/dev/null 2>&1
IID=$($EU 5 2>/dev/null | grep 'Allocated Input ID' | grep -oE '[0-9]+')
$EU 4 "$IID" >/dev/null 2>&1
$EU w "$REIF" >/dev/null 2>&1
i=0
while [ "$i" -lt "$N" ]; do
  $EU 8 >/dev/null 2>&1
  $EU 4 "$IID" >/dev/null 2>&1
  $EU 7 2>/dev/null | grep 'htrace 0:'
  i=$((i + 1))
done
