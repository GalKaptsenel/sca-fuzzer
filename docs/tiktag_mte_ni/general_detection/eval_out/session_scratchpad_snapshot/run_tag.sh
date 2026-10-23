#!/system/bin/sh
# One on-device run: load a REIF input, then N× (TRACE; CHECKOUT; MEASUREMENT) printing each htrace.
# Args: $1 = reif path, $2 = N reps. Keeps all executor_userland traffic on the device (no adb round-trips).
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
  i=$((i+1))
done
