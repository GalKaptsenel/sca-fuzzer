#!/system/bin/sh
# Append one attack input to the already-loaded training batch, trace the whole batch N times, print
# the attack input's htrace each time, then free the attack slot. Args: $1=attack.reif $2=N
EU="/data/local/tmp/executor_userland /dev/executor"
AID=$($EU 5 2>/dev/null | grep 'Allocated Input ID' | grep -oE '[0-9]+')
$EU 4 "$AID" >/dev/null 2>&1
$EU w "$1" >/dev/null 2>&1
i=0
while [ "$i" -lt "$2" ]; do
  $EU 8 >/dev/null 2>&1
  $EU 4 "$AID" >/dev/null 2>&1
  $EU 7 2>/dev/null | grep 'htrace 0:'
  i=$((i + 1))
done
$EU 6 "$AID" >/dev/null 2>&1
