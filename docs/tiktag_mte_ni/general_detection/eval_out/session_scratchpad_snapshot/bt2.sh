#!/system/bin/sh
# Fake the warm-up flag: R rounds of (K training runs with the TRAIN input, then one ATTACK run),
# BPU state preserved between runs (no phr flush in the regime). Print the attack run's htrace each round.
# Args: $1=train.reif $2=attack.reif $3=K $4=R
EU="/data/local/tmp/executor_userland /dev/executor"
TRAIN="$1"; ATTACK="$2"; K="$3"; R="$4"
AID=$($EU 5 2>/dev/null | grep 'Allocated Input ID' | grep -oE '[0-9]+')
r=0
while [ "$r" -lt "$R" ]; do
  $EU 4 "$AID" >/dev/null 2>&1; $EU w "$TRAIN" >/dev/null 2>&1   # load TRAIN (flag=train)
  k=0
  while [ "$k" -lt "$K" ]; do $EU 8 >/dev/null 2>&1; k=$((k + 1)); done   # K training runs
  $EU 4 "$AID" >/dev/null 2>&1; $EU w "$ATTACK" >/dev/null 2>&1  # flip to ATTACK (flag=attack)
  $EU 8 >/dev/null 2>&1                                          # attack run
  $EU 4 "$AID" >/dev/null 2>&1; $EU 7 2>/dev/null | grep 'htrace 0:'   # measure the attack run
  r=$((r + 1))
done
$EU 6 "$AID" >/dev/null 2>&1
