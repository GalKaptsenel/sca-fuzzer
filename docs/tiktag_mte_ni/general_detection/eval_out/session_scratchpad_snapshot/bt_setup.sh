#!/system/bin/sh
# Load K copies of the training input once (ids become the first K slots). Args: $1=train.reif $2=K
EU="/data/local/tmp/executor_userland /dev/executor"
$EU 9 >/dev/null 2>&1
i=0
while [ "$i" -lt "$2" ]; do
  IID=$($EU 5 2>/dev/null | grep 'Allocated Input ID' | grep -oE '[0-9]+')
  $EU 4 "$IID" >/dev/null 2>&1
  $EU w "$1" >/dev/null 2>&1
  i=$((i + 1))
done
echo "loaded $2 training inputs"
