#!/bin/bash
# Launch a detached, monitorable fuzzing campaign.
#   launch_campaign.sh <config.yml> [extra 'key: value' overrides ...]
# Creates ~/revizor/campaigns/<basename>_<ts>/ with config.yml (+ overrides appended), a run.sh that
# writes its OWN pid ($$, preserved across exec -> collision-free id), and launches it via setsid with
# the lock fd closed. Prints the campaign dir.
set -u
BASE_CFG="${1:?config.yml required}"; shift || true
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAMP_ROOT="$HOME/revizor/campaigns"
D="$CAMP_ROOT/$(basename "$BASE_CFG" .yml)_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$D"
cp "$BASE_CFG" "$D/config.yml"
for ov in "$@"; do echo "$ov" >> "$D/config.yml"; done

cat > "$D/run.sh" <<EOF
#!/bin/bash
# Launched via: setsid nohup run.sh ... 9>&-. Writes its own PID (\$\$, preserved across exec) as the
# campaign's authoritative id. Stop: kill -- -\$(cat $D/pid)
echo \$\$ > $D/pid
cd $R && source venv/bin/activate
exec python revizor.py fuzz -s base.json -c $D/config.yml -n 100000 -i 20 --nonstop --save-violations true -w $D
EOF
chmod +x "$D/run.sh"

: > "$D/pid"
setsid nohup "$D/run.sh" >> "$D/fuzz.log" 2>&1 < /dev/null 9>&- &
for _ in $(seq 1 20); do sleep 1; p=$(cat "$D/pid" 2>/dev/null); [[ -n "$p" && -d "/proc/$p" ]] && break; done
echo "campaign dir : $D"
echo "pid          : $(cat "$D/pid")"
echo "monitor with : $(dirname "${BASH_SOURCE[0]}")/monitor.sh \"$D\""
