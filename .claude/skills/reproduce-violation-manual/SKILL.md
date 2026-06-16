---
name: reproduce-violation-manual
description: Manually reproduce ANY Revizor violation through /dev/executor with executor_userland, using the general "fixed context + swap the leaking slot" protocol — load the preceding inputs and the test case once, then run one experiment per counterexample input in the same slot and confirm the htrace divergence (the violation) persists. Use to confirm a saved violation-* by hand, generalize beyond a 2-input swap to V counterexample inputs, or build a manual measurement loop for a known leak. Builds on the executor-userland skill; pair with revizor-violation-triage (predicts the leaking set first).
---

# Manually reproducing a Revizor violation (general protocol)

A Revizor violation is a set of **V counterexample inputs** that are contract-equivalent yet produce
divergent hardware traces. To reproduce it by hand you recreate the *exact same* micro-architectural
context for every one of the V inputs, vary only the one slot under test, and confirm the htrace
divergence is stable and input-determined.

Read the `executor-userland` skill first — this skill is the methodology; that one is the tool.

## Methodology
Let the violation's counterexample inputs sit at boosted indices, and let `i = min(index)` be the
lowest. The inputs `0 .. i-1` are the **real preceding context** (priming the µarch state); slot `i`
is the only thing you swap and the only thing you measure.

1. **Fixed context** — load inputs `0 .. i-1` into slots `0 .. i-1` (allocate → checkout → write),
   then load the test case `T`. Do this **once**; it never changes.
2. **V experiments** — for each of the V counterexample inputs in turn: write that input into slot `i`,
   `TRACE`, then `MEASUREMENT` slot `i`. Record its htrace.
3. **Confirm persistence** — compare the V htraces at the *predicted leaking set* (get it from
   `revizor-violation-triage` / `ce_always_mispredict.py` first). A genuine violation: the leaking set
   is lit for some inputs and ~never for others, the partition matches the contract's speculative
   divergence, and it reproduces across repeated trials (the leak is **intermittent** — collect
   statistics, don't expect every trial to fire).

## Inputs you need
- `VD` = violation dir (`generated.asm`, `sandboxed_test_case`, `input_NNNN.bin`, `report.txt`).
- The V counterexample input numbers (from `report.txt` → `## Counterexample Inputs`).
- `i = min(those numbers)`; you load `0..i-1` as context and test in slot `i`.
- The predicted leaking cache `SET` (from triage). Work in a scratch dir; **never modify the originals**.

## Build the byte artifacts (scratch dir)
```
ROOT=$(git rev-parse --show-toplevel); mkdir -p /tmp/exp
"$ROOT/src/aarch64/asm_to_bytes/asm_to_bytes" < "$VD/sandboxed_test_case" > /tmp/exp/tc.bin
python3 - "$VD" <<'PY'                      # rebuild each input's NZCV slot → PSTATE
import sys, struct; ROOT_VD = sys.argv[1]
sys.path.insert(0, "."); from src.aarch64.aarch64_input_layout import NZCVScheme
FO = 8192 + 6*8                              # flags slot = byte 8240
import os
for f in os.listdir(ROOT_VD):
    if not f.startswith("input_") or not f.endswith(".bin"): continue
    raw = bytearray(open(f"{ROOT_VD}/{f}", "rb").read())
    struct.pack_into("<Q", raw, FO, NZCVScheme.to_pstate(struct.unpack_from("<Q", raw, FO)[0]))
    open(f"/tmp/exp/{f}", "wb").write(raw)
PY
```

## Reload + set the regime (sysfs BEFORE loading)
```
cd "$ROOT/src/aarch64/executor"
sudo rmmod revizor_executor; sudo insmod revizor-executor.ko; sudo chmod 777 /dev/executor
echo "P+P" | sudo tee /sys/executor/measurement_mode        # match the regime the violation was found under
echo 0     | sudo tee /sys/executor/enable_pre_run_flush     # flush ON masks branch-mispredict leaks
```
Reload is required so `ALLOCATE` hands out `0,1,2,…` (the static iid counter persists otherwise).

## Run it (read every iid back — never assume)
```
EU="$ROOT/src/executor_userland/executor_userland /dev/executor"
ht(){ $EU 7 | grep -aoiP 'htrace 0:\s*\K[01]{64}'; }; C=$((63-SET))   # set N → char 63-N

$EU 9 >/dev/null                                                       # CLEAR_ALL_INPUTS
for k in $(seq 0 $((i-1))); do                                         # fixed context 0..i-1
  iid=$($EU 5 | grep -aoiP 'Allocated Input ID:\s*\K[0-9]+')
  $EU 4 "$iid" >/dev/null; $EU w /tmp/exp/input_$(printf %04d $k).bin >/dev/null
done
slot=$($EU 5 | grep -aoiP 'Allocated Input ID:\s*\K[0-9]+')            # the slot under test (= i)
$EU 1 >/dev/null; $EU w /tmp/exp/tc.bin >/dev/null                     # load test case

for V in $CE_INPUTS; do                                                # one experiment per counterexample
  $EU 4 "$slot" >/dev/null; $EU w /tmp/exp/input_$(printf %04d $V).bin >/dev/null
  $EU 8 >/dev/null; $EU 4 "$slot" >/dev/null
  echo "input $V  set$SET=${$(ht):$C:1}"                               # repeat in an outer loop for stats
done
```

## Interpreting
- Leaking set lit for one subset of the V inputs and ~never for the others, matching the contract's
  speculative-divergence prediction ⇒ **genuine** leak reproduced.
- All V inputs identical / arch-only every trial ⇒ no speculation occurred — almost always because
  `enable_pre_run_flush` was left on, or the guarding branch wasn't mistrained (force it via
  `branch_training_config` if needed; see `reproduce-revizor-violation`).

## Gotchas
- **Never assume input ids** — always parse them from `ALLOCATE` output.
- **No static absolute paths** — resolve from the repo root (relative is fine).
- `cmp -n 8256` when verifying read-backs (trailing simd/pad always differs).
- The measurement regime matters: the same leak can be strong under P+P and invisible under F+R —
  reproduce under the regime the violation was found.
- Don't run this while the Python campaign is live (kernel input-id allocator corruption).
