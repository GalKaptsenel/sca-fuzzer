---
name: reproduce-revizor-violation
description: Manually reproduce an AArch64 Revizor violation on real hardware using the executor_userland utility — load the test case + inputs by hand via ioctl/write, recreate the exact µarch state (fixed context + flush off), and read out which input ran from the leaking cache set. Use when asked to manually replicate/confirm a violation-* on HW, demonstrate the leak, or compare P+P vs F+R measurement of a known violation. Pairs with tools/emulate_violation.py and tools/ce_always_mispredict.py (which give the expected arch/spec cache sets first).
---

# Manually reproducing a Revizor violation on hardware

Goal: take a saved `violation-*` (two counterexample inputs with the same contract
trace but different hardware traces), load it by hand through `/dev/executor`, recreate
the *same* micro-architectural state for both inputs, measure them, and confirm the
leaking cache set tells you **which input ran**.

Run analysis FIRST (no HW needed) to know the expected leaking set:
- `tools/ce_always_mispredict.py <vdir> <idxA> <idxB>` → ctraces + arch/spec sets (authoritative).
- `tools/emulate_violation.py <vdir> <x29hex> <idxA> <idxB>` → independent hand-emulation.
The **distinguishing speculative set** (e.g. P+P #15/#65 → set 50 only for #65) is the bit to watch.

## Inputs you need
- `VD` = violation dir (has `generated.asm`, `sandboxed_test_case`, `input_NNNN.bin`, `report.txt`).
- The two counterexample input numbers from `report.txt` ("## Counterexample Inputs", `Input #A` / `Input #B`).
- `LOW` = min(A,B). You load inputs `0..LOW` (the real preceding context) and put the violating
  input in the last slot (iid `LOW`); that slot is the only thing you swap and the only thing you measure.
- The leaking cache `SET` from the analysis above.

## executor_userland command numbers (ioctl NR)
`EU="src/executor_userland/executor_userland /dev/executor"`
1=CHECKOUT_TEST  2=UNLOAD_TEST  3=GET_NUM_INPUTS  4=CHECKOUT_INPUT(arg)  5=ALLOCATE_INPUT
6=FREE_INPUT  7=MEASUREMENT  8=TRACE  9=CLEAR_ALL_INPUTS  10=GET_TEST_LENGTH
Also: `EU w file` (write file to current checkout), `EU r file` (read current checkout to file).

## Step 1 — build the byte artifacts (in a scratch dir; NEVER touch the originals)
```
mkdir -p /tmp/exp
# test case = sandboxed asm assembled to raw machine code (as -march=armv9-a+sve+memtag | objcopy -O binary)
src/aarch64/asm_to_bytes/asm_to_bytes < "$VD/sandboxed_test_case" > /tmp/exp/tc.bin   # ~4 bytes/instr
# inputs: copy each .bin and reconstruct ONLY the NZCV flags slot to PSTATE
python3 - <<'PY'
import sys,struct; sys.path.insert(0,".")
from src.aarch64.aarch64_input_layout import NZCVScheme
VD="<VD>"; FO=8192+6*8                 # gpr slot 6 = flags, at byte 8240
for i in list(range(LOW+1))+[OTHER]:   # 0..LOW plus the other violating input
    raw=bytearray(open(f"{VD}/input_{i:04d}.bin","rb").read())
    struct.pack_into("<Q",raw,FO,NZCVScheme.to_pstate(struct.unpack_from("<Q",raw,FO)[0]))
    open(f"/tmp/exp/in{i}.bin","wb").write(raw)
PY
```

## Step 2 — reload the module and set the regime
```
cd src/aarch64/executor
sudo rmmod revizor_executor; sudo insmod revizor-executor.ko; sudo chmod 777 /dev/executor
echo "P+P" | sudo tee /sys/executor/measurement_mode      # or F+R
echo 0     | sudo tee /sys/executor/enable_pre_run_flush   # CRITICAL — see pitfalls
```
Reload is REQUIRED so the kernel's static input-id counter restarts at 0 → allocations give iids 0,1,2,…

## Step 3 — load context inputs 0..LOW, then the test case
```
$EU 9 >/dev/null                                   # CLEAR_ALL_INPUTS
for i in $(seq 0 LOW); do
  iid=$($EU 5 | grep -aoiP 'Allocated Input ID:\s*\K[0-9]+')   # ALLOCATE
  $EU 4 "$iid" >/dev/null                           # CHECKOUT_INPUT iid
  $EU w /tmp/exp/in$i.bin >/dev/null                # write input #i into iid
done
$EU 1 >/dev/null ; $EU w /tmp/exp/tc.bin >/dev/null # CHECKOUT_TEST ; load test case
```
Verify (read back, compare only the meaningful first 8256 bytes):
```
for i in $(seq 0 LOW); do $EU 4 $i>/dev/null; $EU r /tmp/rb.bin>/dev/null 2>&1; \
  cmp -s -n 8256 /tmp/rb.bin /tmp/exp/in$i.bin || echo "slot $i BAD"; done
```

## Step 4 — the experiment (same context, swap only slot LOW, measure only slot LOW)
```
ht(){ $EU 7 | grep -aoiP 'htrace 0:\s*\K[01]{64}'; }   # MSB-first; set N is char index (63-N)
C=$((63-SET))
for t in $(seq 1 30); do
  $EU 4 LOW >/dev/null; $EU w /tmp/exp/in{A}.bin >/dev/null     # slot LOW := input A
  $EU 8 >/dev/null; $EU 4 LOW >/dev/null; hA=$(ht)             # TRACE; measure ONLY slot LOW
  $EU 4 LOW >/dev/null; $EU w /tmp/exp/in{B}.bin >/dev/null     # slot LOW := input B (only this slot)
  $EU 8 >/dev/null; $EU 4 LOW >/dev/null; hB=$(ht)
  echo "$t A:${hA:$C:1} B:${hB:$C:1}"                          # leaking bit per input
done
```
Count how often the leaking set is lit for A vs B. A clean discriminator lights for exactly one
input and ~never for the other (0 false positives), though it is INTERMITTENT (see pitfalls).

## Key formats / numbers
- Input on the wire = `main_region[4096] + faulty_region[4096] + registers_t[64]` = **8256 useful bytes**.
  Registers at byte 8192: x0,x1,x2,x3,x4,x5 (8192..8239), flags slot (8240), sp slot (8248).
- The saved `.bin` is **12288 bytes** (InputFragment: main+faulty+gpr+simd+pad). The kernel uses only the
  first 8256; the trailing simd/pad bytes are ignored (read-back zeroes them) — `cmp -n 8256` when verifying.
- htrace = one u64; MEASUREMENT prints `htrace 0: <64 chars, MSB first>`. `bit = 63 - char_index`,
  so cache `set N` is at char index `63-N`.
- `x29` (= &main_region, for offset→set math in the emulator) = `cat /sys/executor/print_sandbox_base`;
  sandbox offset = `addr - x29`, in [0,8191] (<4096 main, ≥4096 faulty); set = `(off//64)%64`.

## Pitfalls (most → least painful)
1. **`enable_pre_run_flush=1` (the default) suppresses the leak.** It calls `flush_bpu_phr()` before every
   run, resetting branch history → the conditional branch is predicted correctly (not-taken) → no
   speculative window → no leak (htraces identical, arch-only). **Set it to 0** so the predictor mistrains
   naturally and the branch mispredicts. (Or force it deterministically — see "Forcing the leak".)
2. **iids are only 0..N right after a fresh `insmod`** — the kernel's `static int64_t input_id` persists for
   the module's lifetime, so without a reload your allocations start at whatever it reached before.
3. **Reconstruct the NZCV flags slot to PSTATE** (offset 8240) with `NZCVScheme.to_pstate`; the saved `.bin`
   stores per-flag encoding. Usually harmless for bit-test branches but get it right. (sp slot is
   overwritten by the kernel — ignore it.)
4. **Never modify the original `.bin` files** — always patch copies in a scratch dir.
5. **`cmp` only the first 8256 bytes** when verifying read-backs (the rest is unused simd/pad → always differs).
6. **The leak is intermittent** (the branch mispredicts maybe ~half the runs), so collect statistics over many
   trials. The discriminating set is reliable *when it fires* (0 false positives on the other input); count
   its frequency, don't expect 30/30.
7. **The measurement method matters a lot.** The *same* leak can be strong under one method and nearly
   invisible under the other (observed: P+P set50 16/30 vs F+R 2/30 for one gadget). Reproduce with the
   method the violation was found under; if comparing, reload + redo cleanly per method.
8. **Set `measurement_mode` before `TRACE`** — TRACE builds the JIT harness for the current template.
9. **MEASUREMENT needs TRACED state + a checked-out input** (not the TEST region): TRACE first, then
   CHECKOUT_INPUT(iid), then MEASUREMENT.
10. After `insmod` always `chmod 777 /dev/executor`; build the module with
    `KDIR=/usr/src/linux-headers-$(uname -r | sed 's/-cloud.*/-cloud-arm64/')` (no `/lib/modules/.../build` symlink on this box).
11. If `/dev/executor` ioctls ever hang in `D` state with elevated refcount, a kernel Oops happened under the
    device mutex → reboot to clear (see memory: view[0] RW invariant).

## Forcing the leak deterministically (instead of relying on natural misprediction)
Mistrain the guarding branch to "taken" so it always mispredicts (arch is not-taken):
```
# byte offset of the branch = (#executable instrs before it in sandboxed_test_case) * 4
echo "<offset>:1" > /sys/executor/branch_training_config   # also sets enable_branch_training=1
```
This reapplies training after each PHR flush, so the speculative window opens every run.

## Interpreting the outcome
- Leaking set lit for input B and ~never for A ⇒ the cache encodes which input ran ⇒ **genuine** speculative
  leak reproduced. Cross-check the set against `ce_always_mispredict.py` (ctraces differ under ALWAYS_MISPREDICT,
  identical under seq) and the report's dominant htrace.
- Both inputs identical & arch-only every trial ⇒ no speculation occurred — almost always because flush was
  left on (pitfall 1) or the branch wasn't mistrained.
