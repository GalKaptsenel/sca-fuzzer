---
name: executor-userland
description: Drive the Revizor kernel executor (/dev/executor) by hand with the executor_userland CLI — allocate/checkout/write inputs, load a test case, TRACE, and read MEASUREMENT htraces/PFCs. Use when you need to talk to /dev/executor directly (no Python fuzzer): load a TC + inputs, run a single measurement, dump/inspect a slot, set the measurement regime, or script a manual experiment. Reference for command numbers, the call ordering, on-the-wire byte layout, and the htrace bit mapping. Pairs with reproduce-violation-manual / reproduce-revizor-violation (which use this tool to replay a violation).
---

# Driving /dev/executor with `executor_userland`

`executor_userland` is a thin CLI over the kernel executor's ioctl/read/write API. It lets you load a
test case and inputs, run the measurement harness, and read back hardware traces **without** the
Python fuzzer — the right tool for manual experiments and violation reproduction.

## Locating the binary (no static paths)
Resolve it relative to the repo, never hard-code an absolute path:
```
ROOT=$(git rev-parse --show-toplevel)            # or your known repo root
EU="$ROOT/src/executor_userland/executor_userland /dev/executor"
```
Rebuild if missing/stale (`make -C "$ROOT/src/executor_userland"`). The device path is an argument —
pass whatever the module exposes (usually `/dev/executor`).

## Command surface
`$EU <command-number [arg] | w file | r file>` — argument is decimal or `0x` hex.

| # | Command | Arg | Notes / output |
|---|---------|-----|----------------|
| 1 | CHECKOUT_TEST | — | select the test-case region as the current target (for `w`) |
| 2 | UNLOAD_TEST | — | drop the loaded test case |
| 3 | GET_NUMBER_OF_INPUTS | — | prints `Number Of Inputs: N` |
| 4 | CHECKOUT_INPUT | iid | select input slot `iid` as current (for `w`/`r`/MEASUREMENT) |
| 5 | ALLOCATE_INPUT | — | prints `Allocated Input ID: N` — **read N back, never assume it** |
| 6 | FREE_INPUT | iid | free a slot |
| 7 | MEASUREMENT | — | prints `htrace 0: <64 bits>` (+ more htrace rows) and `pfc i: …` |
| 8 | TRACE | — | run the JIT harness over all loaded inputs for the current regime |
| 9 | CLEAR_ALL_INPUTS | — | drop every input slot |
| 10 | GET_TEST_LENGTH | — | prints `Test Length: N` |
| — | `w file` | path | write file bytes into the current checkout (test or input slot) |
| — | `r file` | path | read the current checkout back into `file` |

Parse the dynamic values out of stdout rather than assuming them:
```
iid=$($EU 5 | grep -aoiP 'Allocated Input ID:\s*\K[0-9]+')   # ALLOCATE → real iid
ht(){ $EU 7 | grep -aoiP 'htrace 0:\s*\K[01]{64}'; }          # MEASUREMENT → 64-bit string
```

## Call ordering (the part that bites)
1. **Set the regime in sysfs BEFORE loading** — `measurement_mode` (P+P / F+R), `enable_pre_run_flush`,
   `enable_ssbs`, `enable_branch_training`, etc. `TRACE` builds the JIT harness for the *current*
   `measurement_mode`, so changing it later means re-`TRACE`.
2. **Allocate → checkout → write** each input: `5` (get iid) → `4 iid` → `w in.bin`.
3. **Load the test case**: `1` (CHECKOUT_TEST) → `w tc.bin`.
4. **TRACE** (`8`) — executes *all* loaded inputs and records their htraces.
5. **MEASUREMENT** (`7`) reads back the **currently checked-out input's** last trace. So to read input
   `k`: `4 k` then `7`. MEASUREMENT needs a prior TRACE and a checked-out input (not the test region).

## On-the-wire formats
- **Input** = `main_region[4096] + faulty_region[4096] + registers[64]` = **8256 useful bytes**. Registers
  start at byte 8192: x0..x5 (8192–8239), NZCV slot (8240), SP slot (8248). A saved fuzzer `.bin` is
  **12288 bytes** (extra simd/pad the kernel ignores) — when verifying a read-back, `cmp -n 8256` only.
  The NZCV slot in a saved `.bin` uses the per-flag byte encoding; convert it to PSTATE with
  `NZCVScheme.to_pstate` before writing if you reconstruct inputs from `.bin`.
- **Test case** = raw machine code: assemble the sandboxed asm
  (`src/aarch64/asm_to_bytes/asm_to_bytes < sandboxed_test_case > tc.bin`).
- **htrace** = one u64 per row, printed MSB-first. **Raw bit `b` == cache set `b`** (identity). In the
  printed string, set `N` is at character index `63 - N`.

## Environment / pitfalls
- After `insmod`: `sudo chmod 777 /dev/executor`. Build the module with
  `KDIR=/usr/src/linux-headers-$(uname -r | sed 's/-cloud.*/-cloud-arm64/')` (no `/lib/modules/.../build`
  symlink on this box).
- **iids only restart at 0 after a fresh module reload** — the kernel's `static int64_t input_id`
  persists for the module's lifetime; `CLEAR_ALL_INPUTS` does *not* reset it. Reload (`rmmod`+`insmod`)
  when you need predictable `0,1,2,…` ids.
- **Never run `executor_userland` (or any CE probe) against /dev/executor while the Python campaign is
  live** — concurrent input-id allocation corrupts the kernel allocator (dmesg: "Checkedout an input id
  that does not exist"). Pause/kill the campaign first.
- If ioctls wedge in `D` state with a stuck refcount, a kernel Oops happened under the device mutex →
  reboot to clear (see memory `view[0] RW invariant`).
