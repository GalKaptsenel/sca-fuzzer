# PoC — `GCR_EL1.Exclude` breaks the MTE seal's `ADDG` retag (architectural tag fault)

## What it proves
On the Pixel 8's `KASAN_HW_TAGS` kernel, `GCR_EL1.Exclude` reserves tags 14 and 15
(`KASAN_TAG_INVALID`, `KASAN_TAG_KERNEL`). The Revizor executor did no `GCR_EL1` management, so JIT'd
test cases (and the seal's own retag instructions) ran with `Exclude=0xC000` inherited from the kernel.

The seal retags a pointer to a target allocation tag with `ADDG xN, xN, #0, #delta`, computing
`delta = (correct_tag - ptr_tag) % 16` — **plain modular** arithmetic — and the contract executor
models `ADDG`/`IRG` assuming `Exclude = 0` (`src/aarch64/contract_executor/mte_tag_plugin.c`). But
`ADDG`'s tag offset uses `ChooseNonExcludedTag`, which **skips excluded tags**. So whenever the tag
walk crosses 14 or 15, the genuine sealed retag lands on a *different* tag on hardware than the model
computed → a tag-mismatched **architectural** access → an EL1 MTE tag-check fault. On this kernel that
surfaces as `BUG: KASAN: invalid-access` (`ptrX/mem6`), aborts the measurement, and returns a malformed
batch → the intermittent "not a batch response (bad magic)".

## Result (real device)
```
kernel-inherited GCR_EL1 = 0x000000000001c000  (Exclude bits[15:0] = 0xc000)   # tags 14,15 reserved
DIVERGE: ADDG(tag= 0, #14)  kernel-GCR -> tag  0   Exclude=0 -> tag 14
DIVERGE: ADDG(tag= 1, #15)  kernel-GCR -> tag  2   Exclude=0 -> tag  0
...
SWEEP: 135 of 240 (base tag, delta) pairs give a DIFFERENT ADDG tag under Exclude=0xc000 than Exclude=0
```
135/240 (56%) of retag combinations diverge → the fault is common but content-dependent (only inputs
whose sealed access happens to hit a divergent (tag, delta) fault), which is exactly the intermittency
we saw.

## The fix
`src/aarch64/executor/mte.c` `mte_gcr_clear_exclude()` zeroes `GCR_EL1.Exclude` for the test run (inside
the IRQ-off window, per input), and `mte_gcr_restore()` puts the kernel's value back at the input and
test-case boundaries. `ADDG`/`IRG` then use plain modular tag arithmetic, matching the tag-blind
contract model, and KASAN's exclusion is intact for kernel code between runs.

## Run it
```
docs/mte_gcr_poc/run_poc.sh
```
Builds `poc_gcr_addg.ko` on host B against the Pixel GKI tree, pushes it over the VM's adb, `insmod`s it
(the module prints the sweep to `dmesg` and returns `-EINVAL` so it never stays resident), and prints
the demonstration. Requires the same environment as the executor build (host B with `~/pixel-kernel/out`
+ `~/toolchains/llvm-bin`) and the device on the VM's adb (`127.0.0.1:5037`).

## Files
- `poc_gcr_addg.c` — the PoC module (sweeps every base-tag × delta, reports divergences).
- `Makefile` — kbuild stub (`obj-m := poc_gcr_addg.o`).
- `run_poc.sh` — build + push + run + print.
