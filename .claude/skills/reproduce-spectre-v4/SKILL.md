---
name: reproduce-spectre-v4
description: Reproduce Spectre-v4 / Speculative Store Bypass (SSB, CVE-2018-3639) on this AArch64 (Neoverse N3) machine end-to-end, by hand through /dev/executor, to demonstrate that Revizor's HW path can detect a v4 violation. Use when asked to build/run a Spectre-v4 (store-bypass) PoC, show SSB leaking on this box, verify v4 detectability, or check whether store bypass is enabled. A working, verified package lives at /home/gal_k_1_1998/spectre_v4_poc (build.sh, run_seq.sh, run_litsets.sh, ce_check.py). Distinct from reproduce-revizor-violation (which replays a saved violation-* dir) and from Spectre-v1 (branch mispredict).
---

# Reproducing Spectre-v4 (Speculative Store Bypass) on Neoverse N3

A **verified, self-contained PoC already exists at `/home/gal_k_1_1998/spectre_v4_poc/`** (see its
`README.md`). Prefer it. This skill explains the mechanism, the recipe, and the non-obvious
conditions so you can rebuild/adapt it.

## What v4 is (and how it differs from v1)
Store `STR v,[x]` then load `LDR y,[x]` to the **same address** must architecturally forward `v`.
If the store's **address** resolves slowly, the memory-disambiguation predictor may guess "no
alias" and let the load read the **stale** (pre-store) value, which then indexes a dependent load
→ a data-dependent cache footprint. **No branch is involved** (that's the v1 vs v4 distinction).
The leaked secret is the value at `[x]` *before* the store.

## The gadget (branchless)
```
MSR SSBS, #1            ; SSBS=1 => bypass ALLOWED (vulnerable). #0 = safe control.
UDIV x8,x1,x1 ; x8,x8 ×5 ; MUL x9,x1,x8   ; long latency => STORE address resolves very late
STR x2,[x29 + (x9&0x1fff)]    ; slow-address store of a FIXED value (e.g. 0x600)
LDR x4,[x29 + (x3&0x1fff)]    ; load offset x3; ==store offset on attack (alias) => may bypass
LDR x5,[x29 + (x4&0x1fff)]    ; dependent leak: cache set = loaded_value&0x1fff /64
```
Arch: load returns the stored value → fixed leak set. Bypass: load returns the input's stale
value at the aliased offset → input-dependent leak set.

## Inputs (built with Revizor's `InputGenerator`)
- **31 no-alias TRAINING inputs** (store offset `O`, load offset ≠ `O`) — common context.
- Two **ATTACK inputs** A/B (store & load both `O` = alias), byte-identical **except** the stale
  value at `O` (e.g. A→set 10, B→set 42). Under `seq` they are contract-equivalent (the load
  returns the *stored* value, identical for both) → ctrace identical → any HW divergence = violation.

## How the executor actually runs it (critical)
`TRACE` (ioctl 8) = `execute()`/`run_experiments()` runs the gadget across **all loaded inputs in
ONE tight RT-pinned, IRQs-off loop, in input-id order**, predictor state persisting across inputs;
it stores a per-input htrace. `MEASUREMENT` (ioctl 7) only **reads back** a stored htrace — it does
NOT execute. So training cannot be done by repeated MEASUREMENT; instead load the **31 training
inputs as ids 0..30 and the attack input as id 31** (runs last → predictor already trained → it
bypasses). Reload the module first so ids are a clean 0..31. htrace string is MSB-first; set N is
char index `63-N`.

## Run
```bash
cd /home/gal_k_1_1998/spectre_v4_poc
./build.sh                          # assemble gadgets + build inputs (needs the venv)
./ce_check.py                       # optional: confirm A/B contract-equivalent -> ctrace [24,32]
./run_seq.sh P+P 30                 # the experiment (vulnerable)
./run_seq.sh P+P 30 ./tc_ssbs0.bin  # the SAFE control (SSBS=0)
./run_litsets.sh                    # arch-flow verification: full lit-set footprint
```

## Expected (the v4 signature)
- Vulnerable (`tc_ssbs1.bin`): attack A lights its set (~30/30) and never B's; attack B vice-versa;
  the architectural set always lit for both.
- Control (`tc_ssbs0.bin`): neither speculative set fires (bypass disabled) → proves it's SSB.
- `run_litsets.sh`: SSBS=0 → footprint == CE arch ctrace exactly (channel fidelity); SSBS=1 →
  arch footprint + exactly one extra (speculative) set, differing by input.

## Conditions that matter (each one was necessary — learned empirically)
1. **P+P only** — F+R does not reveal it (same as v1 on this box).
2. **`SSBS=1` = bypass ALLOWED; `SSBS=0` = safe.** Easy to invert (the "Safe" name is the
   *unsafe-when-1* convention). Verify empirically: SSBS=0 must zero the leak.
3. **Slow store address required** (long UDIV chain). A short delay barely triggers.
4. **No-alias training required** — N3's disambiguator is conservative; without the 31 training
   inputs the attack does not bypass (cold default = no bypass).
5. `enable_pre_run_flush=0`, `warmups=0`.

## Confirming it is genuinely v4 (the argument)
(a) Same contract trace: CE `ARCH_ONLY` ctrace identical for A/B, and HW SSBS=0 matches it.
(b) Divergent HW trace under SSBS=1, differing by exactly the stale-value set.
(c) Branchless gadget (rules out v1) + SSBS=0 control removes the leak (gated on the SSB-safe bit)
→ the divergence is store-bypass, i.e. Spectre-v4.

## Environment gotchas
- `source /home/gal_k_1_1998/revizor/revizor-venv/bin/activate`; `chmod 777 /dev/executor` after insmod.
- Module build: `KDIR=/usr/src/linux-headers-6.12.90+deb13-cloud-arm64`.
- The AArch64 **CE has no store-bypass contract model** yet (only ARCH_ONLY / ALWAYS_MISPREDICT /
  BPU_NEOVERSE_N3); confirming v4 in the CE requires adding a `CONTRACT_STORE_BYPASS` (fork at
  stores: checkpoint → skip store → run window → rollback + replay). Until then, confirm v4 by the
  manual argument above + the SSBS=0 control.
