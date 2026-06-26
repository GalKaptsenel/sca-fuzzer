# AArch64 non-interference (NI) fuzzing — architecture

## 1. Two fuzzing types, orthogonal to generation

`CONF.fuzzer` selects the comparison strategy; it does **not** change how programs are generated.

- **regular** — fuzz *inputs* on one test case; a leak is a hardware-trace difference between inputs.
- **non-interference (NI)** — fuzz *test-case variants* on the **same** input: build a `baseline`
  (all-genuine) and a `decoy` (speculative slots perturbed) of the same TC and compare them on
  hardware. A difference is a speculative leak.

PAC and MTE instructions are generated the same way for both; which one(s) are active comes purely
from the enabled **instruction categories** (`PAC*`, `MTE*`), not a separate mode knob.

## 2. Seals — decoupled, composable register ops

A **seal** (`aarch64_seal.py: Seal`) owns the encoding of one protection on a *value* (the register
holding it), via three fills: `placeholder` (arch-safe, no behaviour change), `genuine` (the
committed value), `decoy` (a perturbed value). Three concrete seals, each a pure register op:

| Seal | slot | genuine | decoy |
|------|------|---------|-------|
| `Sandbox` | `AND mask ; ADD x29` | clamp into the sandbox | (never decoyed) |
| `PacSign` (`aarch64_pac.py`) | `MOVK sig ; AUT*` / `NOP ; XPAC*` | `MOVK correct_sig ; AUT*` (or strip) | `MOVK wrong_sig ; AUT*` |
| `MteTag` (`aarch64_mte.py`) | one instr | `NOP` / `ADDG` (fix tag) | `IRG` / `EOR tagmask` (retag) |

**Only `Sandbox` clamps memory.** `PacSign`/`MteTag` never sandbox. To protect a memory pointer you
**compose** them with `CompositeSeal([...])`, the list giving the apply order:
`[Sandbox, MteTag]`, `[Sandbox, PacSign]`, or `[Sandbox, PacSign, MteTag]`. Composing (rather than a
separate sandbox pass) is required because `Sandbox`'s `AND` mask clears the top bits where the PAC
signature lives — a clamp applied after an auth would wipe the signature.

A **fix point** (`FixPoint` + subclasses `PACFixPoint`, `MTEFixPoint`, `PacMteFixPoint`) is one
sealing site: the slot instructions + their positions + the per-seal data the executor fills in
(`correct_sig`/`alt_sigs` for PAC; `correct_tag`/`ptr_tag` for MTE).

## 3. Sealing passes

PAC has **two** independent fix-point kinds; MTE has one. Neither sandboxes.

- **`SealInstrumentation`** (in `aarch64_mte.py`) — the general memory pass. Over a taint walk it
  seals every memory access with `CompositeSeal([Sandbox] + value_seals)` where the base is not yet
  clamped, the value-seals alone where it is. `value_seals` is the ordered list of active value
  seals; `fixpoint_cls` is the FixPoint subclass holding their data.
- **`PacAuthInstrumentation`** (in `aarch64_pac.py`) — PAC's other kind: replaces each
  generator-emitted `AUT*` instruction with a `PacSign` slot. Register ops only, no memory.
- **`make_seal_pass(generator, primitives)`** (`aarch64_seal_factory.py`) — the factory: wires the
  active primitives' value-seals + their FixPoint class. `pac → [Sandbox,PacSign]/PACFixPoint`,
  `mte → [Sandbox,MteTag]/MTEFixPoint`, both `→ [Sandbox,PacSign,MteTag]/PacMteFixPoint`.

The engine (`SealedNIInstrumentation`) takes the sealed TC + fix points and mints variants:
`baseline()` fills every slot genuine; `decoys()` fills genuine except where the policy
(`seal.name != "sandbox" and is_speculative(fp)`) allows — i.e. only non-clamp seals on speculative
slots; the `Sandbox` clamp is always genuine. When **both** PAC and MTE are active the decoys are
**orthogonal per variant**: each decoy instance enables a random subset of `{pac_sign, mte_tag}`
(PAC-only / MTE-only / both / neither), applied consistently across all of that primitive's slots so
a hardware leak is attributable to one primitive. With a single primitive present it is always
decoyed. MTE `genuine` is the previous MTE behaviour: compare the pointer's tag to the cell's
(`MteTagState`: region tag + `STG*` stores) and `NOP` if they match, `ADDG` to fix it (arch only).

## 4. The unified executor

`Aarch64NonInterferenceExecutor` (`aarch64_executor.py`) is one self-contained executor:

1. **`__init__`** — detect active primitives from categories (`startswith("PAC"/"MTE")`); build the
   memory pass (`make_seal_pass`) + `PacAuthInstrumentation` if PAC; build the engine with the decoy
   policy above.
2. **`load_test_case`** — pin PAC keys (so this process's `pac_sign` and the contract executor share
   key material), seal the TC (memory pass + AUT* pass on one TC), assemble, and build the offset →
   fix-point maps (PAC by the `XPAC` placeholder offset; MTE by the access offset).
3. **`trace_test_case_with_taints`** — for each input: reset fix points, run **one** contract-executor
   trace of the sealed TC (ALWAYS_MISPREDICT, to reveal speculative paths), fill each fix point from
   that trace (PAC: `pac_sign` the reached pointer + a mask-verified pool of wrong signatures that
   provably fail AUTH; MTE: classify each slot's tag + speculation depth), then mint `baseline` +
   `decoy` variants. One CE trace suffices: every slot is a register op, so the sealed trace's
   memory/cache ctrace + taint are the real ones and the variants share its layout.
4. **`trace_test_case_variants_hw`** — run each input's variants on hardware with that input's
   arch-trace mistraining; the per-input htrace difference between baseline and decoy is the leak.

Safety: decoys forge signatures only on **speculative** slots (the contract executor `XPAC`-strips
speculative auths, so a forged AUTH never runs architecturally); the PAC machinery uses kernel SIGN
only, never AUTH on an unproven signature — a failed AUTH at EL1 hard-resets FEAT_FPAC hardware.

## 5. Running it + the debug table

Configs (`fuzzer: non-interference`, categories pick the primitive):
`config_pac.yml` (PAC), `config_mte.yml` (MTE), `config_pac_mte.yml` (both).

```sh
VENV=/home/gal_k_1_1998/revizor/revizor-venv/bin/python   # tests/revizor need this interpreter
mkdir -p /tmp/ni_work
# REVIZOR_NI_TABLE=1 also turns logging on, so the table is written without editing code.
REVIZOR_NI_TABLE=1 $VENV revizor.py fuzz -s base.json -c config_pac.yml     -n 1 -i 2 -w /tmp/ni_work
REVIZOR_NI_TABLE=1 $VENV revizor.py fuzz -s base.json -c config_mte.yml     -n 1 -i 2 -w /tmp/ni_work
REVIZOR_NI_TABLE=1 $VENV revizor.py fuzz -s base.json -c config_pac_mte.yml -n 1 -i 2 -w /tmp/ni_work
```

The table lands at `logs/<timestamp>_<id>/ni/ni_table.log` (relative to the CWD, not `-w`). Each row
is one instruction offset; columns are the `sealed` placeholder TC, the `baseline`, and two `decoy`
variants; `Flow` is the CE arch/spec annotation; `Sig/Tag` is the committed signature/tag from the
fix point. Rows that differ across columns are the seal slots (`<== slot`): on ARCH rows all columns
match (genuine everywhere); on SPEC rows the decoys diverge (forged sig / retag). Other channels:
`ni/flow.log` (inputs), `ni/signing.log` (PAC SIGN ops + per-slot data). `REVIZOR_VERBOSITY=N` sets
the level generally (1 = logs, 2 = + verbose CE-trace comparison).

## 6. Environment limits / follow-ups

- This VM has **no usable PMU**, so the final hardware step (`REVISOR_TRACE`) returns `ENODEV` for
  any executor; the table + all software stages run, but the htrace comparison needs real PMU HW.
- A slot's `Sig/Tag` shows `None` only when the slot was never executed in the CE trace (`Flow = "-"`):
  there's no signature/tag to resolve, so it stays the genuine placeholder (safe).
- Kernel `GCR_EL1.Exclude = 0` (clean ADDG/IRG mod-16 tag math) is still pending.
