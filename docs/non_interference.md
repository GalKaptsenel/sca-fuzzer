# AArch64 non-interference (NI) fuzzing — architecture

## 1. Two fuzzing types, orthogonal to generation

`CONF.fuzzer` selects the comparison strategy; it does **not** change how programs are generated.

- **regular** — fuzz *inputs* on one test case; a leak is a hardware-trace difference between inputs.
  When PAC/MTE categories are active, regular fuzzing runs through the *sealed* executor (§4) so the
  test case is arch-safe on hardware.
- **non-interference (NI)** — fuzz *test-case variants* on the **same** input: a `genuine` (all-correct)
  and a `decoy` (speculative slots perturbed) of the same TC, compared on hardware. A difference is a
  speculative leak.

Which primitive(s) are active comes purely from the enabled **instruction categories** (`PAC*`,
`MTE*`), not a separate mode knob. The NI executor requires at least one of PAC/MTE.

## 2. Sealings — composable, per-site register ops

A **sealing** (`aarch64_sealer.py: Sealing`) owns one site and the slot instructions there. `seal(value)`
returns the slot's instructions for a runtime value; `seal(None)` is the arch-safe placeholder. A
sealing is pure: it never reads a trace and never decides genuine vs decoy. Three concrete sealings,
each a register-only op:

| Sealing | placeholder `seal(None)` | `seal(value)` |
|---------|--------------------------|---------------|
| `SandboxSealing` | `AND mask ; ADD x29` | same (the clamp is value-independent) |
| `PacSealing` | `NOP ; NOP` (inert — the pointer stays the canonical clamped base) | `MOVK sig ; AUT*` (or the `XPAC` strip, with prob `pac_strip_prob`) |
| `MteSealing` | `NOP [; NOP]` (inert) | `ADDG #delta [; ADDG #back]` (retag to the cell tag before the access, then re-converge after — `back`=0 for genuine, non-zero for a decoy; the second op only when the access preserves its base) |

The placeholder emits no `AUT*`/`XPAC`/`PAC*` — it is the inert, arch-safe baseline. The hardware
`genuine()` render loads the precomputed signature with `MOVK`s and keeps the committed `AUT*`; `PAC*`
is a generation-only instruction and never appears in the instrumentation.

**Only `SandboxSealing` clamps memory** — PAC/MTE never sandbox. Within one site the clamp must come
**first**: its `AND` mask clears the top bits where the PAC signature lives, so a clamp after an auth
would wipe the signature. The per-site order is therefore `[Sandbox][PacSign][MteTag]` followed by the
index/displacement cancellation.

## 3. Sealer + per-input resolution

- **`make_sealer(generator, assemble, primitives, signer, trace_bytes_fn)`** picks the concrete
  `Sealer`. `trace_bytes_fn(code, input, canonicalize_auth)` runs the single per-input resolution trace.
  `seal(tc)` walks the TC (the `SandboxWalk` taint
  dataflow decides which memory bases still need an in-region clamp), inserts the arch-safe
  placeholders, and returns a `SealedTestCase`. Standalone generator-emitted `AUT*` instructions are
  each replaced by a `PacSealing` (a register-only auth site the memory walk doesn't reach).
- **`SealedTestCase.resolve(input)`** runs **one** contract-executor trace — the *resolution trace* —
  and computes each sealing's value — PAC: sign the reached pointer plus a pool of mask-verified wrong
  signatures that provably fail AUTH; MTE: classify the accessed cell's allocation tag and the
  speculation depth — returning a `ResolvedSealingTestCase`. The resolution trace is the placeholder
  with the committed `AUT*` spliced at each auth position and a tag-neutral `ADDG` at each MTE position,
  traced with `CONFIG_FLAG_CANONICALIZE_AUTH`: the CE strips every architectural `AUT*` like the
  speculative path instead of verify-or-abort, so an unsigned pointer never FPAC-faults and the resolver
  reads the canonical pointer/context off this trace and signs it. Because the auth sites carry the
  genuine `AUT*` word, this one trace is faithful to the genuine binary and doubles as the genuine
  contract trace — no separate genuine run (see `.cer`).
- **`ResolvedSealingTestCase.cer`** is that resolution trace, reused as the genuine contract trace
  (`compute_ctrace`/`compute_taint`). Its ctrace equals the genuine binary's exactly; its taint equals
  it for canonicality/branch-target/PTE and plain PAC, and is a safe superset for MTE and PAC with
  `pac_strip_prob>0`. (MTE: the genuine `ADDG #delta` gives the pointer the cell tag, which the taint
  model's `effective_address − sandbox_base` offset drops; PAC strip: a genuine `XPAC` reads no context
  while the resolution `AUT*` does. Either way the resolution trace marks *more* input bytes, never
  fewer.) The CE needs no knowledge of PAC/MTE.
- **`ResolvedSealingTestCase.genuine()`** seals every slot with its correct value (`MOVK` signature +
  `AUT*`, genuine `ADDG` retag). **`.decoy()`**
  perturbs a random non-empty subset of the *speculative* (never-architectural) slots with a failing
  value and seals the correct value everywhere else. The `Sandbox` clamp is never perturbed; MTE always
  re-converges to the cell tag after the access. `MTE_INITIAL_TAG = 6` is the uniform region tag, shared
  by kernel + CE + model.

## 4. Executors

- **`Aarch64NonInterferenceExecutor`** (`aarch64_executor.py`) — detect active primitives, seal the TC,
  pin PAC keys (so this process's `pac_sign` and the contract executor share key material), and per
  input mint a `BASELINE` (genuine) + `DECOY` variant; run both on hardware. The per-input htrace
  difference between baseline and decoy is the leak. MTE primitives are refused up front unless the CPU
  reports FEAT_MTE2 (`ID_AA64PFR1_EL1.MTE >= 2`): without tag checking the seal is meaningless and,
  composed with PAC, a baked signature over a tagged pointer FPAC-faults the kernel.
- **`Aarch64RegularSealedExecutor`** (extends the NI executor) — regular fuzzing with PAC/MTE. Inputs
  are grouped into *sealing classes* (identical resolved values across all slots); each class runs one
  shared sealed TC, minted once with a per-class coin (`_DECOY_PROB`) choosing the all-genuine form or
  the speculative-decoy form. Either way the architectural slots are always genuine, so the TC is
  arch-safe and the standard regular by-contract-trace leak comparison stays valid.

## 5. PAC key & auth mechanism

PAC slots are resolved against real key material. The executor pins deterministic PAC keys per test
case so this process's signing and the contract executor use identical keys, and the contract
executor obtains a pointer's signature / authentication through a kernel signing service (the keys
live at EL1). The generator forbids self-context `AUT* Xn,Xn` (ctx == ptr) so a seal's `MOVK` can
never contaminate the auth context before the `AUT*` reads it.

## 6. Running it + the debug table

Configs (`fuzzer: non-interference`, categories pick the primitive): `config_pac.yml` (PAC),
`config_mte.yml` (MTE), `config_pac_mte.yml` (both). For regular fuzzing with PAC/MTE, use a
`fuzzer: basic` config with the same categories.

```sh
VENV=/home/gal_k_1_1998/revizor/revizor-venv/bin/python   # tests/revizor need this interpreter
mkdir -p /tmp/ni_work
# REVIZOR_NI_TABLE=1 turns on the per-slot table without editing code.
REVIZOR_NI_TABLE=1 $VENV revizor.py fuzz -s base.json -c config_pac.yml -n 1 -i 2 -w /tmp/ni_work
```

The table lands at `logs/<timestamp>_<id>/ni/ni_table.log` (relative to the CWD, not `-w`): one row
per instruction offset; columns are the placeholder TC, the genuine baseline, and the decoy variants.
ARCH rows match across columns (genuine everywhere); SPEC rows show the decoys diverging (forged sig /
retag). `REVIZOR_VERBOSITY=N` sets the level (1 = logs, 2 = + verbose CE-trace comparison).

## 7. Environment limits / follow-ups

- A box with **no usable PMU** (< 4 counters) returns `ENODEV` from `REVISOR_TRACE`: the software
  stages and the table run, but the htrace comparison needs real PMU hardware.
- A slot's `Sig/Tag` is `None` only when it was never executed in the CE trace (`Flow = "-"`): nothing
  to resolve, so it stays the genuine placeholder (safe).
- Kernel `GCR_EL1.Exclude = 0` (clean `ADDG`/`IRG` mod-16 tag math) is still pending.
