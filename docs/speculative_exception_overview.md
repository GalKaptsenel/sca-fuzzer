# speculative exception GVT access, high-level overview (orientation for agents)

A one-page map of the AArch64 speculative exception (speculative exception) GVT access subsystem. For the full contract see
[`speculative_exception.md`](speculative_exception.md); for code start at `src/aarch64/gvt/builder.py` and
`src/aarch64/contract_executor/`.

## The idea

speculative exception fuzzing compares two *variants of the same test case* on the **same** input: a `genuine`
(all-correct) and a `decoy` (speculative slots perturbed). A hardware-trace difference is a
speculative leak. To stay safe on real hardware, a test case that uses PAC/MTE (or whose pointers we
want to make faulting) is **instrumented**: static instrumentation inserts register-only slots around each
memory access.

## One-run resolution

`make_gvt_access_builder(...) → builder.emit(tc) → GvtTestCase`. `instrumented.resolve(input)` (memoized per input
identity) runs **one** contract-executor (CE) trace, the *resolution trace*, and returns a
`ResolvedGvtTestCase` whose `.genuine()` / `.decoy(rng)` emit relocations and whose `.cer` is that
single trace, reused as the genuine contract trace (no separate genuine run).

The trace carries an **emulation table**: rows `(slot offset, byte size, op id)`. As the CE reaches a
site it runs the matching **op-emulator**, which materializes the site's genuine instruction *in place*
for the state actually present, on every reach:

- `EMUL_OP_PAC_AUTH`, sign the pointer the committed `AUT*` reads and plant the signature `MOVK`s, so
  the `AUT*` verifies and canonicalizes as ordinary code.
- `EMUL_OP_MTE_RETAG`, plant `ADDG Xn,Xn,#0,#delta` so the access's effective-address tag matches the
  accessed granule's allocation tag (`delta = cell_tag - ea_tag`).

A row owns `[offset, offset+size)`; the op's anchor instruction (the `AUT*`, the guarded access) sits at
`offset+size`; idle slots hold `UDF` (a slot that ever runs un-materialized faults loudly). Because the
real genuine instructions execute in this trace, its ctrace and taint equal the genuine binary's exactly
(a genuine PAC `XPAC` strip is the only safe-superset case).

## Two kinds of GVT access

- **Value GVT accesses** (`PacGvtAccess`, `MteGvtAccess`) carry a per-input value and need an op-emulator (the
  genuine instruction is computed from runtime state: a signature, a tag delta).
- **Flip GVT accesses** (`CanonGvtAccess`, `BranchTargetGvtAccess`, `SignExtGvtAccess`) carry **no** per-input value
  and **no** emulator: genuine is the inert `NOP` placeholder; a decoy EORs one pool mask in before the
  access and the same mask back after (EOR is self-inverse → the decoy re-converges with the baseline).

## Supported primitives

Selected by instruction categories and per-GVT access flags; dispatched in `GvtAccessBuilder.build`:

| primitive set | class | fault axis |
|---------------|-------|------------|
| `{pac}` | `PacGvtTestCase` | PAC auth (FPAC) |
| `{mte}` | `MteGvtTestCase` | MTE tag mismatch |
| `{pac, mte}` | `MtePacGvtTestCase` | PAC + MTE per access |
| `{pac, signext}` | `PacSignExtGvtTestCase` | PAC auth + a `[63:55]` sign-extension flip |
| `{canon}` | `CanonGvtTestCase` | non-canonical `[54:VA]` flip |
| `{branch_target}` | `BranchTargetGvtTestCase` | indirect-call target: non-canonical / misaligned |
| `{}` (PTE) | `SandboxGvtTestCase` | page-table environment only (no code GVT access) |

The CE stays **general**: it knows nothing about PAC/MTE/GVT access beyond dispatching an op id.

## Where things live

- `src/aarch64/gvt/builder.py`, GVT accesses, `GvtTestCase` subclasses, resolvers, `make_gvt_access_builder`.
- `src/aarch64/contract_executor/instruction_emulator.{c,h}`, the shared op dispatch + UDF reset.
- `.../pac_sign_plugin.c` (`pac_emulate_auth`), `.../mte_tag_plugin.c` (`mte_emulate_retag`).
- `src/aarch64/aarch64_executor.py`, primitive detection, PAC keys/profile, hardware run path.
- Configs: `config_pac.yml`, `config_mte.yml`, `config_pac_mte.yml`, `config_pac_signext.yml`,
  `config_canon.yml`, `config_canon_indirect.yml`.
- Tests: `tests/aarch64_tests/unit_trace_equivalence.py` (the one-run == genuine property per primitive)
  and the per-primitive `unit_*_access.py` / `unit_mte_ni_tag_safety.py`.
