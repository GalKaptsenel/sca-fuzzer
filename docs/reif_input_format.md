# REIF — the Revizor Extensible Input File

REIF is the on-the-wire and on-disk format of **one input** to the AArch64 kernel executor. It
replaces the old flat, fixed `InputFragment`/`input_t` dump: instead of a hardcoded byte layout, a
REIF file is a small header plus a **section table**, so new per-input initial state (MTE tags, PAC
keys, code relocations, branch training, …) can be added without another layout break.

The kernel header `src/aarch64/executor/userapi/executor_input_format.h` is the **source of truth**.
The Python writer/reader is `src/aarch64/aarch64_executor_input_encoder.py`; its constants mirror the
header and must be kept in sync with it.

## The same bytes serve two roles

The identical byte string is used both ways:

- **On the wire** — written to `/dev/executor`; the kernel parses and *strictly validates* it,
  locating each section **by type** (never by a hardcoded offset), and rejects anything malformed.
- **On disk** — saved with the **`.reif`** extension. A `.reif` file is complete and
  executor-ready: it can be `cat`/`write(2)` straight into `/dev/executor`, or loaded back in Python
  via `deserialize()` for a `reproduce` run. There is no separate "convert before executing" step —
  the flags are already in ARM PSTATE form and every section is self-located.

`ExecutorInput.save()` writes a `.reif` file; `deserialize(blob)` is its exact inverse and rebuilds
the `ExecutorInput` (architectural input + every section) bit-for-bit.

## Layout

All header and descriptor members are little-endian `u64` — no packing rules to get wrong across the
Python (`np.uint64` / `struct`) and C (`uint64_t`) writers. The preamble is 48 bytes and each
descriptor is 32 bytes, so everything is naturally 8-byte aligned; every payload begins 8-aligned.

```
offset 0   ┌──────────────────────────────────────────────┐
           │ magic       u64  = 0x49525A5652  ("RVZRI")    │  struct revisor_input_header
           │ version     u64  = 1                          │  (48 bytes)
           │ header_len  u64  = 48 + 32*n_sections         │  ← offset of the first payload
           │ n_sections  u64                               │
           │ flags       u64  (reserved)                   │
           │ total_len   u64  (== bytes written to device) │
           ├──────────────────────────────────────────────┤  section table:
           │ [i] type    u64                               │  n_sections × struct revisor_input_section
           │     flags   u64  (reserved per-section)       │  (32 bytes each)
           │     offset  u64  from start of the file       │
           │     length  u64  payload bytes                │
           │ …                                             │
           ├──────────────────────────────────────────────┤  ← header_len
           │ section payloads, each 8-byte aligned         │
           └──────────────────────────────────────────────┘  ← total_len
```

**Reader rule:** iterate the table, dispatch known types, and **skip unknown** types (advance by
`length`). This is what makes the format forward-compatible within a major `version`.

## Section types

| Id | Type | Payload |
|---|---|---|
| `0x01` | `MEMORY_MAIN` | sandbox `main_region` bytes |
| `0x02` | `MEMORY_FAULTY` | sandbox `faulty_region` bytes |
| `0x03` | `GPR` | `registers_t` = x0..x5, the flags register (NZCV), sp. The flags slot holds the **ARM PSTATE** value (NZCV in bits 31:28), loaded verbatim — the writer converts from the per-flag `NZCVScheme` before packing. |
| `0x04` | `SIMD` | v0..v7 (256 B; reserved, not yet loaded) |
| `0x05` | `PAC_KEYS` | `struct pac_keys` (5 keys × {lo,hi} = 10×u64) — the keys this input's signatures were signed under (see below) |
| `0x06` | `MTE_TAGS` | one 4-bit allocation tag per 16-byte granule of the main‖faulty span, packed two per byte, low nibble first (granule 2·i in bits 3:0, 2·i+1 in bits 7:4) |
| `0x07` | `CODE_RELOC` | the per-input **relocation table** (see below) |
| `0x08` | `BPU_TRAINING` | per-input branch-training entries (see below) |

`MEMORY_MAIN`, `MEMORY_FAULTY`, and `GPR` are required. The rest are optional; when a section is
absent the kernel uses its default (e.g. no `MTE_TAGS` ⇒ the region's default tag). Ids `0x10+` are
reserved for future initial state (system registers, page attributes, …).

## The code-relocation table (`CODE_RELOC`)

The relocation table lets **one loaded test case serve many differently-sealed inputs** without
re-assembling or re-loading the body per input. The test-case code is loaded **once**; each input
carries its own list of word patches:

```c
struct revisor_code_reloc_entry { uint32_t offset; uint32_t value; };  // terminated by all-ones
```

Each entry splices the 32-bit little-endian `value` at byte `offset` of the test-case body. The
kernel, on the pinned CPU, immediately before an input executes:

1. writes each `value` into the shared body (through the writable view),
2. performs the local i-cache maintenance for the patched words (`dc cvau` / `dsb` / `ic ivau` /
   `dsb; isb`),
3. runs the input,
4. **reverts** the body to the pristine test case afterward,

so the next input starts from a clean skeleton. This is how the non-interference (PAC/MTE) seal ships
its per-input signature `MOVK`/`AUT*`/`XPAC` (and MTE `ADDG` retag) words: the Python resolver emits
them as a `RelocationPlan` (`aarch64_relocations.py`), the encoder packs them into `CODE_RELOC`, and
the kernel applies them. No per-variant assembler spawn, no Python-side splice, and no runtime
splice-validation on the hot path (correctness is covered by tests). Bounds are validated at trace
time against the loaded test-case length. Up to `REVISOR_INPUT_MAX_CODE_RELOCS` (256) entries.

## Branch training (`BPU_TRAINING`)

```c
struct revisor_bpu_train_entry { uint32_t offset; uint32_t taken; };  // terminated by all-ones
```

Each entry trains the conditional branch at byte `offset` of the body toward `taken` (1 = TAKEN,
0 = NOT-TAKEN) before this input executes, so per-input branch training travels **with the input**
rather than as global config. Gated by `enable_branch_mistraining` (off by default); when off, the
section is omitted entirely. Up to `REVISOR_INPUT_MAX_BPU_TRAIN` (64) entries.

## PAC keys (`PAC_KEYS`)

On a PAC run every input carries the key set its baked signatures were signed under. The keys are
generated deterministically **by the input generator** (one campaign-wide set per run seed), so a
run — and every saved `.reif` — is reproducible and self-contained. The kernel uses each input's
`PAC_KEYS` when running that input, and the sign/auth ioctls (`REVISOR_PAC_SIGN` / `_AUTH`) carry the
keys in the request; there is no set/get-keys ioctl and no live-key fallback. The keys are
campaign-wide (not per-input) because a regular-sealed *sealing class* runs one shared signature set
for all its members, which only verifies if the members share keys.

## Compatibility

Development is free to break the format, so this lands as `version = 1` with **no legacy flat path**.
The magic is a full `u64` sentinel both sides compare for equality; a mismatched magic or version is
rejected outright. Within a major version, a newer writer may add sections an older reader will skip.
