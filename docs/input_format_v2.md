# Input format v1 (extensible, section-addressed)

Status: IMPLEMENTING (2026-06-28). Replaces the flat fixed `InputFragment`/`input_t` input_init with a
small header + a section table, so we can add per-input initial state (MTE tags, PAC keys, …) without
another fixed-layout break each time. Dev-time: we are free to break the wire format, so this lands
as `version = 1` (no legacy flat path kept). All header members are u64.

DECISION (Gal, 2026-06-28): **option A** — the header describes the per-input initialization (memory, gpr,
simd, pac_keys, mte_tags). Test-case **code keeps its own load path**; it can adopt the same header
later. Keep it simple.

## Goals
- One self-describing per-input initialization whose sections the CE locates by **type**, not by hardcoded offset.
- Carry as *initial execution-environment state* (same status as initial memory/registers): MTE tags
  (per 16B granule) and PAC keys.
- Forward-compatible: an unknown section type is skipped, not fatal — so a newer Python can talk to an
  older CE and vice-versa within a major version.

## Open decisions (need Gal)
1. **Scope of the input_init.** Two options:
   - (A) *Input-only envelope* (recommended): the header describes one INPUT (memory, gpr, simd,
     mte_tags). Test-case **code** keeps its own load path; **PAC keys** stay a per-TC ioctl OR get
     their own one-shot env input_init. Smallest change; matches "code loaded once, inputs streamed N times".
   - (B) *Unified env container*: one file = `[code][pac_keys]` (per-TC, once) + repeated input
     fragments `[memory][gpr][simd][mte_tags]`. Closer to "everything is initial env state" but couples
     code+input shipping and the multi-input streaming model.
   → Draft below assumes (A): a per-input header, with PAC keys expressible as a section for the cases
     where keys are per-input. Flip to (B) if you want code folded in.
2. **Per-input vs per-TC for each section.** memory/gpr/simd/mte_tags are clearly per-input. PAC keys
   are today per-TC (one signing key for the campaign). Keep keys per-TC (ioctl/one-shot) unless a
   test wants per-input keys — the section table allows either (present ⇒ per-input override).
3. **MTE tag coverage.** Tags over `main`+`faulty` only (the sandbox data the clamps target) =
   8192 B / 16 = **512 granules**. Packed 2 tags/byte ⇒ **256 bytes**. Extend to stack/pads later if a
   test ever tags them.
4. **Tag bit position.** File stores the *logical* 4-bit tag value (0–15) per granule, LSB-first
   nibble packing. (Note: the existing MTE ioctl comment says bits 55:52, but the Python model uses
   [59:56]; v2 sidesteps this by storing the logical value, and the CE places it in the architectural
   tag bits. Reconcile the ioctl separately.)

## Wire layout (option A)

All integers little-endian, **every member is u64** (for simplicity — no packing surprises across
Python `np.uint64` and C `uint64_t`). The header is naturally 8-byte aligned; every section payload
starts 8-byte aligned.

```
offset 0   ┌──────────────────────────────────────────────┐
           │ magic      u64  = 0x52565A49 ('RVZI')         │
           │ version    u64  = 1                           │
           │ header_len u64  = 48 + 32*n_sections          │  (start of payloads)
           │ n_sections u64                                │
           │ flags      u64  (bit0 MTE present, …)         │
           │ total_len   u64  total bytes of this input_init      │
           ├──────────────────────────────────────────────┤  section table: n_sections × 32 bytes
           │ [i] type   u64                                │
           │     flags  u64  (reserved per-section)        │
           │     offset u64  from start of input_init            │
           │     length u64  bytes                         │
           │ …                                             │
           ├──────────────────────────────────────────────┤  ← header_len
           │ section payloads, each 8-byte aligned         │
           └──────────────────────────────────────────────┘  ← total_len
```

Preamble = 6×u64 = 48 bytes; each section descriptor = 4×u64 = 32 bytes.

### Section type ids
```
0x01 MEMORY_MAIN     4096 B   sandbox main page
0x02 MEMORY_FAULTY   4096 B   sandbox faulty page
0x03 GPR               64 B   x0..x5, flags, sp  (8×u64; flags = per-flag NZCV, converted to PSTATE
                              before the CE sees it, as today in aarch64_input_layout.py)
0x04 SIMD             256 B   v0..v7 (8×256-bit)
0x05 PAC_KEYS          72 B   struct ce_pac_keys (9×u64); present ⇒ per-input key override
0x06 MTE_TAGS         256 B   512 granules of main+faulty, 4-bit tags, 2/byte, LSB-first
0x07 CODE              var    assembled AArch64 (only used if we move to option B)
0x10+ reserved for future env state (system regs, vector predicate, page attrs, …)
```

Reader rule: iterate the table; dispatch known types; **skip unknown** (use `length` to advance).
Required-section policy: MEMORY_MAIN/FAULTY/GPR/SIMD must be present in v2 (CE errors if missing);
MTE_TAGS optional (absent ⇒ region default tag, today's behavior); PAC_KEYS optional.

## Migration / compatibility
- Bump `version` to 2. The CE accepts v2; keep the v1 flat path working until Python fully emits v2
  (read the first u32: if it equals the magic, parse v2; else treat as legacy flat input).
- Keep `REVISOR_SET_PAC_KEYS` (#11) and `REVISOR_MTE_TAG_REGION` (#16) ioctls during transition; once
  tags/keys ride in the input_init, deprecate #16 (and #11 for the per-input case).
- `USER_CONTROLLED_INPUT_LENGTH` becomes `total_len` (variable) instead of a fixed equality check.

## Touch list (when we implement)
- Python writer: `src/interfaces.py` (`Input`/`InputFragment` → a v2 serializer) and
  `src/aarch64/aarch64_input_layout.py` (`_input_bytes_with_pstate` → emit the section table; fold the
  PSTATE conversion into the GPR section).
- Device write path: `src/aarch64/aarch64_executor.py:817` (`write(ExecutorMemory(...))`).
- CE/kernel reader: `src/aarch64/executor/inputs.h` (`input_t` → header + parsed pointers),
  `src/aarch64/executor/chardevice.c` (`copy_input_from_user_and_update_state`, size check),
  `src/aarch64/executor/userapi/executor_user_api.h` (`USER_CONTROLLED_INPUT_LENGTH`).
- Ground-truth CLI: `src/executor_userland/src/executor_userland.c` (writes the file verbatim — no
  change needed beyond producing v2 files).
- New shared header `executor/userapi/executor_input_v2.h` with the magic/version/type enum, included
  by both the kernel and `executor_userland` (same pattern as `executor_pac_api.h`).
```
