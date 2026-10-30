# Speculative-exception fuzzing on AArch64 — mechanism deep-dive

A file/line-referenced account of the machinery behind the TikTag-v1 detection, written so the paper
and another working session can build on it. All references are `path:line` against this repo at the
current branch (`speculative-exception-fuzzing`). Sections 1–6 are the general machinery; section 7 is
the TikTag-v1 instantiation; sections 8–9 are triage and transport.

## 0. One-paragraph overview

"Speculative-exception fuzzing" (formerly "non-interference") tests that a precondition which holds
architecturally but is violated only on a speculative path cannot leak. For each sealed access the
fuzzer runs several *variants* of one architectural input: a **genuine** variant (the precondition
holds — correct PAC signature / matching MTE tag / canonical address) and one or more **decoy**
variants (the precondition is violated only on the speculative path). The test-case code is
**byte-identical across every variant**; the only per-variant difference is a small table of values the
code loads from reserved vector registers — the **GlobalValueTable (GVT)**. A violation is an
equivalence class of variants whose hardware traces diverge: the speculative precondition violation
left a microarchitectural footprint.

## 1. GlobalValueTable (GVT): in-register value delivery

`src/aarch64/seal/global_value_table.py` — the GVT is "a GOT, but in a register": the code reads each
seal's value from a reserved vector lane instead of having the value spliced into the code.

- `VectorGlobalValueTable` (`global_value_table.py:99`) backs the table with the vector file, allocating
  top register first (`v31` down, `TOP_REG=31`, `SIMD_BASE_REG=0`, `global_value_table.py:13-14`),
  byte-aligned, never straddling a register (`allocate`, `:113-123`). Capacity is a bit budget rounded
  up to whole registers (`:103-109`); overflow raises `GlobalValueTableCapacityExceeded` (`:119-121`) —
  no fallback. The sealer builds it over the whole vector file (512 bytes) at `sealer.py:615`.
- `_VectorLocation.load` (`global_value_table.py:71-78`) emits the one fixed instruction that reads a
  lane: `UMOV <w|x>tmp, v<reg>.<b|h|s|d>[idx]`, element size chosen from the field width
  (`_UMOV_LANE`, `:11`). `byte_range()` (`:80-82`) maps a lane to its byte offset in the REIF SIMD
  section.
- `_VectorGVTImage` (`global_value_table.py:85-96`) is one per-input fill: `store(location, value)`
  writes the value little-endian into the lane's byte range (`:89-93`); `to_section()` emits a
  `SimdSection` (`:95-96`). The whole image is `SIMD_SUBREGION_SIZE` bytes.

## 2. SealValue: what each lane holds and how the fixed slot applies it

`src/aarch64/seal/seal_values.py` — one class per value kind; each owns its width, encoding
(right-aligned in the lane), and the fixed apply sequence (a lane load + one data-processing op).

- `MteTag` (`seal_values.py:60-70`): a 4-bit allocation tag in a 1-byte lane; `apply` = `UMOV` +
  `BFI target, tmp, #56, #4` — inserts the tag into the pointer's tag field [56,60).
- `PacSignature` (`seal_values.py:35-57`): the PAC-field window right-aligned; `apply` = `UMOV` +
  `BFI target, tmp, #lsb, #nbits`.
- `MaskFlip` (`seal_values.py:73-90`): a bitmask XORed in (canonicality / branch-target / sign-ext);
  `apply` = `UMOV` + `EOR target, target, tmp, LSL #lo`. The genuine value is mask 0 (a no-op), so a
  canon/branch seal needs no CE injection — its genuine image is all-zero.

The scratch GPR for the `UMOV`→`BFI`/`EOR` is a fixed reserved register (`SEAL_SCRATCH_REGISTER`,
defined in `src/aarch64/aarch64_target_desc.py`, imported at `sealer.py:29`, used at `sealer.py:134,173`),
excluded from the body's register budget.

## 3. The sealer: slots, resolution, genuine/decoy images

`src/aarch64/seal/sealer.py`.

- A `Sealing` (`:60`) owns one seal: its fixed slot instructions (`slot_insts`) and `store_into(image,
  applied, genuine)` (`:71`). `MteSealing` (`:158`) is the TikTag seal: it BFIs an **applied** tag from
  one lane before the access, and — when the access preserves its base — BFIs the **genuine cell tag**
  from a second lane after it, so every variant re-converges to the real tag and only the sealed access
  sees the perturbation (`:158-178`). `store_into` (`:180-183`) writes the applied lane (decoy-variable)
  and the re-converge lane (always genuine).
- `SealedTestCase` (`:600`) places the slots (`_insert_slots`, subclass-specific) and resolves. The
  salt that seeds every decoy choice is derived from the **test-case bytes** (`stable_hash_bytes`,
  `:613`), and the global RNG is seeded from it only around slot insertion then restored (`:617-622`),
  so fuzzing and `reproduce` seal **identically** (this is what makes the find reproducible).
- `resolve(inp)` (`:652-663`) runs one CE trace per input (memoized), builds the `_Resolved` entries,
  and returns a `ResolvedSealingTestCase`.
- `ResolvedSealingTestCase` (`:383`): `object_code` (`:406-408`) is the input-independent code; a
  variant is just a GVT image. `genuine()` (`:415-416`) stores every slot's correct value; `decoy(rng)`
  (`:418-422`) perturbs a non-empty subset of the *eligible* slots — those that are **speculative and
  carry alternatives** (`_eligible`, `:424-428`). `_build_image` (`:433-442`) is the whole image fill:
  for each entry, `store_into(image, alt-if-perturbed-else-genuine, genuine)`. `collapse_key` (`:444`)
  groups inputs whose every slot resolves to the same (value, speculative) so one sealed program is
  arch-safe for all of them.

## 4. The genuine contract trace + CE value injection

The genuine contract trace must run the *same fixed code* as the genuine variant, but the CE cannot
know the runtime PAC signature / cell tag a priori. The resolve trace ships a **blank** GVT image plus
an **emulation table**, and the CE injects each site's genuine value into the image as it reaches it.

- `SealedTestCase._resolve_trace` (`sealer.py:671-680`): assembles the fixed code, ships
  `simd = blank image` and `emul_table = pack_emul_table(self._emul_rows())`. `MteSealing.emul_row`
  (`sealer.py:194-201`) carries the slot's first instruction, the access anchor, and the two lane byte
  offsets (`EMUL_OP_MTE_RETAG`).
- CE dispatch: `instruction_emulator.c:13-14` maps `EMUL_OP_PAC_AUTH`→`pac_emulate_auth`,
  `EMUL_OP_MTE_RETAG`→`mte_emulate_retag`.
- `mte_emulate_retag` (`src/aarch64/contract_executor/mte_seal_plugin.c:22`) reads the accessed cell's
  tag and writes it into the GVT image buffer at both lane offsets
  (`simulation.sim_input.simd[row->gst_offset] = applied; ...simd[row->gst_offset2] = applied;`,
  `mte_seal_plugin.c:65-66`), with bounds checks that abort loudly (`:60-62`).
- `base_hook.S` reloads the whole vector file from the image after each C hook, so the next `UMOV`
  reads the just-injected value: `gst_image_target` is patched to `&sim_input.simd` (`base_hook.S:2,10`)
  and v0–v31 are reloaded before control returns to the test case (`base_hook.S:45-47`). This is the
  GVT analogue of the old "op writes code" machinery; there are no code splices.

Because the genuine image makes the `AUT*`/tag-check pass, the single resolve pass **is** a faithful
genuine contract trace (same trust basis as the pre-GVT, relocation-era code).

## 5. The executor: per-variant images, strict vs contract_driven

`src/aarch64/aarch64_executor.py`.

- `variants_for_input` (`:817-826`) builds `{variant_name: ExecutorInput(inp, gvt=image, mte_tags=…,
  pac_keys=…)}` from the resolved genuine/decoy images. `genuine_variant` (`:835`) is the baseline;
  decoys are `SpecExcVariant.decoy_n(i)` (`:850-853`).
- Strict mode `_seal_input` (`:939-947`) ships one sealed program per sealing class — genuine, or a
  decoy by a per-class coin — so a stale `AUT*`/tag fault can never reset the box. contract_driven mode
  (`as_executor_input`, `:882-887`) carries the GVT in the input's own SIMD subregion (`gvt=None`) and
  freezes only the arch-consumed lanes.
- The image is shipped as the REIF SIMD section: `ExecutorInput.serialize` (`aarch64_executor_input_
  encoder.py:169-177`) uses `self.gvt` if set, else the input's own SIMD; `build_input_init`
  (`:127-142`) appends it as `SEC_SIMD`.

## 6. The hardware path: load the GVT into v0–v31

`src/aarch64/executor/`.

- Parse: `chardevice.c:702-710` copies `SEC_SIMD` into `input_t.simd` (zeroed when absent; exact-size
  checked).
- Stage: `measurement.c:69-72` copies `input->simd` into the sandbox right after the 8 register slots,
  so the JIT loads from a fixed address.
- Load: `templates_jit.c:125-127` emits `jit_ldr128` for every register, loading **v0..v31** from that
  staged image immediately before the body — the registers the `UMOV` slots read.
- Preserve: the run clobbers v0..v31, and it executes in IPI context where `kernel_neon_begin()` is
  illegal, so `chardevice.c:288-309` saves/restores q0..q31 by hand around the run.

## 7. TikTag-v1 instantiation (the find)

The TikTag-v1 gadget is a conditional-branch speculation-shrinkage leak: a tag-checked load on a
mispredicted path shrinks the speculation window on a tag **mismatch**, so a downstream probe's cache
footprint depends on the tag.

- **Template** `docs/tiktag_mte_ni/general_detection/user_mte_skeleton.asm`: a Spectre-v1 window (an
  SDIV dependency chain → `ORR` → `CBNZ .bb_0.2`), one sealed tagged `LDR x12,[x5]` in the
  not-architecturally-taken shadow block `.bb_0.1`, and `.macro.random_instructions.*` fill before and
  after. The sealer wraps that `LDR` in an `MteSealing` (applied tag before, re-converge after).
- **Contract** `docs/tiktag_mte_ni/general_detection/s1_mte_hw_hard_noMT.yml`: `contract_execution_
  clause: [cond]` (the conditional-branch window, not store-bypass), `l1d` observation. The CE models
  the AArch64 contract where the tag is TBI-masked for addressing, so under the contract the
  speculative load touches the same address regardless of its tag — **tag-blind**. On hardware the MTE
  tag check squashes the speculative load on a mismatch. So: genuine (matching tag) and the tag-blind
  contract agree; a decoy (mismatching tag) diverges from the contract on hardware → the equivalence
  class splits → violation. Core 8 (Cortex-X3) is the v1 / Gadget-1 core.
- **Result** `violation-261030-140431`: TTD ≈ 171 s, 40-variant class, reproduces deterministically
  (section in `TIKTAG_V1_RESULTS.md`). No MTE exception is written to the kernel log on the genuine
  path (the executor would report `TFSR tag fault` at `measurement.c:404`; zero seen).

## 8. Triage machinery (genuineness gates)

A violation is only called GENUINE after four gates, all GVT-aware
(`docs/tiktag_mte_ni/general_detection/{auto_triage,triage_sweep,tag_track}.py`,
`.claude/skills/verify-mte-tag-leak/accounting.py`):

1. **Tag-only** (`accounting.py`): genuine vs decoy GVT images differ only in `MteSealing` applied-tag
   lanes (code is byte-identical by construction).
2. **Speculative-only leaking set** (`accounting.py`): a hardware-diverging set must be a CE
   speculative access (or a prefetch stream of one), not an architectural demand — architectural
   demands commit with the correct tag regardless of the decoy, so their divergence is prefetcher/batch
   contamination.
3. **16-tag pointer sweep** (`triage_sweep.py`): rewrite the suspect site's applied-tag lane over all
   16 values; the matching tag must dominate every checked mismatch (TCMA-unchecked tags 0/15 excluded).
4. **Memory-tag tracking** (`tag_track.py`): patch the guess-load granule's *memory* tag and confirm
   the detected match tag moves with it — the decisive pointer-vs-memory-check proof. An ineffective
   patch is declared INVALID, never a false negative.

The sweep/track build variants by rewriting the GVT image (`store_into` on the suspect sealing, cell
lane kept genuine) and shipping it with `dataclasses.replace(genuine_variant, gvt=image)` — no
relocation plans.

## 9. Transport (adb)

`src/aarch64/aarch64_connection.py` `ADBConnection.run` drives the measurement batch over adb's
**shell-protocol v2 (raw, no PTY)**: stdout, stderr and the exit code are framed on separate streams
(`:149-...`), so the binary batch response is never polluted by a diagnostic the userland writes to
stderr. A nonzero exit raises `IOError` carrying stderr, which `run_batch` (`aarch64_kernel.py`) maps
to `HardwareTracingError` (skip the test case). Previously the `exec:` service merged stderr into
stdout and a device error surfaced as an opaque "bad magic".

## 10. Determinism

The sealer salt is the test-case byte hash (`sealer.py:613`), and the global RNG is seeded from it only
around slot insertion (`:617-622`). So a saved violation's `generated.asm` + its `.reif` inputs
reconstruct byte-identical variants under `revizor.py reproduce`, which is why the find reproduces
exactly.
