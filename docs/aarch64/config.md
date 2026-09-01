# AArch64 Configuration Reference

Architecture-specific configuration options for AArch64. Common options (shared with other
architectures) are documented in [the common config reference](../user/config.md); the
cross-architecture speculative-store-bypass knob (`enable_speculative_store_bypass`) lives there
too. The options below only take effect under `instruction_set: aarch64`.

## Generation

```yaml
Name: avoid_extended_memory_operands
Default: True
```

Skip memory-access instruction forms whose address uses an extended-register index
(UXTW/SXTW/SXTX/UXTX), keeping base / base+immediate / plain (LSL) register-offset forms. Set
`False` to also emit extended-register addressing. **Temporary / WIP**: defaulted on because
emitting the extended forms was observed to reduce the number of violations found, for a reason
not yet understood; this option should be removed once that is investigated.

```yaml
Name: supported_instructions
Default: (allow-list)
```

Allow-list of instruction mnemonics the generator may emit (an instruction is generatable only if
it appears here). Empty/unset means "no allow-list".

## Executor

```yaml
Name: in_memory_assembler
Default: True
```

Assemble each test case in memory, so a fuzzing run writes no per-test-case asm/object files (they
are produced only when a violation artifact is saved). x86 leaves this `False` and loads the object
file from disk.

```yaml
Name: enable_branch_mistraining
Default: False
```

Before measuring an input, saturate each architectural conditional branch in the opposite
direction so the first hardware run mispredicts and opens a speculative window. **Keep off (WIP)**:
the current implementation trains toward the *architectural* direction and *suppresses* the
misprediction Spectre-v1 needs.

## Non-interference

```yaml
Name: noninterference_mode
Default: None
Options: 'pac' | 'mte'
```

Which non-interference contract the `fuzzer: non-interference` fuzzer tests: pointer
authentication (`pac`) or memory tagging (`mte`). No default — must be set explicitly when using
that fuzzer.

```yaml
Name: ni_decoys_per_input
Default: 1
```

Number of decoy test-case variants compared against the genuine baseline per input (`>= 1`). Each
decoy perturbs a random subset of speculative slots, so `K` decoys give `~1 - 0.5^K` per-slot
coverage; all `K` share the input's single contract-executor resolve. This broadens coverage on the
test-case axis without input-boosting.

```yaml
Name: pac_auth_weight
Default: 0.2
```

Relative weight of AUTH-strip insertions in the PAC non-interference stage-1 instrumentation.

```yaml
Name: pac_xpac_weight
Default: 0.2
```

Relative weight of XPAC-strip insertions in the PAC non-interference stage-1 instrumentation.

```yaml
Name: pac_seal_prob
Default: 1.0
```

Probability that an eligible memory access is PAC-sealed (authenticated) at all. Values `< 1` leave
some accesses as a raw, sandbox-clamped pointer with no AUT*. Decided once per test case.

```yaml
Name: pac_strip_prob
Default: 0.0
```

Probability that a sealed slot renders as the arch-safe XPAC* strip instead of a real AUT*. A strip
never poisons under speculation, whereas an AUT* against the decoy signature does.

## Canonicality (non-interference)

Canonicality is a third non-interference primitive, alongside PAC and MTE, for the
`fuzzer: non-interference` fuzzer. Unlike PAC/MTE it introduces **no new instructions**: it seals
the memory accesses the generator already emits, so it is independent of `instruction_categories`
and is switched on by its own flag. The genuine run uses the canonical (sandbox-clamped) pointer;
the decoy **flips** a non-canonical mask into the base register right before a speculative-only
access — so the architectural path stays canonical (a non-canonical address would fault) — and flips
it back right after (when the access preserves its base), so the decoy **re-converges** with the
baseline: only the sealed access differs, and the flipped bits cannot cascade downstream (e.g. a
later shift moving them into the cache-set index). Any hardware htrace divergence between genuine and
decoy is therefore a canonicality leak: the microarchitecture distinguished two
architecturally-identical programs.

The mask is applied with `EOR` (a flip), not `ORR`, because a canonical address is regime-dependent:
a TTBR0 (user) pointer has its high bits `0`, a TTBR1 (kernel) pointer has them `1`. The Revizor
sandbox runs at EL1 over a **kernel** pointer, so ORing high bits would be a no-op — the bits are
already `1`. A flip breaks canonicality in both regimes. The flip is confined to `[54:VA_SIZE]`
(never bit 55, the regime selector, and never the VA/set-index bits), and is reverted after the
access so only that one access differs.

Because the decoy is an ordinary `EOR Xd, Xd, #imm` spliced through the existing code-relocation
path, canonicality needs no contract-executor plugin and no per-input REIF section.

```yaml
Name: enable_canonicality
Default: False
```

Master switch. Detected as the `canon` seal primitive. Requires the non-interference fuzzer and a
category that emits memory accesses (`BASE-MEM-LOAD` / `BASE-MEM-STORE`) for there to be anything to
seal. Currently exclusive with PAC/MTE (combined primitives are not yet wired).

```yaml
Name: canonicality_seal_prob
Default: 1.0
```

Probability that an eligible memory access is canonicality-sealed. Values `< 1` leave some accesses
canonical in both variants. Decided once per test case.

```yaml
Name: canonicality_mask
Default: None
```

Fix the non-canonical flip mask. `None` (default) means each decoy independently picks a **random
contiguous run** within the guaranteed-fault range `[54:VA_SIZE]` — a random subset of the faulting
bits per decoy, analogous to PAC's forgery pool. If set, it must be a single contiguous run within
`[54:VA_SIZE]`: flipping any `[54:VA]` bit disagrees with the never-flipped regime-selector bit 55, so
the address is non-canonical in either regime and under TBI on/off, while the VA bits and the
cache-set index `[11:6]` stay untouched. Set a single bit (e.g. `1 << 47`) to probe one boundary.

```yaml
Name: va_size
Default: None
```

Effective kernel (TTBR1) VA size in bits — the single source of truth for both PAC and
canonicality (the PAC/canonicality field lives at bits `[54:va_size]`). `None` (default) means it is
read from the executing device (`TCR_EL1.T1SZ`, exposed at `/sys/executor/system/va_bits`), which is
correct for local and remote runs alike. Set it explicitly only when the generating machine cannot
read the executing target (its value then wins). A mismatched `va_size` makes the PAC auth/strip
overwrite real address bits and corrupt the sandbox pointer, so leaving it unset is preferred.
