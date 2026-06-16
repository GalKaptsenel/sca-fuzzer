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
Name: pac_auth_weight
Default: 0.2
```

Relative weight of AUTH-strip insertions in the PAC non-interference stage-1 instrumentation.

```yaml
Name: pac_xpac_weight
Default: 0.2
```

Relative weight of XPAC-strip insertions in the PAC non-interference stage-1 instrumentation.
