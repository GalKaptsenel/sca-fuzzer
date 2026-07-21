---
name: craft-executor-inputs
description: Hand-build the two artifacts the Revizor kernel executor loads — a test-case body (raw AArch64 machine code) and a REIF input file (initial memory + registers) — from human-friendly sources (a .asm file and a JSON spec). Use when you need to run a specific, controlled program+input on /dev/executor without the Python fuzzer: a PoC, a minimized violation, a targeted experiment. Both tools reuse the real Revizor code paths (the asm_to_bytes assembler and the REIF encoder), so their output is byte-identical to what the fuzzer produces. Feed the results to the executor-userland skill (`w file`).
---

# Crafting executor inputs by hand

The kernel executor consumes two separate artifacts, each written to `/dev/executor` with
`executor_userland ... w file` (see the **executor-userland** skill):

1. a **test-case body** — raw AArch64 machine code (stripped `.text`, no ELF), and
2. one or more **REIF input files** — the per-input initial state (main/faulty memory + GPRs).

These two tools build those from readable sources. Both **reuse the actual Revizor code** (no
re-implemented format logic), so what you load equals what the fuzzer loads.

## Locating the tools (resolve relative to the repo)
```
ROOT=$(git rev-parse --show-toplevel)
PY=/home/gal_k_1_1998/revizor/revizor-venv/bin/python     # needs numpy; /usr/bin/python3 won't do
GEN="$ROOT/src/executor_userland/input_generator/generate_reif_input.py"
ASM="$ROOT/src/executor_userland/asm_compiler/compile_test_case.py"
```

## 1. Test-case body — `asm_compiler/compile_test_case.py`
Compiles an AArch64 `.asm` into the raw binary the module loads. Reuses
`Aarch64Generator.in_memory_assemble` → the `asm_to_bytes` helper (cross `as` + `objcopy -O binary`).
```
$PY "$ASM" --input body.asm --output test_case.bin [--print]
printf 'ldr x1,[x0]\nadd x1,x1,#1\nret\n' | $PY "$ASM" --output test_case.bin   # stdin also works
```
- Output is the flat machine code (e.g. `ldr x1,[x0]` → `01 00 40 f9`).
- Warns if the body exceeds `MAX_TEST_CASE_SIZE` (one page, 4096 B) — the kernel rejects larger.
- Sample: `asm_compiler/test_case_pattern.asm`.

## 2. Input — `input_generator/generate_reif_input.py`
Turns a JSON spec into a `.reif` file. Reuses the Revizor encoder `build_input_init`, so the section
layout (magic, table, MEMORY_MAIN/FAULTY, GPR) is the real wire format (see `docs/reif_input_format.md`).
```
$PY "$GEN" --input spec.json --output input.reif [--seed N] [--print]
```
- `--seed N` makes the random fill reproducible (same seed → byte-identical file).
- `--print` also writes `<output>.txt` with the register summary.
- Sample spec: `input_generator/input_pattern.json`.

### JSON schema (every field optional; unspecified bytes/registers are randomised)
```json
{
  "registers": { "x0": 1, "x1": -8, "x2": 1024, "x3": 101112,
                 "x4": 432345564227567616, "x5": 161718, "flags": 0, "sp": 4096 },
  "memory": {
    "main_region":   { "0": 1, "128": 192, "1025": 8 },
    "faulty_region": { }
  }
}
```
- Register values are written **verbatim** (negatives are two's-complement u64; `flags` is the ARM
  PSTATE value with NZCV in bits 31:28; the kernel overrides `sp` at run time regardless).
- Memory keys are byte offsets (decimal or `0x…`) into the 4 KB region; values are bytes (masked to
  `0xFF`). Offsets out of `[0, 4096)` raise.

## Putting it on the device (via the executor-userland skill)
```
EU="$ROOT/src/executor_userland/executor_userland /dev/executor"
$EU 1 ; $EU w test_case.bin                 # CHECKOUT_TEST, load the body
IID=$($EU 5 | sed -n 's/.*: //p')           # ALLOCATE_INPUT → read the id back
$EU 4 $IID ; $EU w input.reif               # CHECKOUT_INPUT, load the input
$EU 8 ; $EU 4 $IID ; $EU 7                   # TRACE, re-checkout, MEASUREMENT
```

## Notes
- Both tools import the Revizor package (numpy); run them from the repo with the venv Python.
- They deliberately do **not** re-pack the formats — if the REIF layout or the assembler changes, the
  fuzzer's own code changes and these tools follow automatically. No format constants are duplicated.
- REIF sections beyond memory+GPRs (MTE tags, PAC keys, code relocations, branch training) are owned
  by the fuzzer's sealer/generator, not by this manual path; add them via the encoder if ever needed.
