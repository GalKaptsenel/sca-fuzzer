# `arm_isa_extractor` — A64 ISA extractor

A self-contained, Revizor-independent extractor of the AArch64 (A64) instruction set from ARM's
machine-readable XML. It turns ARM's official ISA specification into a structured, queryable Python
model of every instruction encoding — operands, register access (read/write), immediate domains,
memory roles, and inter-operand UNPREDICTABLE constraints — which `isa_downloader` then serialises
into the `base.json` that Revizor's `isa_loader` consumes.

## Design philosophy: correct, complete, or loud

The single rule: **never be silently wrong.** Every encoding is either extracted correctly and
validated, or it **loud-fails** — it is skipped and recorded with a reason, never emitted with a
guessed/partial result. Concretely:

- No silent coercion or dropping of values (`validate.py` rejects out-of-domain/inconsistent fields).
- No unjustified single-form assumptions — each concrete asm form is enumerated and resolved.
- Register access and memory semantics come from the instruction's **ASL** (the architectural
  source of truth), not from name heuristics.
- `build_json` returns `{filename: reason}` for every encoding it could not yet handle; those are the
  known gaps (the "loud-fail families"), visible and countable, not hidden.

This is what lets `base.json` be trusted for correct-by-construction test-case generation.

## Pipeline

```
                 download.py                 pipeline.build_json()                 isa_downloader
ARM CDN ──▶ ISA_A64 XML (cached) ──▶ parse+extract+validate+serialize ──▶ ISA IR ──▶ tag + base.json
   stage 1                                  stages 2+3                              (Revizor shape)
```

1. **Stage 1 — download** (`download.py`): fetch + cache ARM's `ISA_A64` XML tarball (pinned release
   `A64-2025-09`), extract to a local XML dir. Re-downloads only when forced or the cache is absent.
2. **Stages 2+3 — `pipeline.build_json(out_json, xml_dir)`**: for every instruction XML, parse it,
   build the operand/semantics model, validate, and serialise to JSON. Returns the loud-fail map.

`isa_downloader.Downloader` (wired in `factory` as the aarch64 downloader, invoked by
`revizor.py download_spec`) drives stage 1 then `pipeline`, and adds the Revizor-facing layer
(tagging + the `base.json` shape `isa_loader` reads).

## Module map

| Module | Responsibility |
|---|---|
| `download.py` | Stage 1: download/cache/extract ARM's A64 XML (pinned releases). |
| `pipeline.py` | Orchestrates stages 2+3 over every XML; returns the loud-fail map. |
| `extract.py` | Parse one instruction XML: encodings, asm template, ASL sections (comments stripped so they never contribute reads/writes). |
| `asl.py` | Read **register access** and **memory** semantics from ASL: `X{64}(t)`/`V{}(n)`/… accessors → read/write reg-vars; `SP` when reg-var==31; `AccessDescriptor` → load/store/exclusive memory role. |
| `immediate.py` | Immediate domains from ASL constants + prose ("multiple of N", "A to B"); architectural constants (e.g. `LOG2_TAG_GRANULE`). |
| `constraints.py` | Inter-operand **UNPREDICTABLE** constraints: `if <regvar==regvar> then ConstrainUnpredictable(...)` → operand pairs that must differ (with FALSE-guard handling). |
| `operands.py` | Map asm tokens to the operand model: register-file/width prefixes (`W`/`X`/`V`/`Z`/`P`/…), extend/shift, immediates, memory components. |
| `models.py` | Data classes + enums: `Instruction`, `Operand`, `OperandKind` (REG/IMM/COND/EXTEND/…), `RegFile`, `MemRole`, `MemAccess`, `ExtractionError`. |
| `validate.py` | Per-encoding loud-fail validation: element sizes, register widths per file, field consistency — a wrong/out-of-domain field is a parsing bug, recorded, never emitted. |
| `serialize.py` | Serialise a validated `Instruction`/`Operand` to the JSON dict (the on-disk schema). |

## Regenerating `base.json`

```
python revizor.py download_spec -a aarch64 -o base.json
```
The cache lives under `~/.cache/arm_isa_parser`; regeneration is deterministic and reproduces
`base.json` byte-for-byte from the same cached XML.

## Output schema (per instruction)

A serialised instruction carries `name`, `category`, asm `template`, `tags`, and an `operands` list.
Each operand (`serialize.operand_dict`) has `kind`, `read`/`write`, `width`, `signed`, `values`,
`imm_ranges`, `reg_range`, `reg_file`, `asl_index`, `sp_capable`, and `mem_role`. `isa_loader`
reads exactly this shape (the x86 path stores its single category in `tags` instead).

## Known gaps (loud-fail families)

`build_json`'s return map enumerates the encodings not yet handled (e.g. SME ZA-tile template
placeholders with no operand). These are *skipped and reported*, never emitted wrong. Recovering
the remaining families is tracked separately; the count is a direct measure of extractor coverage.

## Extending it

To support a new encoding family: add the asm-token/operand mapping in `operands.py`, the ASL
read/write or memory rule in `asl.py`, any immediate-domain rule in `immediate.py`, and a
`validate.py` rule for the new field shape. Keep the correct-or-loud contract: if a case can't be
handled correctly yet, let it loud-fail rather than emit a guess.
