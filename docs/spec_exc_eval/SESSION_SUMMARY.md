# Complete summary — TikTag-v1 detection, triage, and the GVT machinery

Date 2026-10-30, branch `speculative-exception-fuzzing`. This ties together the results, every code
change, and the mechanism. Companion docs: `MECHANISM_DEEPDIVE.md` (file/line machinery),
`TIKTAG_V1_RESULTS.md` (the find + reproducibility).

## 1. Headline result

The Revizor AArch64 **speculative-exception fuzzer** detected a **genuine TikTag-v1 speculative
MTE tag-check leak** on the Pixel 8 **Cortex-X3** (core 8), from the `user_mte_skeleton.asm` template.
The violation reproduces deterministically and passes a full genuineness triage: it is a
**speculative** (not architectural) tag-check leak, and it **requires** MTE tag-checking to exist.

Violation: `violation-261030-140431`, test case #9, **time to detection ≈ 171 s**, 40-variant
equivalence class, 0 hardware tracing errors.

## 2. Recipe

- Device: Pixel 8, Tensor G3, pinned **Cortex-X3 = core 8** (the v1 / Gadget-1 speculation-shrinkage
  core; A715 cores 4–7 carry v2 store-to-load-forwarding, not v1).
- Template: `docs/tiktag_mte_ni/general_detection/user_mte_skeleton.asm` — a Spectre-v1 window (SDIV
  chain → `ORR` → `CBNZ`) + one sealed tagged `LDR [x5]` in the not-taken shadow + random fill.
- Config: `docs/tiktag_mte_ni/general_detection/s1_mte_hw_hard_noMT.yml` — `fuzzer:
  speculative-exception`, `strict`, `contract_execution_clause: [cond]`, `l1d`, `F+R`,
  `enable_pre_run_flush: true`, `executor_sample_sizes: [2000]`, no mistraining, remote adb.
- Command: `revizor.py tfuzz -s base.json -c <config> -t <template> -n 150 -i 10`.

## 3. Genuineness triage — GENUINE

| gate | tool | result |
|------|------|--------|
| 1. tag-only | accounting.py | PASS — genuine vs decoy GVT images differ only in MTE applied-tag lanes |
| 2. speculative-only leaking set | accounting.py | PASS — set 49 is a CE speculative demand (base x3); set 6 a prefetch of one; arch demands excluded |
| 3. 16-tag pointer sweep | triage_sweep.py | PASS — match tag 6 = **1.00**, every checked mismatch ≤ **0.07** |
| PMU arch-vs-spec | pmu_variant.py + knobs | **EXC_TAKEN = 0, INST_RETIRED = 69** identical for match and all mismatches → **100% speculative, not architectural** |
| MTE enable/disable | pmu_variant.py + knob | checks ON → 1.00 vs ~0.1 separation; checks OFF → ~0.36 vs ~0.36, **separation gone** → leak **is** the tag check |
| 4. memory-tag tracking | tag_track.py | INVALID control (guess-granule finder returned the wrong granule — the known 155905 limitation); inconclusive, **not** evidence against |
| dmesg | — | clean: 0 `TFSR tag fault`, 0 ESR/BUG/panic/KASAN on the genuine path |

The PMU signature is the contrast to the reclassified-architectural 072421 (which had `EXC_TAKEN`↑ and
`INST_RETIRED`~195k). Here nothing extra retires and no exception is taken: the sealed access runs under
speculation and is squashed before retirement on a tag mismatch, leaving only the set-49 cache
footprint. The MTE enable/disable control closes it: with tag checks overridden off (`TCO=1`) the
genuine-vs-decoy divergence disappears, so the leak is contingent on the architectural tag comparison.

## 4. Mechanism (one paragraph; full detail in MECHANISM_DEEPDIVE.md)

The test-case code is **byte-identical across every variant**; a variant is selected purely by its
**GlobalValueTable (GVT)** image — a small table of per-input values the code reads from reserved vector
registers (v0..v31, the SIMD section), like an in-register GOT. The MTE seal BFIs an *applied* tag into
the pointer before the sealed load and re-converges to the genuine cell tag after. The genuine variant's
tag matches the cell; a decoy's mismatches only on the speculative path. The CE models the AArch64
contract where the tag is TBI-masked for addressing (tag-blind), so genuine agrees with the contract and
a decoy diverges from it on hardware (the tag check squashes the speculative load) — the equivalence
class splits, and that split is the violation.

## 5. Changes this session (commits on `speculative-exception-fuzzing`)

- `10df5412` **stderr isolation** — `ADBConnection.run` now drives the batch over adb shell-protocol v2
  (raw, no PTY); stdout/stderr/exit are framed separately so a device error no longer pollutes the
  binary response as "bad magic". Test `unit_adb_shell_v2.py`. *(Root cause of the earlier "Hardware
  Tracing Errors" was a leftover cron-resurrected campaign racing `/dev/executor`; stopped + cron
  disabled — infra, not a commit.)*
- `5049b93e` **triage tools → GVT + TikTag-v1 results** — accounting.py / triage_sweep.py / tag_track.py
  migrated off relocation plans to GVT images; `TIKTAG_V1_RESULTS.md`.
- `235ad555` **mechanism deep-dive** — `MECHANISM_DEEPDIVE.md`, file/line references.
- `2bdeac4a` **executor control knobs** — sysfs `pfc_event1/2/3` (configurable PMU counters, for the
  EXC_TAKEN/INST_RETIRED control) and `enable_mte_check` (TCO override, for the MTE enable/disable
  control).
- `cad6ae75` **rename GST → GVT** — GlobalSealTable → GlobalValueTable repo-wide (Python, CE, kernel
  comments, tests, docs); CE rebuilt; 79 seal/CE tests pass.

## 6. TTD / rates (measured so far)

- TikTag-v1 (template, X3): **TTD ≈ 171 s** to the first genuine violation; yield ≈ a few genuine per
  tens of fills (probabilistic v1 signal, 2000 reps/measurement).
- A full TTD + bits/second table across channels (AUT*, canonicality, TikTag; with/without template;
  random) is a separate pending task (see §7).

## 7. Pending (requested, not yet done)

1. **Random** MTE-v1 hunt (`fuzz`, no template) on X3.
2. **PAC + MTE fuzzing without the speculative-exception variation** (`config_pac_mte_basic.yml`,
   fuzzer `basic`), and **MTE+PAC on the Pixel** (a remote variant of that config).
3. **Reloc-vs-GVT comparison**: assembly logic, bits/second rate, and a Pixel dmesg check that the
   genuine path raises no MTE exception (grep `TFSR tag fault`).
4. **Summarize all TTD + rates** across channels.
5. **Yesterday's cleanup**: remove mistraining end-to-end; kernel relocation removal; paper rate task.
6. (optional) fix tag_track.py's guess-granule finder to make gate 4 a valid control.
