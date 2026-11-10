# Results and methodology summary (for the paper)

A consolidated reference of the methodology, the measurements, and the timings produced so far.
Platforms are named explicitly per result. Numbers marked `[PENDING]` are being measured and will
be filled when the runs complete.

---

## 1. Platforms

| Label | Core | Device | Role |
|---|---|---|---|
| N3 | Arm Neoverse N3 | server | primary evaluation of the canonicality and alignment leak and the PAC bit-55 leak |
| X3 | Arm Cortex-X3 (Tensor G3, core 8) | Pixel 8, over the adb tunnel | MTE non-interference (TikTag) hunt |
| A715 | Arm Cortex-A715 (Tensor G3, core 4) | Pixel 8, over the adb tunnel | MTE non-interference (TikTag) cross-core check |
| Alder Lake | Intel Alder Lake | x86-64 | partial cross-ISA reproduction |

The Tensor G3 cluster layout is cores 0-3 Cortex-A510, cores 4-7 Cortex-A715, core 8 Cortex-X3.
The transmit-side time-to-detect and the channel rates below for the canonicality and alignment
leak are measured on N3. The TikTag results are measured on the Pixel 8.

---

## 2. Methodology

### 2.1 Fuzzer
Revizor ported to AArch64, driven through a contract executor (the CE, a local reference model) and
a kernel executor on the device. Two fuzzing entry points are used:
- random generation (`revizor.py fuzz`), and
- template fuzzing (`revizor.py tfuzz`), which plants a fixed gadget and fills the remaining
  `random_instructions` holes freshly per test case.

### 2.2 Precondition fuzzing
Instructions impose preconditions on their operands and take one path when the precondition holds
and another when it is violated on a speculative path. The fuzzer drives the violated path and
compares it against the held path. Three preconditions are covered:
- indirect branch target valid vs corrupt (`blr`),
- pointer-authentication signature valid vs wrong (`autia`),
- memory-tagging tag match vs mismatch (`ldr` under MTE).

The familiar good-input vs bad-input pair is the special case where one lane keeps every
precondition and the other violates it speculatively.

### 2.3 Source-agnostic detection (generalizes priming)
The detector observes only the L1D cache and still catches a leak carried by any structure whose
effect reaches the L1D (branch-target buffer, hardware prefetcher, translation-lookaside buffer).
The combined leaking-and-detecting input is split into a leaking input and a detecting input. The
detecting input is fixed as the last input and the input series is searched for the true leaking
input with two localizers:
- linear search, complexity O(n),
- binary search over the experiments, complexity O(log n).

The method reduces to priming: it asks for the same program, the same starting microarchitecture,
and two inputs that differ, which is exactly what priming asks for, while adding no assumption
about which structure carries the leak.

### 2.4 Non-interference (NI) fuzzing for MTE and PAC
An NI violation compares variants of one architectural input that differ only in a pointer's MTE
tag (or PAC signature): a baseline with the matching tag and a decoy with a mismatched tag. A real
TikTag leak is a speculative access behind the tag check that executes on the match and is
suppressed on the mismatch, so the two variants' hardware traces diverge.

### 2.5 Triage pipeline (100 percent integrity)
Every NI violation is triaged to a verdict of GENUINE, QUESTIONABLE, or NOISE through the following
gates, and nothing is called genuine unless it passes the decisive ones:
1. Tag-only diff: the two counterexample binaries differ only in ADDG tag words, never in code.
2. CE speculative-only accounting: the diverging cache set is reached only speculatively (not an
   architectural demand).
3. Pointer-tag sweep: over all 16 tag values under a fixed prefix (swap only the suspect input),
   the matching tag's leak must dominate every mismatch beyond the repeat jitter.
4. PMU differential, using a kernel module that exposes INST_RETIRED, EXC_TAKEN and MEM_ACCESS: a
   genuine speculative tag-check leaves INST_RETIRED flat and EXC_TAKEN at zero (the mismatch
   causes no retiring fault and no exception) while MEM_ACCESS drops on the mismatch (the
   speculative tag-checked access and its dependent are suppressed). An architectural fault instead
   raises INST_RETIRED and EXC_TAKEN, which is how an architecturally-retiring tagged access is
   excluded.
5. Memory-tag tracking: retag the accessed granule and confirm the signal moves with the granule
   tag. This is read on the L1D cache and is reliable only when that channel is clean.

---

## 3. Leaks and measurements

### 3.1 Canonicality and alignment leak (speculative), measured on N3

| Boundary | Channel | Rate |
|---|---|---|
| EL2 to EL0 | dTLB | about 1230 bit/s |
| VM to host | BTB | 100 percent at about 130 bit/s |
| EL0 to EL0 | BTB | about 65 bit/s with the PMU, about 7 bit/s rootless with cntvct |

- Time to detect (the fuzzer surfacing the leak): about 10 s.
- End to end: recovery of an RSA-2048 private key through an OpenSSL ROP chain, and blind
  stack-canary recovery.
- Reproduced on Cortex-A715 (mobile) and partially on Intel Alder Lake (x86-64).

### 3.2 Pointer-authentication bit-55 leak (AUT*), cross-privilege

Every AUT* sign-extends using bit 55 over the PAC bits. Running in the victim's kernel, when bit 55
is 1 the sign extension is correct and the access updates the microarchitecture; when bit 55 is 0
the sign extension is incorrect and the access has no effect. AUT* therefore leaks bit 55, one bit
per run, across the privilege boundary. The branch-target buffer allocates for any canonical target
regardless of range, whereas this leak needs the cache and translation-lookaside buffers that gate
on an in-range access.

### 3.3 Known leaks detected by the tooling
- TikTag, an MTE speculative tag-check leak, found via precondition fuzzing using a Spectre-v1
  template (Kim et al., S&P 2025).
- AfterImage, an x86 hardware-prefetcher leak, found via source-agnostic detection (Chen et al.,
  ASPLOS 2023).

---

## 4. TikTag on the Pixel 8 (new, this evaluation)

### 4.1 Setup
- Template: `user_mte_skeleton.asm`, driven by `revizor.py tfuzz`.
  Structure: random prologue; a chain of eight `SDIV x9,x9,x2` so x9 resolves late; `ORR x9,x9,#1`
  then `CBNZ x9,.bb_0.2` (architecturally taken, so the shadow block is skipped); in the shadow
  block, reached only on a mispredicted fall-through, the sealer-decoyed tag-checked load whose base
  is the ADDG-sealed x2; random epilogue.
- Config: `s1_mte_hw.yml`: NI fuzzer, categories MTE and MTE-TAGMEM, cond contract, L1D observation,
  Flush and Reload, superbatch size 1, inputs-per-class 2, pinned core 8 (X3) or core 4 (A715) via
  `s1_mte_hw_a715.yml`, remote over the adb tunnel.

### 4.2 Finding on X3
First run found `violation-261026-134933` at test case 9 in 34.2 s. Counterexample: Input #4
(matching tag) vs Input #12 (mismatched tag), a tag-only variant pair. The diverging cache set is
27, reached speculatively with base register x2.

A late-resolving branch is required. The alternative Python template `mte_tiktag_pytmpl.py`, which
forces the branch architecturally not-taken with `SUBS`->`EQ`, found nothing in about 290 test cases
because its window never opened and every test case was discarded on the fast path. The user
template's `CBNZ` fed by the slow SDIV chain is what opens the window.

### 4.3 Triage result on X3 (GENUINE, PMU-confirmed)

| Variant | set 27 residency | INST_RETIRED | EXC_TAKEN | MEM_ACCESS |
|---|---|---|---|---|
| baseline (match) | 1.00, 1.00, 1.00 | 78 | 0 | 9.0 |
| decoy (mismatch) | 0.99, 0.06, 0.99 | 78 | 0 | ~7.0 |
| noncanon (mismatch) | 0.05, 0.06, 0.97 | 78 | 0 | ~7.0 |

- Tag-only diff: true (ADDG tag words on x0, x2, x5).
- CE speculative-only: set 27 (base x2); architectural-demand sets are {3, 17, 47, 57, 58}.
- Pointer-tag sweep (fixed prefix, 2000 reps): matching tag 1.00, every checked mismatch at most
  0.17; removing the sealed site at offset 0xbc collapses the signal to 0.01.
- PMU differential: INST_RETIRED flat at 78 and EXC_TAKEN zero across every variant, so the
  mismatch causes no retiring fault and no exception (this excludes an architecturally-retiring
  tagged access). MEM_ACCESS is a stable 9.0 on the match and about 7.0 on the mismatch, so the
  mismatch suppresses about two speculative accesses (the tag-checked load and its dependent).
- Memory-tag tracking: inconclusive on the L1D channel, which is prefetcher-contaminated here (the
  set-27 residency flips between 0.06 and 0.97 on the mismatch with a wide multi-set block).

Verdict: a genuine speculative MTE tag-check leak (TikTag-v1), confirmed on the INST_RETIRED,
EXC_TAKEN and MEM_ACCESS counters. The L1D set-27 cache channel it surfaces on is noisy for this
gadget.

### 4.4 Time to detect (TTD)

TTD is the wall-clock time from the start of a template-fuzzing run to the first violation. Each run
uses a different program-generator seed, so the holes are filled differently and the first violation
lands at a different test case. Twenty runs per core, stopped at the first violation, with a 300 s
cap per run.

Device throughput on the adb path is about 0.44 to 0.62 test cases per second.

**X3 (Cortex-X3, core 8):** 20 runs, seeds 1 to 20. The initial discovery at seed 0 hit test case
9 in 34.2 s.

- Runs that found a violation: 20 / 20
- Median TTD: 86.5 s
- Mean TTD: 131.9 s
- Min / max: 39.0 / 305.0 s
- Standard deviation: 92.8 s
- Per-run TTD in seconds, sorted: 39.0, 51.2, 52.1, 57.6, 67.4, 70.5, 74.3, 76.4, 78.3, 79.3,
  93.6, 99.0, 99.6, 114.3, 205.5, 230.8, 260.9, 286.0, 296.8, 305.0
- The detecting test case ranged from 11 to 118 across the seeds, which is the source of the spread.

**A715 (Cortex-A715, core 4):** reproduction check with the v1 template (find only).

- The v1 template found no violation on the A715 within the budget. This is the expected result:
  the A715 is not vulnerable to v1 (see the cross-core characterization below).

### 4.5 Cross-core characterization with the official PoC

The TikTag authors' own proof of concept, run on this Pixel 8, confirms that the two gadgets are
microarchitecture-specific and complementary. Each run is 100 trials over all 16 tags; the score is
the number of trials in which the leaked tag equals the correct tag (random baseline 100/16 ~ 6).

| Core | v1 (Gadget 1, speculation shrinkage) | v2 (Gadget 2, store-to-load forwarding) |
|---|---|---|
| Cortex-X3 (core 8) | 100 / 100 | 1 / 100 |
| Cortex-A715 (core 4) | 6 / 100 (and 3, 7, 4 / 100 for variant 1, variant 2, async) | 100 / 100 |

The X3 is vulnerable to v1 and not v2; the A715 is vulnerable to v2 and not v1. No v1 variant or MTE
mode makes v1 leak on the A715. Our Revizor template `user_mte_skeleton.asm` is a v1 gadget, so it
finds the leak on the X3 and nothing on the A715, which matches this characterization.

### 4.6 TikTag-v2 template (store-to-load forwarding)

`user_mte_skeleton_v2.asm`: the same late-resolving SDIV window and `CBNZ`, and in the speculative
shadow a store and a load to the same sealed pointer (`str x4,[x5]`, `ldr x4,[x5]`) so the load takes
store-to-load forwarding, followed by a probe on the forwarded value. Config `s2_mte_hw_a715.yml`
uses the cond and bpas contract clauses and disables the fast-path model, because the contract
executor does not model tag-gated store-to-load forwarding and would otherwise discard every test
case. v2 is a precondition-fuzzing target under the same MTE tag-match precondition as v1; the
violated path manifests as suppressed forwarding rather than a suppressed load.

Findings so far (A715, core 4):
- Fuzzing the v2 template produced no violation (224 test cases with SSBS off, then under SSBS on).
- The sealer does seal stores: the data-site eligibility in sealer.py treats any non-STG/LDG memory
  access, load or store, as a sealable tag-check site (ADDG tag decoy), with base_preserved true for a
  store. So v2's store-gating is expressible; this is not a sealer limitation.
- A controlled measurement of the template's baseline (match) vs decoy (mismatch) variants
  (measure_variants.py) does show a reproducible tag-dependent htrace difference, but it is a
  prefetcher artifact, not the leak: the mismatch has HIGHER residency (wrong polarity for a real
  tag-check, which fires on the match), and the diverging sets are stride-16 forward streams
  (12/28/44/60, 13/29/45/61), i.e. the Pixel prefetcher reacting to the tag fault. The fuzzer's
  priming normalises this away, which is why it filtered every test case.
- An all-16-tags sweep over the sealed sites (co-sweeping the re-converge ADDG so the pointer ends at
  the cell tag) is the decisive polarity/tag-check test. Result: the sites in the architectural
  prologue/epilogue (offsets 0xc, 0x5c, 0x84, 0x204) show a sharp tag-check signature (absent at the
  match tag AND at the TCMA-exempt tag, present at checked mismatches) = an architectural tag fault,
  not a speculative leak. Every site in the speculative shadow (the core store-to-load forwarding) is
  flat across all 16 tags. So the executor's tag-check measurement works and respects TCMA, but the
  speculative v2 forwarding does not surface as an L1D divergence.

Interpretation: the SSB environment (SSBS) and the sealer both support v2, and Revizor's measurement is
not blind to the structure. What is missing is the forwarding-timing window: the official PoC opens it
with explicit per-call branch training (511 warmup iterations) plus a slow-load-timed store-to-load,
which Revizor's batch harness does not reproduce. Catching genuine v2 needs a template/harness that
reproduces that window (and/or a contract clause modelling tag-gated forwarding so the fast path drives
it efficiently).

### 4.7 Variable-load and no-planted-load templates (X3, new)

Two follow-up templates asked whether the fuzzer finds TikTag-v1 when the number of loads in the
speculative shadow is not fixed, and when no load is planted in the speculative path at all.

- `mte_tiktag_ldvar.py`: the same late-resolving SDIV window and `CBNZ`, and in the shadow a hole of
  kind LOAD with a count drawn from one to sixteen per test case, so the tag-checked load count varies.
- `mte_tiktag_noload.py` (and the hand-written `user_mte_skeleton_noload.asm`): the same window, but the
  shadow body is a single hole of kind ANY with no load planted. A load appears in the speculative window
  only if the generator emits one at its ordinary memory-access density. The question is whether the
  fuzzer discovers the tag-check leak unaided once the window is scaffolded for it.

Config `config_mte_userskel_ladder.yml`: cond contract, L1D observation, Flush and Reload, pinned core 8
(X3), pre-run flush on (view_rotation), remote over the adb tunnel, with a laddered
`executor_sample_sizes` of 100, 500, 2000 so clean inputs resolve cheaply and only the effective cases
climb to the full sample.

**Result: the no-planted-load template surfaced a genuine leak.** `user_mte_skeleton_noload.asm` produced
five violations on X3 with no load written into the speculative path. The representative violation
`violation-261104-160250` (suspect input #12) was triaged to GENUINE. The tag-checked load was placed by
the generator inside the mispredicted window, not by hand.

Triage of `violation-261104-160250` (no planted load):

| Gate | Result |
|---|---|
| Tag-only diff | true (ADDG tag words only) |
| CE speculative-only | diverging set 44 is a speculative demand with base x2 (spec-only CE sets 8, 24, 36, 44) |
| Pointer-tag sweep (2000 reps) | set 44 match delta 6 at 0.85, every checked mismatch at 0.00, the TCMA-exempt delta 15 at 0.88 |
| PMU differential (2000 reps, prefix 14, 3 repeats) | see below |

| Variant | set 44 residency | INST_RETIRED | EXC_TAKEN | MEM_ACCESS |
|---|---|---|---|---|
| baseline (match) | 0.92, 0.91, 0.93 | 69 | 0 | 7.1 |
| decoy0 (mismatch) | 0.03, 0.01, 0.04 | 69 | 0 | 5.0 |
| decoy1 (mismatch) | 0.02, 0.25, 0.44 | 69 | 0 | 5.0 |
| decoy2 (mismatch) | 0.02, 0.04, 0.01 | 69 | 0 | 5.0 |

INST_RETIRED is flat at 69 and EXC_TAKEN is zero on every variant, so the mismatch causes no retiring
fault and no exception, which excludes an architecturally-retiring tagged access. MEM_ACCESS is 7.1 on
the match and 5.0 on every decoy, so the mismatch suppresses about two speculative accesses, the
tag-checked load and its dependent. The set-44 shrink is consistent on the decoys, with one decoy1 repeat
drifting to a wide multi-set block (0.25 to 0.44), a single contaminated measurement. Verdict: a genuine
speculative MTE tag-check leak (TikTag-v1) found with no load planted in the speculative path.

**Comparison with the planted-load template.** The hand-planted `user_mte_skeleton.asm` under the same
laddered config produced `violation-261104-153047` (suspect input #3). It passes the first three gates
(set 4, base x0, match delta 6 at 1.00 with every checked mismatch at most 0.01) and it clears the
architectural-fault exclusion (INST_RETIRED flat at 77, EXC_TAKEN zero). It does not pass the PMU
differential as a clean leak: MEM_ACCESS is flat at 12.0 on the match and on every decoy, so there is no
access suppression, and the set-4 and set-5 divergence does not hold across the counterexample decoys
(decoy1 sits at 1.00 on all three repeats, fully present on the mismatch, while decoy0 and decoy2 suppress
on some repeats and blow out to wide multi-set blocks on others). Verdict: QUESTIONABLE.

The cleaner genuine result is the no-planted-load one. The harder template, in which the generator had to
place the transmit load in the speculative window on its own, produced the more consistent leak for this
seed than the hand-planted template did.

Open item: gate 5 (memory-tag tracking) is not yet sealed on `violation-261104-160250`. The guess-granule
finder needs a fix before the tracking control is valid, so the decisive separation here rests on the PMU
differential (the MEM_ACCESS suppression and the flat INST_RETIRED with EXC_TAKEN zero).

Artifacts: `eval_out/noload_ldvar_20261104/` holds both violations' `generated.asm`, `AUTO_TRIAGE.md`,
`pmu_retirement.log` and `report.txt`. Templates `templates/mte_tiktag_ldvar.py`,
`templates/mte_tiktag_noload.py`, skeleton `user_mte_skeleton_noload.asm`, config
`config_mte_userskel_ladder.yml`.

---

## 5. Artifacts

- Violation directory: `eval_out/tfuzz_usertmpl_20261026_134845/violation-261026-134933/`
  (contains `generated.asm`, the enacted test cases, the inputs, `report.txt`, `ROOTCAUSE.md`, and
  the control outputs `auto_triage.stdout`, `memtag_track.stdout`, `pmu_variant.stdout`).
- Template: `user_mte_skeleton.asm`. Config: `s1_mte_hw.yml` (X3), `s1_mte_hw_a715.yml` (A715).
- TTD runs: `eval_out/ttd_x3/` and `eval_out/ttd_a715/`, each with `results.tsv` and per-run logs.
- Device session log: `eval_out/DEVICE_SESSION_tfuzz_20261026_133229.log`.
