# General precondition-gated leak detection — running log

Append-only. Each experiment: hypothesis, plan, exact commands BEFORE; verbatim command, verbatim key
output, interpretation AFTER. Negatives are recorded in full, never summarized away.

Device: Pixel 8 Cortex-X3 = core 8, adb 127.0.0.1:5037, serial 47061FDJH001K3.
venv: /home/gal_k_1_1998/revizor/revizor-venv/bin/python
Sandbox base this module load: 0xffffff8873821000 (re-read live each run).

All new work lives under docs/tiktag_mte_ni/general_detection/. No existing repo file is edited.

---

## Phase 0 — code study of the Step-1 pipeline (before any run)

Goal: know exactly what the NI fuzzer + MTE sealer require of a template, so Step 1 is designed to the
real constraints rather than to the design doc's assumption.

Pipeline for `tfuzz -t X.asm -c <ni-config>`:
- cli.py: a non-`.py` template -> `start_from_template` -> `generator.create_test_case_from_template`
  -> `asm_parser.parse_file` (+ `expand_template` for `.random_instructions` macros only) -> passes.
- NI fuzzer (`Aarch64NoninterferenceFuzzer`): `_boost_inputs` keeps only inputs with
  `executor.has_decoy(inp)`, then compares ONE input's genuine baseline vs its own decoys (same arch
  input, seal is the only thing that differs). A class is keyed by (ctrace, sealing class, input identity).
- MTE sealer (`src/aarch64/seal/sealer.py`): wraps each tagged-pointer access with an ADDG pair; the
  genuine variant uses ADDG #0 (matching tag), a decoy uses ADDG #delta (mismatched tag) then re-converges.

### The decisive eligibility rule (read from source, `_resolve_mte` + `_Resolved`)

`_Resolved.speculative = (spec_nesting != 0)`. A decoy may perturb a slot ONLY when it is speculative,
i.e. `spec_nesting != 0`. `_resolve_mte` computes `spec_nesting` as the **minimum** speculation nesting
over ALL reaches of the access in the CE trace:

```
for ite in cer:
    if has_sandbox_memory_access and off == access_off:
        if spec is None or nest < spec: spec = int(nest)   # MIN over reaches
        if nest == 0: arch_cells.add(cell)
```

So if the tag-checked load is reached architecturally (nest 0) even once, `spec_nesting == 0`,
`speculative == False`, the slot is NOT decoy-eligible, every decoy == genuine (null decoy), `has_decoy`
is False, and the NI fuzzer drops the input. **The gated load must be reached ONLY speculatively (or never)
in the CE trace to be sealer-eligible.**

### The structural tension this creates (hypothesis to validate, not yet a wall)

The manual PoC (Step 0, verified genuine) trains the window branch IN PLACE: training iterations fall
through and execute the body ARCHITECTURALLY (that is what trains the predictor not-taken); the attack
iteration mispredicts into the same body speculatively. So the body's load is reached at nest 0 during
training. Under the sealer that load would be classified architectural -> not eligible.

Moreover the manual PoC's match/mismatch asymmetry lives in the POINTER (csel picks matching x1 on
training, swept x5 on attack) — one static load, different dynamic tag. The static-slot sealer cannot
express "same load, different tag per dynamic instance"; it perturbs a static slot uniformly.

General form of the tension: for an in-place-trained conditional branch, the HW-speculated path equals
the architectural-majority path (what training executes), which the CE sees at nest 0. The only way to
make the gated load's path never-architectural is to speculate a direction the branch never architecturally
takes — i.e. branch mistraining, which is deprecated and yields 0 violations on this core. The v4
store-bypass window reaches a spec-only load (eligible) but produces NO tag-dependent shrink (verified
negative). So eligibility and the HW shrink are, a priori, satisfied by disjoint window types.

This is the central risk the task flagged ("validate early whether the asymmetry is resolved"). I will
NOT assume it either way; the experiments below measure it directly. If the naive loop is ineligible I
will search structures that break the tension (decoupled two-load, return-misprediction window via the
uncommitted rsb clause, self-establishing-precondition CT-SEQ path) and report honestly which, if any,
yields a fuzzer-reported violation.

### Established constraints reused (not re-derived)

- Branch mistraining deprecated, 0 violations on X3. Window must come from a genuinely trained branch.
- Cross-input/batch training does not transfer; only an INTERNAL loop trains pollution-free.
- v4 window opens but gives no tag-dependent shrink (known negative).
- asm_parser accepts a backward-branch loop, rejects `DC CIVAC`; use an SDIV compute-delay window.
- reload_timer_shift = -1 => htrace bit = L1 residency; set N at printed-string index 63-N.

### Coordinator levers recorded (to apply once eligibility is cleared)

1. SDIV window may be narrower than the DC-CIVAC flush window: if so, widen hard (longer/deeper
   dependent SDIV chains, more iterations, tune ldnum/gap, pointer-chase slow load, vary
   reload_timer_shift, raise reps). Physics is proven by the manual PoC; keep tuning.
2. Analyser boundary may be too coarse for a ~0.6-0.8 probabilistic signal: lower
   CONF.analyser_outliers_threshold; cross-input path uses MergedBitmapAnalyser.merged_bitmap +
   ChiSquaredAnalyser; NI uses analyser.filter_violations. Raise reps/executor_sample_sizes so the
   per-set divergence is statistically resolvable. A finer boundary must NOT become a false-positive
   generator — confirm every candidate with the verify-mte-tag-leak tag-tracking control.

Note on ordering: `has_decoy` gates measurement BEFORE the analyser sees anything. If the gated load is
not sealer-eligible, the input is dropped and no threshold matters. So eligibility is gate #1; analyser
tuning and window widening are gates #2-3, applied only once a decoy exists and reaches HW.

---

## Experiment 1 — eligibility of the manual-style internal-loop gadget (device-free CE probe)

HYPOTHESIS: In a manual-style internal-loop gadget (branch trained not-taken by training iterations that
fall through into the body), the body's tag-checked load is reached architecturally (nest 0) during
training, so `_resolve_mte` gives spec_nesting 0, the slot is not decoy-eligible, and `has_decoy` is
False. A control gadget whose branch is ALWAYS architecturally taken away from the body (body never arch)
should instead give spec_nesting > 0 and `has_decoy` True. This isolates the structural tension.

PLAN: Two ASM templates under general_detection/:
  - s1_mte_loop.asm  : manual-style, branch not-taken on training -> body arch during training.
  - s1_mte_neverarch.asm : branch ALWAYS arch-taken away from body -> body never arch (control).
Device-free probe probe_eligibility.py builds the real NI pipeline (get_fuzzer non-interference +
initialize_modules), parses each template, load_test_case, presets executor._sandbox_base to the live
base to avoid the device, generates 4 inputs, and prints per-MTE-site (value, alts, speculative,
spec_nesting), has_decoy, and the genuine-vs-decoy ADDG deltas.

COMMANDS:
  REVIZOR_LOG=0 python docs/tiktag_mte_ni/general_detection/probe_eligibility.py \
     -s base.json -c docs/tiktag_mte_ni/general_detection/s1_mte.yml \
     -t docs/tiktag_mte_ni/general_detection/s1_mte_loop.asm
  (same for s1_mte_neverarch.asm)

### Experiment 1 RESULT (DECISIVE NEGATIVE for the looped template)

Command (verbatim):
  python docs/tiktag_mte_ni/general_detection/probe_eligibility.py -s base.json \
    -c docs/tiktag_mte_ni/general_detection/s1_mte.yml \
    -t docs/tiktag_mte_ni/general_detection/s1_mte_loop.asm

Verbatim key output (both s1_mte_loop.asm AND s1_mte_neverarch.asm):
  File ".../src/aarch64/seal/sealer.py", line 1105, in seal
    sandbox, data_sites = self._walk.sandbox(tc)
  File ".../src/aarch64/seal/primitives.py", line 166, in _dfs
    raise GeneratorException("CFG contains a cycle; MTE taint dataflow assumes a DAG")
  src.interfaces.GeneratorException: CFG contains a cycle; MTE taint dataflow assumes a DAG

Confirmed by reading primitives.py `_topo_sort` (lines 151-175): the cycle check is UNCONDITIONAL (no
config flag, no loop-tolerant mode). grep for unroll/back_edge/cycle across generator + sealer: the ONLY
hit is that raise. There is NO loop unrolling anywhere in the generator/template pipeline.

INTERPRETATION (hard architectural fact):
  The MTE sealer's sandbox/taint walk requires a DAG CFG. A template with a back-edge loop throws at
  `load_test_case` -> `sealer.seal` -> `_walk.sandbox` -> `_topo_sort`, BEFORE eligibility is ever
  computed. The same `_walk.sandbox` + `_topo_sort` is shared by every sealer (MTE, PAC, Canon,
  BranchTarget), so the restriction is not MTE-specific — NO looped template can be sealed.

CONSEQUENCE for Step 1 as specified:
  The task's established facts say the ONLY pollution-free branch training on this core is an INTERNAL
  LOOP inside one test-case execution. But an internal loop is a cyclic CFG, which the sealer rejects.
  Therefore the literal Step 1 (internal-loop ASM template + MteSealing under the NI fuzzer) cannot be
  built without editing the sealer's DAG assumption — which is shipped source and OUT OF SCOPE (hard
  rule e). This is a real wall for the looped-template + sealer combination, not a tuning problem.

  It also explains the prior straight-line `mte_tiktag_v1_branch.py` 0-violations result from a second
  angle: its if-body was entered architecturally (x0 != 0 arch) -> min nesting 0 -> not decoy-eligible ->
  null decoy -> dropped by `_boost_inputs` (has_decoy False) -> never measured. Not "no training"; the
  body was simply ineligible.

### Levers still open (to exhaust before declaring the wall, per coordinator)

L1. DAG-shaped (loop-free) template where the ATTACK body is on the branch's ARCH-NOT-TAKEN path, so the
    body load is never arch (eligible), and HW speculates into it via default/cold misprediction of a
    first-seen forward branch (classic Spectre-v1 shape) — possibly with a BPU flush each run so the
    forward branch re-defaults not-taken. No loop needed. TEST eligibility (CE probe) then HW window.
L2. If cold/default misprediction is too weak without training, see whether the window can be widened by
    the SDIV chain enough that even a short/weak misprediction suffices, and sweep BPU-flush / reps /
    reload clock.
L3. Non-sealer paths (basic fuzzer + CT-SEQ, self-establishing precondition) — but the regular aarch64
    executor ALSO seals when MTE/PAC categories are present (factory.get_executor), so those paths ALSO
    hit the DAG wall whenever a loop is used. A loop-free self-establishing form would be needed.
L4. Return-misprediction (rsb) window (uncommitted infra): a DAG-compatible control-flow window that
    reaches a spec-only target — candidate for Step 2 if L1 fails.

Proceeding with L1: loop-free DAG templates + CE eligibility probe.

## Experiment 2 — DAG neverarch eligibility CONFIRMED (device-free CE probe)

HYPOTHESIS: A loop-free DAG template whose tag-checked body sits on the branch's ARCH-NOT-TAKEN path
(branch forced arch-taken via `ORR x9,x9,#1; CBNZ x9,.skip`) makes the body load reached only
speculatively (nest>0) -> sealer-eligible -> has_decoy True and a real tag-divergent decoy. A control
whose body is arch (CBZ, branch arch-not-taken) is ineligible.

COMMAND (verbatim):
  python docs/tiktag_mte_ni/general_detection/probe_eligibility.py -s base.json \
    -c docs/tiktag_mte_ni/general_detection/s1_mte.yml \
    -t docs/tiktag_mte_ni/general_detection/s1_dag_neverarch.asm --n 2

VERBATIM key output:
  s1_dag_neverarch:  has_decoy=True;  site[2] MteSealing value=7 speculative=True;
                     site[3] MteSealing value=7 speculative=True
     GENUINE relocs: [(56,0x91801ca5),(64,0x918000a5),(108,0x91801d4a),(116,0x9180014a)]
     DECOY   relocs: [(56,0x91803ca5),(64,0x918020a5),(108,0x9180194a),(116,0x9180054a)]
  s1_dag_bodyarch (control): has_decoy=False; both sites speculative=False; decoy==genuine

INTERPRETATION:
  - The DAG constraint is satisfied (no cycle crash): a loop-free forward-branch template seals fine.
  - Eligibility works exactly as the source predicted: body-on-arch-not-taken -> speculative=True ->
    eligible; body-arch -> speculative=False -> ineligible (null decoy). The control confirms the
    mechanism is the arch-reachability, not an artifact.
  - The decoy relocation differs from genuine at the retag ADDG (site 2 before-slot 0x91801ca5 ->
    0x91803ca5 and its re-converge 0x918000a5 -> 0x918020a5): the decoy variant runs the speculative
    guess load with a MISMATCHED tag and re-converges after, exactly the TikTag tag perturbation.
  - Caveat: BOTH body loads (guess site 2 AND probe site 3) are eligible MTE sites, and `_decoy_subset`
    perturbs a RANDOM non-empty subset. So a given decoy may mistag the guess load, the probe, or both.
    For a clean guess-load-shrink TikTag signal I want only the guess load mistagged; this is a
    readout-isolation concern to address after the HW window is confirmed.

OPEN CRUX (needs HW): the body is never architectural, so HW must SPECULATE into it via misprediction of
the arch-taken forward branch. With no internal loop (DAG constraint) there is no in-place training. So
the window must come from COLD/DEFAULT misprediction of a first-seen forward branch (static not-taken
prediction) — which is only sustained across measurement reps if the BPU is flushed each run (otherwise
the predictor learns the arch-taken direction after a few reps and stops speculating). Next: find the
BPU-flush knob and test the HW window.

## Finding — kernel per-input branch training (the no-loop window lever)

Reading bpu.c + measurement.c:
- `apply_input_branch_training(measurement_code, &current_input->input)` is called at measurement.c:348
  INSIDE the per-input measurement path, BEFORE the TC executes (line 384). So the branch is re-trained
  before EVERY execution of the TC, i.e. every rep mispredicts (not just the first). The window opens
  repeatedly -> the signal accumulates over reps.
- Training (`apply_branch_training`, bpu.c) writes a direction-forcing stub (cbz/cbnz x0, x0=&sandbox!=0)
  at the branch PC across ALL measurement views (aliased physical page), executes it MAX_MEASUREMENT_VIEWS
  times with flush_bpu_phr between, saturating the shared base-PHT entry, then restores the real branch.
  This is a genuine kernel-side training loop — it does not need an internal loop in the test case.
- The config is per-input `bpu_train[]`, populated by `_bpu_entries` -> `branch_mistraining_entries`,
  which for each ARCHITECTURAL conditional branch in the CE trace emits an entry training the OPPOSITE
  direction, so the first HW execution mispredicts. Gated by CONF.enable_branch_mistraining (default off,
  "WIP pending hardware efficacy confirmation").

Reconciliation with "mistraining deprecated, 0 violations on this core": that note may reflect gadgets
whose speculated body was ALSO architectural (ineligible -> null decoy -> dropped before measurement),
so the window never had an eligible decoy to reveal. My DAG neverarch gadget is different: the body is
eligible (never-arch) AND tag-gated. So enabling mistraining here is a legitimate, untested combination,
not a known-dead path. I TEST it on HW rather than assume either outcome.

## Experiment 3 — HW: does mistraining open a window at the arch-taken branch in s1_dag_neverarch?

HYPOTHESIS: With enable_branch_mistraining=true, the kernel saturates the window branch (arch-taken) to
not-taken before each run, so HW speculates the never-arch body; the MTE sealer's decoy (mismatched tag
on the guess load) shrinks the window so the probe set stays absent, while genuine (matching) leaves it
present -> an NI htrace divergence the contract (tag-blind cond) does not predict -> a fuzzer-reported
violation. NULL hypothesis to respect: mistraining may genuinely fail to open a usable window on X3
(the deprecated-note outcome), giving 0 violations. Report whichever occurs.

PLAN: config s1_mte_hw.yml = s1_mte.yml + enable_branch_mistraining:true, enable_phr_flush:false,
enable_pre_run_flush:true (match the manual PoC regime). Run tfuzz small first (-n 1 -i 4, sample
[200,1000]) with REVIZOR_LOG=1, inspect ni/ logs + violation verdict. Serialize device access.

## Experiment 3+4 RESULTS — window opens, but no tag-dependent shrink (yet)

### Exp3 (tfuzz, s1_dag_neverarch, mistraining on): 0 violations
Command: python revizor.py tfuzz -s base.json -t .../s1_dag_neverarch.asm -c .../s1_mte_hw.yml -n 1 -i 4
Stats: Test Cases 1, Inputs/tc 8.0, Violations 0, Effective Cls 4.0, Fast Path 1. Duration 1.3s.
basic/hw.log: every input `mispredicts: avg=1.00 reps=[1,1,...]`  -> the kernel mistraining DOES open a
misprediction at the arch-taken branch on EVERY rep (window lever works). (spec: retired=0 inst_spec=0
wrongpath=0.0 are unpopulated counters in this mode, not evidence of no speculation.)

### Exp4 (measure_variants, s2_tiktag_16_80_24, baseline vs decoy vs noncanon): NO divergence
Command: python .../measure_variants.py -s base.json -c .../s1_mte_hw.yml -t .../s2_tiktag_16_80_24.asm
         --n 3 --reps 3000
Verbatim:
  input 0: baseline/decoy0/noncanon all resident set 50 @1.0; DIVERGE []  []
  input 1: baseline/decoy0/noncanon all resident set 56 @1.0; DIVERGE []  []
  input 2: baseline/decoy0/noncanon all resident set 30 @1.0; DIVERGE []  []

INTERPRETATION:
  - The never-arch body IS reached speculatively: the probe set (input-dependent: 50/56/30) is resident
    @1.0, reached only on the mispredicted path. So the window opens and reaches the probe reliably.
  - But mistagging EVERY speculative guess load (noncanon, forced-all-bad on all 16 loads + probe)
    leaves the probe set resident @1.0 — IDENTICAL to baseline. So in this template form a speculative
    MTE tag mismatch produces NO window shrink and no observable divergence.
  - Contrast with the manual PoC: match residency ~0.62 (MARGINAL window, probe at the edge), mismatch
    0.0. My template's window is WIDE (probe @1.0, well inside the window), so even if a tag fault
    shortened the window slightly, the probe would still issue. The manual PoC operates at the window
    EDGE where a small shrink tips present->absent.
  - (Side note: the arch STR xzr/LDR warm did not zero the body's x12 -> probe set is input-dependent,
    not fixed 32. Cosmetic; consistent within a class, does not affect the shrink question.)

NEXT (coordinator lever #1): push the probe to the window EDGE by lengthening the gate / shortening the
window, and test baseline vs noncanon AT the edge. If noncanon falls off the edge earlier than baseline,
the tag-fault shrink is real and I tune the fuzzer to that operating point. If baseline and noncanon
track together across the whole edge, the tag fault has no speculative effect here (honest negative).

## Experiment 5 — THE SHRINK REPRODUCES at the window edge (fixed probe set)

Fix vs Exp4: the probe now lands at a FIXED set 32 (gate `AND x11,x12,xzr; ORR x10,x10,x11` keeps the
probe address at 0x800 while staying data-dependent on the guess result x12). Earlier "hottest set @1.0"
was the GUESS line (allocates regardless of tag); the probe is a different, now-fixed set.

Command: python .../sweep_s2.py -s base.json -c .../s1_mte_hw.yml --gaps 20,80,200,400 --sdiv 24 --n 2 --reps 1500
Verbatim (probe=set32, guess=hottest-other; baseline -> noncanon):
  gap=  20: in0: probe32 0.68->0.05  guess0 1.00->1.00   in1: probe32 0.57->0.00  guess15 1.00->1.00
  gap=  80: in0: probe32 0.04->0.04  guess0 1.00->1.00   in1: probe32 0.02->0.02  guess15 1.00->1.00
  gap= 200: in0: probe32 0.05->0.05  guess0 1.00->1.00   in1: probe32 0.03->0.03  ...
  gap= 400: in0: probe32 0.00->0.00  ...

INTERPRETATION (POSITIVE, TikTag-v1 signature):
  - At gap=20 the probe (set 32) is PRESENT on baseline (matching tag) at 0.57-0.68 and ABSENT on
    noncanon (all guess loads mistagged) at 0.00-0.05. The guess line stays 1.0 in both (the checked
    load's own line allocates regardless of tag; the leak is the DOWNSTREAM probe) — exactly TikTag-v1.
  - The baseline ~0.6 matches the manual PoC's marginal match residency (0.62): the window is at its
    EDGE at gap=20. At gap>=80 the probe falls off the window in BOTH variants (baseline ~0.04), so the
    operating point is a SHORT gate (~16-24). The misprediction window here is short (~tens of cycles),
    but it is enough for the tag-check differential.
  - So the TikTag tag-check shrink DOES reproduce in a loop-free DAG NI template with kernel mistraining.
    Exp4's null was a measurement artifact (tracking the guess line, wide-gate off-window), not a real
    absence of the effect.

RECHECK PLAN (hard rules c,d): (1) finer gap sweep 8..40 to pick the best separation; (2) confirm the
RANDOM decoy0 (not just forced noncanon) diverges; (3) repeat/re-seed for stability; (4) run the real
tfuzz and see a reported violation; (5) verify-mte-tag-leak tag-tracking control for genuineness.

## Experiment 6 — ldnum=1 makes the RANDOM decoy shrink (fuzzer-triggerable)

Problem found at ldnum=16: the RANDOM decoy0 (~0.5) tracked baseline (~0.5); only the forced noncanon
(all 16 mistagged) shrank. A random subset of 16 loads leaves the gate-feeding load matching too often.

Command: python .../sweep_s2.py ... --ldnum 1 --gaps 4,8,16,24 --sdiv 24 --n 3 --reps 2000  (and ldnum=4)
Verbatim (probe=set32; base / random-decoy0 / noncanon):
  ldnum=1 gap=4 : in0 0.54/0.03/0.04  in1 0.52/0.02/0.02  in2 0.53/0.03/0.03
  ldnum=1 gap=8 : in0 0.56/0.02/0.02  in1 0.50/0.01/0.01  in2 0.52/0.02/0.02
  ldnum=1 gap=16: in0 0.55/0.05/0.05  in1 0.52/0.01/0.01  in2 0.49/0.00/0.00
  ldnum=1 gap=24: in0 0.52/0.01/0.01  in1 0.54/0.01/0.01  in2 0.51/0.01/0.01
  ldnum=4 gap=8 : in0 0.55/0.55/0.07  in1 0.50/0.47/0.02  in2 0.53/0.43/0.02  (decoy tracks base: NO)

INTERPRETATION:
  - With ldnum=1 the single guess load feeds the gate, and the random decoy (deterministic per input via
    stable_rng) perturbs it, so the probe (set 32) collapses from ~0.5 (baseline, matching tag) to ~0.02
    (decoy, mismatched tag) for EVERY input across gaps 4-24. The guess line stays resident regardless.
  - This is the fuzzer-triggerable TikTag-v1 signal: baseline vs its OWN random decoy diverge ~0.5 at a
    fixed set, with a tag-blind contract -> an NI violation. Stable across 3 inputs; baseline ~0.5
    matches the manual PoC's marginal window.
  - ldnum>=4 dilutes the random decoy (gate-feeding load often left matching) -> use ldnum=1.

WINNING STEP-1 CONFIG: ldnum=1, gap=8, sdiv=24, kernel branch-mistraining on, probe fixed set 32.
Next: run the real tfuzz and confirm a REPORTED violation; then recheck stability + verify genuineness.

## Experiment 7 — FUZZER REPORTS THE VIOLATION (Step 1 win condition met)

Command (verbatim):
  REVIZOR_LOG=1 python revizor.py tfuzz -s base.json -t .../s1_tiktag_win.asm \
    -c .../s1_mte_hw.yml -n 1 -i 10 -w .../eval_out/win1
Verbatim stats:
  Test Cases: 1   Inputs per test case: 20.0   Violations: 1   Effective Cls: 10.0   Fast Path: 0
  Hardware Tracing Errors: 0   Duration: 12.9
  Violation saved: .../eval_out/win1/violation-261022-141431/

The non-interference fuzzer, given the loop-free DAG TikTag-v1 ASM template + MteSealing + kernel branch
mistraining, REPORTS a non-interference violation: a baseline (matching-tag) vs its own decoy
(mismatched-tag) htrace divergence at the probe set that the tag-blind `cond` contract does not predict.
The trace log confirms the gadget assembled as designed (arch retag addg #7 + str + warm ldr; 24-SDIV
window; arch-taken cbnz; speculative body with the sealer's addg retag + guess ldr + AND-xzr gate + probe).

This is the first fuzzer-reported reproduction of the TikTag-v1 (precondition-gated speculative) leak
through Revizor's template/NI pipeline. Pending: genuineness (tag-sweep) + stability recheck.

## Experiment 8 — GENUINENESS: tag-sweep tracking control (verify-mte-tag-leak)

Command: python .../tag_sweep.py -s base.json -c .../s1_mte_hw.yml -t .../s1_tiktag_win.asm
         --retag-off 0x90 --reps 1500 --n 1
Verbatim (probe set 32 residency by speculative-guess retag delta 0..15):
  0:0.73  1:0.03  2:0.07  3:0.08  4:0.01  5:0.07  6:0.01  7:0.04  8:0.01
  9:0.63  10:0.02 11:0.04 12:0.03 13:0.03 14:0.02 15:0.05

INTERPRETATION (GENUINE):
  - Probe present only at delta 0 (identity = MATCHING tag, baseline) and delta 9; absent (0.01-0.08)
    at every other delta. delta 9 is a TCMA-unchecked tag (0b0000/0b1111 are match-all under TCR.TCMA1,
    never tag-checked, so the guess load is not gated -> probe always issues), exactly as the manual-PoC
    verification flagged tag 15.
  - The signal TRACKS the tag relationship: high at match + unchecked, low at all 14-ish checked
    mismatches. Uniform noise or a non-tag artifact would fire at ~0.5 for every delta. This sharp
    structure is the decisive genuineness evidence: the leak is the MTE tag CHECK gating the speculative
    window, reproduced through the fuzzer's sealer.
  - Honest caveat: the two TCMA-unchecked tags (0 and 15) would predict TWO adjacent high deltas; only
    delta 9 is high (its neighbor is low). Expected is one match + up to two unchecked = up to three
    present; we see two. The sharp tag-dependence conclusion is unaffected (14 checked mismatches all
    suppressed), but the exact unchecked-delta count is not fully pinned and is noted as such.

## Experiment 9 — CONTROL A (mistraining OFF) + stability

mistraining OFF (s1_mte_hw_noMT.yml), measure_variants on s1_tiktag_win.asm:
  input 0: baseline set32=0.69, decoy0=0.74, noncanon=0.03  (DIVERGE base vs noncanon: set32 0.69->0.03)
  input 1: baseline set32=0.48, decoy0=0.43, noncanon=0.01  (DIVERGE base vs noncanon: set32 0.48->0.01)

INTERPRETATION:
  - The probe is STILL present (~0.5-0.7) and noncanon STILL shrinks it (->0.01-0.03) even with
    mistraining OFF. So the misprediction window opens from the NATURAL cold/late-resolve behaviour of
    the arch-taken forward branch (SDIV chain resolves the branch late; the forward branch's cold
    prediction fall-through speculates the body), not only from the kernel mistraining. The leak is
    therefore ROBUST / not mistraining-dependent (good for generality).
  - decoy0 did NOT shrink here (0.74/0.43) while noncanon did: the random decoy subset is salt-dependent
    (per-TC random salt), so for ldnum=1 the single guess slot is perturbed ~2/3 of the time. The forced
    noncanon always perturbs it. Over many inputs the fuzzer still hits shrinking decoys (exp7 did).
  - No architectural fault/crash in any run: the mistagged access is speculative-only (an architectural
    tag mismatch would fault at EL1 and panic), which inherently confirms window-dependence.

GENUINENESS VERDICT: CONFIRMED genuine TikTag-v1 tag-check leak, by (1) the tag-sweep tracking control
(present only at match + TCMA-unchecked, absent at all checked mismatches), (2) noncanon shrinks the
downstream probe while the checked-load's own line stays resident (TikTag-v1 downstream signature),
(3) speculative-only (no arch fault).

## Experiment 10 — Step-1 reliability hardening (in progress)

Re-run #2 of the plain winning config found 0 violations (Fast Path 1): the fuzzer REPORT is run-to-run
probabilistic even though the leak reproduces every time. Causes: (a) random decoy subset perturbs the
single guess slot only ~50% of the time (ldnum=1, 2 eligible slots); (b) input-dependent SDIV window;
(c) marginal baseline (~0.5). Default analyser is already chi2 (probabilistic).

Hardening applied: fixed-divisor (input-independent) SDIV window (gen_s2 --det-window), inputs_per_class=4
(more decoys/class -> P(some decoy perturbs the guess) ~ 1-0.5^3 per input), executor_sample_sizes [2000].
Templates/configs: s1_tiktag_hard.asm + s1_mte_hw_hard.yml ; canon: s2_canon_hard.asm + s2_canon_hw_hard.yml.
Measuring the fuzzer hit-rate over 4 runs (eval_out/hits.sh). Result recorded next.

## Experiment 10 RESULT — Step-1 fuzzer hit-rate after hardening: 3/4

Command: eval_out/hits.sh (4x tfuzz, fresh seed, s1_tiktag_hard.asm + s1_mte_hw_hard.yml, -i 8)
Verbatim: run1 Violations=1  run2 Violations=1  run3 Violations=0  run4 Violations=1

INTERPRETATION: ~75% per-run report rate with inputs_per_class=4 + det window + 2000 reps (up from the
flaky plain config). The underlying leak is 100% reproducible (measure_variants baseline~0.5 vs
noncanon~0.02 and the tag-sweep every time); the residual report flakiness is the marginal window
(~0.5 baseline, probabilistic per rep) interacting with the chi2 fast-path sample on some input sets.
Further levers (not all exhausted): higher inputs_per_class, more inputs, lower analyser_stat_threshold.
Reported honestly as 3/4 rather than claimed as reliable.

## Experiment 11 — STEP 2: canonicality precondition on HW

HYPOTHESIS: the same DAG-neverarch skeleton with CanonSealing (enable_canonicality, no MTE) shrinks the
probe when the speculative guess load's pointer is made non-canonical -> generalizes the detector to a
SECOND precondition. PLAN: measure_variants baseline/decoy/noncanon on s2_canon_hard.asm, then tfuzz.

## Experiment 11 RESULT — STEP 2 canonicality: CLEAN divergence (generalizes)

Command: python .../measure_variants.py -s base.json -c .../s2_canon_hw_hard.yml -t .../s2_canon_hard.asm --n 3 --reps 2000
Verbatim:
  input 0: baseline set19=1.0; decoy0=0.0; decoy2=0.0; noncanon=0.0   (DIVERGE set19 1.0->0.0)
  input 1: baseline set62=1.0; decoy0=0.03; noncanon=0.03             (DIVERGE set62 1.0->0.03)
  input 2: baseline set39=1.0; decoy0=0.0
  (decoy1 did not diverge: its random subset perturbed the probe, not the guess load)

INTERPRETATION:
  - Canonicality GENERALIZES the detector and gives a CLEANER, near-binary signal: baseline (canonical
    pointer) 1.0 -> decoy (non-canonical) 0.0. The RANDOM decoy0 shrinks too (fuzzer-triggerable).
  - MECHANISM DIFFERS from MTE: here the divergence is at the GUESS LOAD's OWN line (set 19/62/39), not a
    downstream probe. A speculative non-canonical address is squashed before it allocates its line, so
    the checked load's own set shrinks. (MTE: the tag check is post-allocation, so the checked line
    stays and only the downstream probe shrinks.) Both are precondition-gated speculative cache leaks.
  - No arch fault/crash -> the non-canonical access is speculative-only (genuine).

## Experiment 12 — STEP 2 canon FUZZER: VIOLATION REPORTED

Command: tfuzz -s base.json -t .../s2_canon_hard.asm -c .../s2_canon_hw_hard.yml -n 1 -i 8
Stats: Violations: 1   Effective Cls: 8.0   Fast Path: 0   (violation dir saved)

The NI fuzzer reports a canonicality-gated speculative violation. Detector now covers TWO preconditions
through the fuzzer: MTE tag-match (Step 1) and address canonicality (Step 2). Step 2 ACHIEVED.

## Experiment 13 — STEP 3: fuzz the body in a fixed skeleton (canonicality)

Template s3_canon_randbody.asm: fixed det-SDIV window + arch-taken CBNZ; the speculative body is a
`.macro.random_instructions.32.8.0` block the FUZZER fills (32 instrs, 8 mem). CanonSealing seals the
fuzzer-generated speculative loads.

Eligibility probe: has_decoy=True with MANY eligible CanonSealing slots (sites 8..16+, speculative=True).
measure_variants (baseline vs decoy0/1/2 vs noncanon), probe = hottest set:
  input 0: baseline set31=1.0; ALL decoys + noncanon = 0.04   (DIVERGE set31 1.0->0.04)
  input 1: baseline set10=1.0; ALL decoys + noncanon = 0.00   (DIVERGE set10 1.0->0.00)
  input 2: baseline set26=1.0
INTERPRETATION: the fuzzer-GENERATED body still yields a clean canonicality-gated divergence, and ALL
random decoys diverge (many sealed loads -> the random subset almost always perturbs an observed one).
Robust, not marginal. The fuzzer now contributes STRUCTURE, not just the precondition value.

## Experiment 13 RESULT — STEP 3 canon FUZZER: VIOLATION REPORTED
Command: tfuzz -t s3_canon_randbody.asm -c s2_canon_hw_hard.yml -n 3 -i 8
Stats: Test Cases 1, Violations: 1, Effective Cls 8.0, Fast Path 0 (stopped at first violation).
Step 3 ACHIEVED (canonicality): the fuzzer GENERATES the speculative body and the NI fuzzer still
reports the precondition-gated violation.

## Experiment 14 — STEP 4: fully random generation (no template), canonicality NI

HYPOTHESIS: the fully-random NI generator, with canonicality sealing + the natural forward-branch
misprediction window, produces precondition-gated speculative canon leaks without any template. A random
non-canonical speculative access that leaks its own line IS a precondition-gated leak by our definition.
PLAN: revizor.py fuzz (random, no -t) with fuzzer=non-interference + enable_canonicality, bounded -n.

## Experiment 14 RESULT — STEP 4 fully random (canonicality NI): rare but present

Command: revizor.py fuzz -s base.json -c s4_canon_random.yml -n 40 -i 8 --nonstop  (no template)
Stats: Test Cases 40, Violations: 1, Effective Cls 6.8, Fast Path 32, Duration 216s. 1 violation dir.

INTERPRETATION:
  - The fully-random NI generator (no template) finds a precondition-gated canonicality violation, but
    RARELY (1/40 random programs). The random generator does not bias toward the needed structure
    (arch-not-taken forward branch with a long resolve + a speculative sealed access on the mispredicted
    path), so most programs (32 fast-path discards) show nothing; occasionally one arises.
  - By NI construction the comparison is baseline vs its OWN decoy (only the canonicality seal differs),
    so the divergence is canonicality-seal-driven, i.e. a precondition-gated speculative leak, not a
    data-dependent Spectre-v1 artifact. (Not individually tag-swept; the self-input NI design guarantees
    the seal is the only variable.)
  - Honest status: Step 4 is PARTIALLY achieved — the capability exists end-to-end but at low yield; the
    rate is the open gap, matching the design doc's "largest gap" note. A structure-biased generator
    (emit an arch-taken forward branch + speculative sealed access) would raise the yield; that is the
    single most promising next step.

## SUMMARY OF LADDER
  Step 0: verified manual PoC (prior).                              GROUND TRUTH
  Step 1: MTE TikTag-v1 as NI ASM template -> fuzzer violation.     ACHIEVED + genuine (tag-sweep)
  Step 2: swap precondition -> canonicality -> fuzzer violation.    ACHIEVED (cleaner binary signal)
  Step 3: fuzz the body in a fixed skeleton (canon) -> violation.   ACHIEVED
  Step 4: fully random (canon NI) -> violation 1/40.                PARTIAL (rare, seal-driven)

## Experiment 15 — STEP 4 fully-random MTE (no template): 0/150

Command: revizor.py fuzz -s base.json -c s4_mte_random.yml -n 150 -i 8 --nonstop (MTE NI, mistraining on)
Stats: Test Cases 150, Violations: 0, Effective Cls 7.1, Fast Path 139, HW tracing errors 1, Duration 604s.
INTERPRETATION: pure-random generation does NOT surface TikTag/MTE in a bounded 150-program run (contrast
canon 1/40). Confirms MTE needs the downstream data-dependent access in the speculative window, which
random generation does not reliably emit. Honest negative for "complete random MTE".

## Experiment 16 — user skeleton: planted window + planted sealed load, random tail (does it detect?)

user_mte_skeleton.asm = random(before) / planted SDIV window / random / if(forced-taken) { random /
LDR x5 (sealed) / random } / random(after). Eligibility: has_decoy=True (sealed load + random loads are
speculative MteSealing slots). HW measure below tests whether the RANDOM tail after the sealed load
supplies a shrink-sensitive downstream access.

## Experiment 16/17 RESULT — random-body MTE does NOT reliably detect (timing-critical shrink)

user_mte_skeleton (planted window + planted sealed load + random tail):
  measure_variants (n=4): WEAK, WRONG-DIRECTION tag effects — baseline ~0.01 -> decoy/noncanon ~0.16-0.27
    at a few sets (decoy GAINS sets, not the shrink). Same sets under decoy and noncanon -> seal-driven,
    not noise, but NOT the TikTag shrink (which is baseline-present -> decoy-absent).
  FUZZER (tfuzz -n1 -i8): Violations: 0 (Fast Path 1). The weak/seed-dependent effect did not survive
    into a fuzzer run (fuzzer regenerates a different random body; chi2 did not flag it).
thin_mte_collector (generic collector tail folding x0..x5 -> probe LDR): measure_variants = NO resident
  sets at all (baseline included) -> the collector probe isn't reliably reached in the short window.

ROOT CAUSE (crisp): the MTE TikTag shrink is TIMING-CRITICAL. The differential appears only when the
downstream probe issues AFTER the tag check resolves but BEFORE the branch resolves — i.e. the probe must
sit at the MARGINAL window edge (Step 1 needed a tuned gap=8 data-dependent gate, baseline ~0.5). A random
tail / generic collector places the downstream access at an uncontrolled latency: too early (fills
regardless), beyond the short window (absent in both), or on an inconsistent set. Randomizing the
downstream destroys the exact timing the shrink needs.

CONTRAST with canonicality: a non-canonical speculative access fails to allocate its OWN line — binary,
no downstream, no timing — so it survives full randomization (random body AND no-template both detect).

CONCLUSION for "auto-detect TikTag with random body":
  - Fully-random MTE: 0/150. Random tail around a planted load: fuzzer 0. Generic collector: no signal.
  - The clean genuine TikTag shrink needs the PLANTED downstream dependent probe tuned to the window edge
    (Step 1). That one timing-critical piece cannot currently be left to random generation for MTE.
  - To push further: a STRUCTURE-AWARE generator that emits the sealed load and a data-dependent probe at
    a controlled small latency (not free-random) — i.e. constrain the latency, randomize the rest.

## Experiment 18 — Python object-template with constrained holes (user suggestion)

Built mte_tiktag_pytmpl.py: planted TikTag chain (warm/zero, deep SDIV window, forced branch, sealed
guess load, data-dependent gate, fixed-set probe) + Kind.ALU holes constrained via regs= to EXCLUDE the
chain registers (so random padding can't clobber the chain). Confirmed: holes support Kind + regs (reg
exclusion) — strictly more control than the ASM .random_instructions macro.

Results (fuzzer tfuzz and direct measure_variants, offsets pinned via Mem(X,offset=Imm(0))):
  - fuzzer: 0 violations (n=3 and n=5 fills, hardened config).
  - measure: NO divergence; probe set 32 NEVER resident (resident set is the input-dependent guess line
    from the arch warm). Holds with mistraining ON and OFF.

ROOT CAUSE (structural Builder limitation, from the dumped asm):
  - The Builder's `if_` puts the speculative body on the branch's TAKEN block. A cold forward branch
    predicts NOT-taken, so a taken-path body is speculated only if the branch is mistrained-taken.
  - The kernel mistraining stub is cbz/cbnz x0 (PC-indexed base-PHT). My working ASM gadget's branch is
    CBNZ with the body on the FALL-THROUGH (not-taken) path, so (a) a cold forward CBNZ naturally
    mispredicts into the fall-through body (no mistraining needed), and (b) when mistrained, the stub
    type matches. The Python if_ emits a flags-based b.eq with the body on the TAKEN path -> no cold
    misprediction into it, and mistraining here did not open a usable window -> the body never
    speculated -> probe never reached -> no signal.
  - Also found: Builder Mem(X5) defaults to a RANDOM offset; must pin Mem(X5, offset=Imm(0)) or the
    warm/zero/guess/probe land on different lines (fixed, but did not change the no-window outcome).

CONCLUSION: Python holes DO give constrained random fill (Kind + regs), which the ASM macro lacks, so
they are the right tool for RANDOM PADDING around a correctly-structured MTE chain. BUT the Builder's
`if_` cannot express a not-taken-body (fall-through) conditional, which is the branch shape the reliable
cold-misprediction MTE window needs. Making the Python-hole path work for MTE would need a new Builder
primitive for a not-taken-body branch (shipped-source change in src/aarch64/template/builder.py) — OUT OF
SCOPE per the no-edit-shipped-source rule; reported rather than done. The working MTE detector remains the
fully-planted ASM gadget; the ASM macro in turn can't protect the chain registers, so neither path yet
gives a reliable random-body MTE detector.

## Experiment 19 — detector works WITHOUT mistraining (deprecated); v4 window next

Correction (user): branch mistraining is deprecated. Re-ran the ASM MTE gadget fuzzer with
enable_branch_mistraining=FALSE (s1_mte_hw_hard_noMT.yml), natural cold forward-branch misprediction only:
  run1 Violations=1  run2 Violations=1  run3 Violations=1   (3/3 — MORE reliable than mistraining-on 3/4)
So the TikTag detector does NOT depend on mistraining; the window is the cold misprediction of the
arch-taken forward branch (body on the fall-through path). Mistraining is dropped from the configs.
The earlier Python if_ failure is now fully explained: its body is on the TAKEN path, which has no cold
misprediction and would need the deprecated mistraining -> dead end. Fall-through-body is required.

## Experiment 20 — v4 (store-bypass) window with the sealed load placed tight (user request)

Reference template templates/mte_tiktag_v4.py: v4 store-bypass window + a branch on the bypassed stale
value (keeps the body arch-not-taken -> eligible). Prior established result: window opens + sealer decoys
but NO tag-dependent divergence. Retesting with MY tuned downstream (warm/zero guess, data-dependent
AND-xzr gate, FIXED probe at set 32 window edge) and a deep slow-store chain (wide v4 window), placed
tight after the bypass. Honest test of the established negative.

## Experiment 20 RESULT — v4 (store-bypass) window: NO clean MTE shrink (confirms established negative)
measure_variants on existing templates/mte_tiktag_v4.py (known-eligible) + config_mte_tiktag_v4.yml:
  baseline value-selected probe set @1.0; decoy/noncanon keep it @1.0 and only ADD weak sets (~0.1-0.2);
  only 1/4 inputs a weak DIVERGE (set 60: 0.02->0.19) = the same "mistag perturbs the tail" noise, not
  the shrink. CONFIRMED: v4/store-bypass does NOT produce the TikTag shrink for MTE. The branch
  (cold-misprediction) window is required. (My ASM v4-clean gadget was read arch-eligible-False due to
  store/load clamp aliasing in the CE; the existing Python v4 template is the faithful eligible test.)

## Experiment 21 — MOST GENERAL working MTE template: planted chain + random tails

User goal: most general template that WORKS (any variant). Non-negotiable ingredients for a working MTE
signal (from all experiments): cold-mispredicted BRANCH window (body on fall-through), warm guess load,
data-dependent gate, FIXED probe at the window edge. Everything else -> random. general_mte.asm keeps the
critical chain and adds .random_instructions AFTER the probe (speculative tail) and after the join (arch
tail). Test: does set-32 shrink survive the random tails?

## Experiment 22 — CONFIG CONFOUND CORRECTION (flush off vs on); random tails actually WORK

User prompted "try more tests". Found a confound: my recent measures (Exp16/17/21 for general_mte,
general_mte_alu, user_skeleton) used s1_mte.yml which has enable_pre_run_flush: FALSE. The F+R signal
needs pre_run_flush TRUE (flush the sandbox before each run so the probe line starts absent). Re-measured
with s1_mte_hw_hard_noMT.yml (flush ON, mistraining OFF):
  s1_tiktag_hard : set32 0.50-0.56 -> 0.01-0.02  (known-good restored; earlier sanity "fail" was flush-off)
  general_mte_alu: set32 0.51-0.75 -> 0.00-0.01  (planted chain + RANDOM ALU tails -> CLEAN shrink, all inputs)
  general_mte    : set32 0.54-0.78 -> 0.00-0.04 under noncanon (shrink present) + cache-pollution noise
                   from the random MEMORY tail (extra weak diverging sets).
CORRECTION: the earlier "random tails / random body break MTE" conclusion was a flush-off artifact, NOT a
real property. Random content around the planted chain DOES preserve the shrink (cleanly with ALU-only
tails; with memory tails the shrink survives but adds pollution noise). Re-testing the random-BODY
templates with flush ON next.

## Experiment 23 — random-body MTE with flush ON: genuine tag-dependent divergence (gain flavor)

flush ON (s1_mte_hw_hard_noMT.yml):
  thin_mte_randbody (fully random body): DIVERGE at one set every input, baseline 0.54-0.61 ->
    decoy/noncanon 0.83-1.0 (decoy GAINS residency). Seal-driven (decoy & noncanon agree) -> genuine
    tag-dependence, but a "mistag perturbs the speculative footprint" flavor, NOT the clean window-shrink
    (which is baseline-high -> decoy-low, seen only with the planted chain).
  user_mte_skeleton: weak/mixed gain-flavor.
So fully-random-body DOES give a genuine seal-driven tag-dependent divergence (earlier negative was the
flush-off confound + tracking only set 32). Confirming the FUZZER reports it next.

## Experiment 23 RESULT — general template reliability (flush ON, mistraining OFF): 3/3 both

general_mte_alu (planted core + random ALU tails) : run1=1 run2=1 run3=1  -> 3/3
general_mte     (planted core + random MEM tails) : run1=1 run2=1 run3=1  -> 3/3

The general template (planted cold-mispredict window + warm guess + data-dependent gate + fixed probe,
wrapped in random tails) RELIABLY reports the TikTag violation, including with random MEMORY tails. This
supersedes the Exp21 flush-off false-negative. Fully-random BODY (no planted core) remains genuine but
weak/unreliable (gain flavor; fuzzer 0). Reliable clean shrink needs the planted timing-critical core.

## Python object-template (user asked): tried, blocked by Builder `if_`
mte_tiktag_pytmpl.py plants the core + Kind.ALU/regs-excluded holes (holes DO support Kind+regs, > ASM
macro). But Builder `if_` puts the speculative body on the branch's TAKEN block; a cold forward branch
predicts not-taken, so a taken-body speculates only via the DEPRECATED mistraining. The detector needs
the body on the FALL-THROUGH (not-taken) path (ASM `CBNZ x9,.skip`), which `if_` cannot express. Python
run got 0 even with flush-on+mistraining-on -> structural, not config. A fall-through-body Builder
primitive would be a shipped-source change (out of scope).

## Experiment 24 — triage of the general-template violations (correct vs incorrect tags)

tag-sweep (probe set 32 residency by spec-guess retag delta 0..15), flush ON, mistraining OFF:
  general_mte_alu: 0:0.76 9:0.59 ; all 14 checked mismatches 0.00-0.10  -> GENUINE TikTag tag-check.
  general_mte    : 0:0.92 9:0.92 15:0.66 ; all checked mismatches <=0.08 -> GENUINE TikTag tag-check.
  (delta 0 = matching tag; 9/15 = TCMA-unchecked tags 0b0000/0b1111; every genuinely-checked mismatch
   suppresses the downstream probe -> the signal tracks the tag CHECK, not any tag change = genuine.)

randbody (thin_mte_randbody) direct measure: seal-driven divergence but GAIN flavor (decoy/noncanon
  increase residency at the random body's sets, e.g. 16: 0.38->0.6, 48: 0.53->0.9, consistent over all 3
  decoys + noncanon). This is genuine tag-DEPENDENCE but NOT the TikTag downstream-probe shrink; it is a
  mistagged-load-enlarges-speculative-footprint effect. The randbody FUZZER found 0 violations.

User's suggested skeleton (user_mte_skeleton.asm = random / planted window / random / if(planted){ random
/ sealed load / random } / random): FUZZER found 0 violations (flush ON). Reason: it plants the sealed
load but NOT the tuned downstream (data-dependent gate + fixed probe AT the window edge); random code sits
between the sealed load and any probe, so there is no clean timing-gated downstream -> only weak
gain-flavor divergence, no clean shrink. general_mte_alu works because it keeps the tuned core
(window->guess->gate->probe) intact and puts random ONLY in the tails after the probe.

## Correction (coordinator, accuracy): Python if_ — separate FACT from CAUSE
VERIFIED MECHANICAL FACT: Builder `if_` puts the body in the TAKEN block; the fall-through path is just
`B join` (body cannot be placed on the not-taken/fall-through path with `if_`).
UNTESTED CAUSE: the claim that this taken-block shape is WHY the Python template measured 0 is NOT
established. The 0 is equally consistent with (a) a single fixed-PC branch not mispredicting at all
(fixed-single-PC ~0% mispredict effect) and (b) an unverified assumption about cold forward-branch
prediction direction on the X3. Treat the block-shape as fact; treat the causal explanation of the 0 as
a hypothesis, not a conclusion.

## Builder.py scope: DEFERRED by coordinator/user ("do it later"). No shipped-source edit, no new-module
primitive, no diff. The reliable detector (ASM planted core + random tails) stands as the deliverable.

## Experiment 25 — triage of the USER-SKELETON violations (user_mte_skeleton, flush ON)

user_mte_skeleton (user's template) found 3 violations in the extended run (40 fills x10 inputs):
eval_out/try_user_mte_skeleton/violation-261022-{155905,160128,160705}. Suspect = input #5
(counterexamples #5/#15/#25/#35 = baseline + 3 decoys of input 5). Batch (b78ycq7s4) stopped to free the
device for triage; device recovered (rmmod/insmod).

S1 (tag-only diff, no device): ALL 3 TAG-ONLY (enacted variants differ only in ADDG tag deltas).
HW diverging set (#5 baseline vs #15 decoy): 160705 -> set 32 ; 160128 -> set 39 ; 155905 -> set 9.

Divergence FLAVOR (measure_variants on each generated.asm, fresh inputs, baseline/decoy/noncanon):
  160705: GAIN flavor (decoy raises residency), weak, decoy0/noncanon inconsistent.
  160128: consistent GAIN flavor (decoy & noncanon RAISE sets, e.g. 27:0.58->0.94, 1:0.61->0.95).
  155905: one fresh input shows a clean SHRINK at set 9 (1.0 -> 0.01, decoy AND noncanon) = candidate
          genuine; other inputs weak/gain.

Skill scripts (verify-mte-tag-leak) are stale for this pipeline: accounting.py expects legacy .bin
(worked via a .reif->.bin symlink for S1) but its S2 crashes on a removed `_pac_keys` attribute. So the
decisive 16-tag sweep is done with a new script triage_sweep.py (loads the real input #5 from .reif,
knocks out each sealed-load retag to find the one feeding the diverging set, then sweeps that load's tag
0..15 watching the diverging set). GENUINE = match delta (+ TCMA 0/15) separated; NOISE = uniform/gain.
Result recorded next (Exp26).

## Experiment 26 — DECISIVE 16-tag sweep of the USER-SKELETON violations (on real input #5)

triage_sweep.py (loads generated.asm + real input #5 from .reif; knocks out each sealed-load retag to
find the one feeding the diverging set; sweeps that load's tag delta 0..15 watching the diverging set):

  violation 160705  set 32  leaking off=0x12c (genuine delta 7), baseline 0.69:
    delta 0..15 = 0.61 0.62 0.64 0.65 0.60 0.58 0.58 0.56 0.59 0.59 0.62 0.60 0.55 0.62 0.64 0.62
    => UNIFORM, match(7)=0.56 sits mid-range, no separation => NOISE (prefetcher/batch artifact).
  violation 160128  set 39  leaking off=0x38 (genuine delta 7), baseline 0.61:
    delta 0..15 = 0.54 0.12 0.18 0.15 0.06 0.09 0.09 0.57 0.11 0.10 0.11 0.12 0.12 0.08 0.09 0.07
    => match(7)=0.57 separated; all checked mismatches 0.06-0.18; delta0=0.54 (TCMA-unchecked escape)
    => GENUINE MTE tag-check.
  violation 155905  set 9   leaking off=0xdc (genuine delta 7), baseline 1.00:
    delta 0..15 = 0.00 0.00 0.00 0.00 0.00 0.05 0.00 1.00 0.01 0.00 0.03 0.00 0.00 0.00 0.02 0.03
    => ONLY match(7)=1.00, every other tag ~0 => TEXTBOOK GENUINE TikTag tag-check (cleanest possible).

VERDICT: 2 of 3 user-skeleton violations are GENUINE TikTag tag-check leaks (155905 textbook, 160128
clear); 1 is noise (160705). The user's template (planted window + ONE sealed load + random code, NO
deliberate probe/warm) DOES find genuine TikTag via random fuzzing — the random tail supplied a dependent
downstream access inside the window, and the tag mismatch cleanly shrinks it.

CORRECTION to earlier claims: my earlier "user skeleton = 0 / only weak gain-flavor / random-body can't
do clean MTE" was based on (a) a single -n1 run (0 hits) and (b) FRESH-input measure_variants (which
showed gain-flavor at other sets because the set/load are input-specific). The faithful per-input 16-tag
sweep on the REAL violation input is decisive and shows genuine clean tag-check shrinks. So a mostly-random
template DOES auto-detect TikTag at some rate (here 2 genuine / 40 fills for user_mte_skeleton); it is
rarer and lower-yield than the fully-planted gadget, but it is NOT zero and NOT only gain-flavor.

## Experiment 27 — STABILITY RECHECK of the 2 genuine user-skeleton violations (2 extra passes, reps 1500)

  155905 set 9  off=0xdc (genuine delta 7):
    pass1: 7:1.00 ; all others 0.00-0.06      pass2: 7:1.00 ; all others 0.00-0.02
    => ONLY the matching tag leaks, every pass. GENUINE TikTag, STABLE (3/3 passes incl. original).
  160128 set 39 off=0x1c (genuine delta 7; 0x1c/0x38 are the top-2 knockouts = the same load's retag pair):
    pass1: 0:0.57 7:0.59 ; checked mismatches 0.00-0.10
    pass2: 0:0.57 7:0.55 ; checked mismatches 0.00-0.04
    => match separated from every checked mismatch every pass (delta 0 = TCMA-unchecked escape).
       GENUINE tag-check, STABLE (3/3 passes).
  160705 set 32: NOISE (uniform across all 16 tags; no separation) — unchanged.

FINAL TRIAGE of user_mte_skeleton's 3 violations: 2 GENUINE TikTag tag-check leaks (reproducible), 1 noise.

## Experiment 28 — 24h nonstop RANDOM MTE fuzzing on the working config (user request)

Config: docs/tiktag_mte_ni/general_detection/s24_mte_random.yml = the working config
(s1_mte_hw_hard_noMT.yml: MTE NI, cond, flush ON, mistraining OFF, inputs_per_class 4, sample [2000],
pinned core 8) with min/max_bb_per_function 6/10 for richer random structure (more branch windows+loads).
Launch (background by4vas6wr): revizor.py fuzz -s base.json -c s24_mte_random.yml -n 500000 -i 8
--nonstop -w eval_out/run24h  (no template = fully random generation).
Start 2026-10-22T16:36:18Z; seeds program:931199 input:3145587579; ~0.13 tc/s (~11k tc/day); saving
violations to eval_out/run24h/. Log: eval_out/run24h/run24h.log.

TRIAGE PLAN for its violations: the run monopolizes /dev/executor, so triage (which needs the device)
happens by pausing/stopping the run, then for each saved violation-*: triage_sweep.py <cfg> <vd>
<suspect_idx> <diverging_set> — the decisive per-input 16-tag sweep (match tag separated => genuine
TikTag; uniform => noise). Suspect_idx + diverging set come from the violation report.txt counterexamples
and the baseline-vs-decoy per-set diff (as in Exp25-26). Expect mostly noise at a low genuine rate.
