---
name: verify-mte-tag-leak
description: Verify whether an AArch64 MTE non-interference (TikTag-class) violation is a GENUINE speculative tag-check leak or just HW-prefetcher / measurement noise. Runs the contract executor (CE) locally to account for EVERY evicted set (architectural demands + HW-prefetcher forward streams + speculative accesses that sit behind a tag check), then measures on the device with the CONTROLLED protocol (fixed prefix, swap only the suspect) and — decisively — SWEEPS the sealed pointer's tag over all 16 values: a real tag-check leak makes the MATCH tag stand out; noise fires uniformly across tags. Use AFTER a `fuzzer: non-interference` + MTE violation is found, when you must confirm the tag is truly the leak source (not batch contamination, not the prefetcher). Pairs with revizor-leak-flow (mechanism) and reproduce-violation-manual (generic controlled swap).
---

# Verifying an MTE non-interference (TikTag) tag leak

An MTE non-interference violation compares **variants of one arch input** that differ ONLY in a
pointer's MTE tag: a *baseline* (genuine/correct tag = MATCH) vs a *decoy* (wrong tag = MISMATCH). The
claim is TikTag: a speculative access behind the tag check executes on MATCH but is **suppressed** on
MISMATCH, so the two variants' hardware traces diverge. This skill decides whether that divergence is
**real (the tag)** or an **artifact** (batch/prefetcher noise). It is brutally easy to fool yourself
here — do every step.

## The two traps this defends against (both bit me)
1. **Batch/prefetcher contamination.** An input's htrace depends on its predecessors in the same
   super-batch (the X3/A510 prefetcher state carries across). Measuring several variants together, or
   the same variant in different batch positions, fabricates or *flips* the signal. Proof it's real: a
   GENUINE match tag scattered exactly like a mismatch when measured in a 16-variant batch, and was
   clean in a 2-batch. ALWAYS use a fixed prefix and swap only the suspect.
2. **A single controlled run is not enough.** One fixed-prefix swap showed baseline>decoy (looked like
   v1). The definitive **tag sweep** (16 tags x repeats) then showed MATCH sat in the *middle* of the
   mismatch range — no tag dependence => it was prefetcher noise that passed priming once. Only the
   sweep is decisive.

## Prerequisites
- Config is `fuzzer: non-interference` with MTE categories; the same-input NI fix is in
  (`_boost_inputs` keys class_ctrace by `hash(inp.tobytes())` — see [[ni-compare-same-input-only]]), so a
  violation's counterexamples are ONE input's baseline + its decoy(s), not two different inputs.
- Use a `superbatch_size: 1` executor config for the device work (large batches hit the
  "not a batch response (bad magic)" transport bug — [[superbatch-badmagic-remote]]). The scripts here
  load such a config by path.
- **Never run these device scripts while a Python campaign is live** — concurrent input-id allocation
  corrupts the kernel allocator. Pause the campaign. If you `pkill -9` a campaign mid-`run_batch`,
  reload the module: `su -c 'rmmod revizor_executor; insmod /data/local/tmp/revizor-executor.ko; chmod 777 /dev/executor'`.

## Step 1 — Local structural triage (no device)
From the violation dir:
- Counterexamples (Counterexample section only): confirm they are `#N` (baseline) and `#N+K` (decoy) of
  the SAME input (K = kept-input count). Two *different* inputs => it's a data (Spectre-v1) split, not a
  tag leak — the same-input fix should already prevent this.
- Code-diff the `enacted_test*.bin`: the ONLY differences must be ADDG (tag-delta) / NOP words. Any
  non-ADDG diff means it is not tag-only.
- Per-set divergence from `report.txt` htraces. **report strings are MSB-first: set = 63 - char index**
  (at the raw `HTrace.raw` int level, bit b == set b — no reversal). The diverging sets are usually a
  short *consecutive run* (a prefetch stream).

## Step 2 — CE per-set accounting (still no device)
Run the CE on the baseline (COND|BPAS, max nesting) and attribute every hot set:
- **arch demand** (nesting 0): always ~1.0, tag-independent.
- **speculative-only demand** (nesting>0): reached only speculatively — the leak-relevant accesses.
- **prefetch stream**: each demand D trains a forward run D+1..D+~16 (the Pixel-8 prefetcher footprint;
  footprint = demand ∧ a TRAINED stream — see [[a510-prefetcher-covert-channel]]). Attribute each extra
  hot set to the nearest demand below it.
- **Verify the diverging sets are prefetch streams of SPECULATIVE demands whose base register is
  tag-sealed** (decode the access at each spec-demand pc; its base reg must be an eligible MTE slot).
  The same instruction typically hits one set architecturally and a *different* set speculatively
  (the base differs on the mispredicted path). This is the CHECK->TEST structure, usually in one BB.

If the diverging sets are NOT prefetch streams of tag-sealed speculative accesses, it is almost
certainly noise — stop here.

**Fast structural noise-classifier (usually decides it without the device):** decoys perturb ONLY
speculative slots, so ARCHITECTURAL demands and their prefetch streams commit with the correct tag and
MUST be tag-independent. If the HW divergence is dominated by prefetch streams of ARCH demands, the
violation is *definitively* prefetcher/batch noise — no sweep needed. Only when the divergence
concentrates on SPECULATIVE-demand prefetch streams (base register = an eligible MTE slot) is a
controlled sweep (Steps 3-4) worth the device time.

## Step 3 — Controlled single measurement (device; pause the campaign)
One `load_test_case` (consistent skeleton — the seal salt is random per load, so saved code_reloc
offsets do NOT match a fresh skeleton; ALWAYS measure with FRESH variants, never splice a saved
code_reloc). Build a FIXED prefix of ~14 baseline variants of other inputs, then measure
`[prefix + suspect_baseline]` vs `[prefix + suspect_decoy]`, reading the LAST slot's per-set frequency
over many reps. Same context, swap only the suspect. A persistent per-set difference on the predicted
sets is necessary but NOT sufficient (see trap 2).

## Step 4 — THE DECISIVE tag sweep (device)
Sweep the sealed register's tag over all 16 deltas (one is the genuine MATCH), each measured with the
SAME fixed prefix (separate batches, never all-in-one), repeated ~6x. Watch the predicted leak sets.
- **GENUINE tag-check leak (TikTag-v1):** the MATCH delta is clearly separated — highest leak-sum
  (streams present, access executes) with all MISMATCH deltas lower (suppressed). Persists across
  repeats.
- **NOISE:** leak sets fire ~uniformly for all deltas; MATCH is not separated; repeat jitter (max-min)
  exceeds the match-vs-mismatch gap. (This is what every fuzzer-found suspect has shown so far on the
  X3 — the generated gadgets likely don't meet v1's cycle-distance timing.)

## Verdict
GENUINE only if: tag-only diff (S1) AND diverging sets are prefetch streams of speculative tag-sealed
accesses (S2) AND the controlled sweep separates MATCH from all mismatches beyond the repeat jitter
(S4). Anything less => prefetcher/measurement noise (a false positive that slipped through priming).
To claim a real leak on a new box, also run the OFFICIAL TikTag PoC as a positive control
(reproduce-spectre-v4 is the template for a hand-built device gadget).

## Scripts (in this skill dir; each takes `<config.yml> <violation-dir> ...`)
- `accounting.py <cfg> <vd> <suspect_input_idx>` — S1+S2: counterexample check, tag-only code-diff,
  CE per-set accounting + which diverging sets are prefetch streams of speculative tag-sealed accesses,
  and the base register behind each. Prints the register + leak sets to feed the sweep. No device.
- `ctrl_sweep.py <cfg> <vd> <REG> <suspect_input_idx> <leak_sets_csv>` — S3+S4: fixed-prefix controlled
  tag sweep (16 deltas x 6 repeats), prints per-delta leak-sums and the MATCH-vs-mismatch verdict.
  Device; pause the campaign first.

Run `pip`-free with the repo venv: `/home/gal_k_1_1998/revizor/revizor-venv/bin/python`.

## Cross-references
- **revizor-leak-flow** — structured per-instruction CE trace (arch/spec phase, MTE tag, set).
- **reproduce-violation-manual** — the generic controlled-swap protocol (verify.py; NOTE: verify.py does
  NOT support NI sealed artifacts — its input_gen.load drops the seal sections and the device rejects
  with "Malformed input initialization"; use this skill's scripts instead).
- **executor-userland** / **craft-executor-inputs** — raw /dev/executor + building a positive-control gadget.
- Background: [[a510-prefetcher-covert-channel]], [[prefetch-batch-contamination]],
  [[ni-null-decoy-bug-and-fix]], [[mte-decoy-nonrevert-and-scatter]].
