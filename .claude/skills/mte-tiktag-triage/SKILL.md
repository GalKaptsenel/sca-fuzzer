---
name: mte-tiktag-triage
description: Triage an AArch64 MTE non-interference (TikTag-class) violation to a 100%-integrity verdict — GENUINE speculative tag-check leak vs QUESTIONABLE vs prefetcher/batch NOISE. Runs the full pipeline (tag-only diff, CE speculative-only accounting, all-16-tags pointer sweep, and the decisive granule/memory-tag TRACKING control) and refuses to call anything genuine unless it passes every gate. Use after a `fuzzer: non-interference` + MTE violation is saved, or to re-verify a "genuine" claim. Pairs with verify-mte-tag-leak, revizor-leak-flow, executor-userland. Encodes the hard pitfalls below — read them before trusting any number.
---

# MTE / TikTag non-interference triage (genuine vs noise)

A TikTag NI violation compares variants of ONE arch input differing only in a pointer's MTE tag: a
genuine (matching tag) baseline vs decoy(s) (mismatched tag). The claim: a speculative access behind the
tag check runs on match and is suppressed on mismatch, so a downstream cache set diverges. Your job is to
decide, with zero self-deception, whether the divergence is the TAG CHECK or an artifact. It is brutally
easy to fool yourself here. Do every gate; never shortcut to "looks genuine."

## The verdict gates (GENUINE requires ALL; else QUESTIONABLE or NOISE)
1. **Tag-only.** The enacted variants differ ONLY in ADDG tag-delta words (+NOP padding). Any non-ADDG
   code diff ⇒ not a tag violation ⇒ NOISE.
2. **Spec-only leaking set.** Run the CE (COND|BPAS, max nesting). At least one HW diverging set must be
   a CE **speculative-only** access, or a prefetch stream of a *speculative* tag-sealed access — NOT an
   architectural demand or its prefetch stream. Architectural demands commit with the correct tag
   regardless of the decoy, so their divergence is batch/prefetcher contamination ⇒ NOISE. If CE
   spec-only sets is empty and every diverging set is an arch demand/prefetch, it is definitively noise.
3. **All-tags pointer sweep separates the matching CHECKED tag.** Sweep the sealed pointer's retag ADDG
   over all 16 deltas; the matching (genuine) tag must dominate every CHECKED mismatch beyond jitter.
4. **Granule/memory-tag TRACKING control** (the decisive proof). Change the guess load's MEMORY tag and
   confirm the detected match delta MOVES with it. Only this separates a real pointer-vs-memory check
   from a fixed pointer-tag artifact.

## Tools (run with the repo venv `/home/gal_k_1_1998/revizor/revizor-venv/bin/python`, `PYTHONPATH=<repo>`)
- `docs/tiktag_mte_ni/general_detection/auto_triage.py <cfg> <violation-dir> [--reps N] [--ledger P]`
  — gates 1–3 end to end; verdict GENUINE/QUESTIONABLE/NOISE; writes `AUTO_TRIAGE.md`; exit 42 = GENUINE.
- `docs/tiktag_mte_ni/general_detection/tag_track.py <cfg> <vd> <suspect> <divset> [--reps N]`
  — gate 4 (granule-tag tracking) with a built-in validity gate.
- `.claude/skills/verify-mte-tag-leak/accounting.py` and `triage_sweep.py` — the primitives auto_triage drives.

## PITFALLS — the ones that cost real time (explicit and learned)
- **TCMA-unchecked tags 0 and 15 are ALWAYS resident** (their check is skipped). Exclude deltas 0/15 when
  computing the match/peak. A "leak" carried only by 0/15 is NOT a checked-tag leak. In the sweep, tag 0
  often sits at ~0.6 and will out-rank the real tracked peak by a hair — exclude it or you misread genuine
  as suspect.
- **The diverging set is the DOWNSTREAM PROBE, not the tag-checked granule.** For the tracking control you
  must retag the GUESS LOAD's granule — the load whose base register is the gating ADDG's `Xd`, at a pc
  AFTER that ADDG — not the granule of the access that hits the diverging set. Retagging the probe granule
  changes nothing and fakes a "not tracking" result.
- **CE `effective_address` carries the MTE logical tag in the top byte.** Mask `& ((1<<56)-1)` before
  computing a granule index, or you get a garbage (huge/negative) index. The cache-SET computation hides
  this (the tag is a multiple of the set span) but the granule index does not.
- **Per-input `mte_tags` IS honored** (`measurement.c` `mte_apply_sandbox_tags`); index = `(addr-base)/16`
  over the main|faulty span (faulty is contiguous after main). So the tracking control is expressible —
  patch ONE granule (the checked one), not globally (global retag faults architectural sealed accesses).
- **Built-in validity gate for the tracking control:** if changing the granule tag does NOT collapse the
  old match delta d0, the patch did not reach the checked granule — declare the control INVALID
  (inconclusive), NEVER a false negative. Fix the guess-granule finder, then re-run.
- **`view_rotation` is what makes the leak reproduce across reps** (set via the legacy combined
  `enable_pre_run_flush: true`, which also sets phr_flush). It serves a fresh code view
  (`invalidate_bpu_entries`) each run → tagged-BTB miss → the forward branch cold-mispredicts → the
  speculative window opens. It is NOT a data-cache flush (the F+R reload flush is unconditional in
  `templates_jit.c`), and phr_flush alone does nothing. With view_rotation off, the leak vanishes — a
  config confound that will make a genuine gadget look dead.
- **Never trust a single measurement.** The 16-tag sweep is itself the recheck; re-run marginal cases;
  demand the separation exceed the repeat jitter. Use reps≥2000.
- **Device exclusivity.** Never triage while a fuzz campaign holds `/dev/executor` — concurrent input-id
  allocation corrupts the kernel allocator. Pause collection first (PID-safe kill of the tracked fuzz PID
  + a PAUSE flag), triage, then resume. If the module wedges: reload (`rmmod`/`insmod`/`chmod`).
- **Fully-random MTE/NI is mostly prefetcher/batch NOISE** on the X3; the spectre-v1 + random-instructions
  template (`user_mte_skeleton.asm`: SDIV window → always-taken CBNZ → speculative fall-through sealed
  load → random tails) is what surfaces GENUINE TikTag. A clean-looking pointer sweep on an arch-demand
  set is still noise — gate 2 catches it.

## Integrity standard (non-negotiable)
State only what is proven. Never fallback, never fabricate, never round a "suspect" up to "genuine."
An invalid control is inconclusive, not a result. Log before and after every experiment; persist every
artifact (AUTO_TRIAGE.md, sweep text, tracking text) in the violation dir. A GENUINE verdict means all
four gates passed and the memory-tag tracking moved the detection — nothing less.
