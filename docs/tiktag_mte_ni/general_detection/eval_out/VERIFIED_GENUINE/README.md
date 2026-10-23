# MTE TikTag-v1 triage — CORRECTED verdicts (ground-truth genuine-vs-decoy)

These two violations came from the user's spectre-v1 + random-instructions template (`user_mte_skeleton.asm`).
An earlier write-up of this file overclaimed 160128 as "tracking-confirmed genuine"; that was WRONG — the
tracking control had swept an ARCHITECTURAL tag gate (off 0x1c, identical in genuine and decoy), not the
speculative one. The decisive test is the ground-truth genuine-vs-decoy residency (view_rotation ON,
4000 reps, `gd_check.py`), which is independent of gate selection. Results (`GD_CHECK.txt`):

## userskel_155905 — GENUINE TikTag-v1 (set 9)
- set 9 residency: baseline (genuine) = **1.00**; decoy0/1/2 = **0.00**; noncanon = **0.00**.
- The genuine (matching) tag lets the speculative guess load execute so the downstream probe fills set 9;
  ANY mismatched tag suppresses it (clean TikTag "shrink" direction). Robust across all decoys at 4000 reps.
- set 9 is CE spec-only, and bb_0.1 (the fall-through carrying the sealed load) is architecturally dead
  (always-taken `ORR x9,#1; CBNZ`), so set 9 is reachable only speculatively. Tag-only diff holds.
- VERDICT: GENUINE speculative MTE tag-check (TikTag-v1) leak.

## userskel_160128 — GENUINE speculative MTE tag leak, GAIN flavor (set 39)
- set 39 residency: baseline (genuine) = **0.66**; decoy0/1/2 = **0.85–0.97**; noncanon = **0.89**.
- Robustly tag-dependent (genuine distinguishable from every decoy + noncanon by ~0.2–0.3 at 4000 reps),
  at a CE spec-only set, in the architecturally-dead fall-through block (speculative-only), tag-only diff.
  By every NI criterion this is a genuine speculative MTE-tag leak: the tag value is observable via the
  cache. It is the "GAIN" manifestation — a mismatched tag ENLARGES the speculative footprint at set 39
  rather than suppressing a downstream probe (the 155905 "shrink"). Different direction, same genuineness.
- CORRECTION (integrity): the earlier "memory-tag tracking confirmed genuine" PROOF for 160128 was an
  artifact — the knockout picked the architectural gate off 0x1c on shared granule 0; restricted to the
  decoyed/speculative gates the pointer sweep is flat and the validity gate fails. That invalidates the
  TRACKING proof method for 160128, NOT the leak — genuineness here rests on the ground-truth
  genuine-vs-decoy divergence above, which is independent of gate selection.
- VERDICT: GENUINE speculative MTE-tag leak (gain-flavor). The clean memory-tag tracking control still
  needs a correct decoyed-gate + spec-only-granule run to also confirm the mechanism; pending.

## Record of the error (integrity)
- `TAG_TRACK.txt` = the flawed tracking run (swept arch gate 0x1c) — kept for transparency.
- `TAG_TRACK_CORRECTED.txt` = restricted to decoyed/speculative gates -> flat, VALIDITY FAIL for 160128.
- `GD_CHECK.txt` = the decisive ground-truth measurement (this file's numbers).
- The 3-model audit (Opus/Fable GENUINE, Haiku UNSURE) split on exactly this: Haiku correctly flagged the
  arch-gate problem. Haiku was right.
