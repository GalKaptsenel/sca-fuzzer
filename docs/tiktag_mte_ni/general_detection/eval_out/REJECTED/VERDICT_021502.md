# violation-261023-021502 — REJECTED: architectural MTE tag-check effect, NOT a speculative TikTag leak

An earlier draft of this file claimed a GENUINE two-sided speculative TikTag leak on L1D set 52. That
claim is WITHDRAWN. A six-model audit (opus/fable/haiku, informed and uninformed; four completed, two
stopped early) unanimously rejected it. The set-52 result was produced by sweeping the wrong seal — an
architectural load the fuzzer's decoy never varies — the same defect that was retracted for 160128.

## Why the set-52 claim is wrong
- The fuzzer's NI decoy perturbs exactly ONE word: the ADDG at code offset 0x15c (verified by `cmp -l`
  of enacted_test{0000,0008,0016,0024}.bin — they differ only at that word, tag 7/14/11/4). 0x118 is
  byte-identical in all four. The sealer marks exactly one of its 9 MteSealing entries speculative=True:
  the 0x15c one.
- 0x15c gates a SPECULATIVE store (CE nesting 1) to granules 171/69/17 (sets 42/17/4). The seal I swept,
  0x118, gates an ARCHITECTURAL load (CE nesting 0) of granule 442 (set 46, faulty page but
  ENABLE_FAULTY_DEFAULT=0). Different access, different granule, different phase.
- Set 52 is UNTOUCHED by the CE in any phase; set 52 = 46 + 6 is the hardware prefetcher forward stream
  of the architectural demand at set 46.
- In the real saved counterexample set 52 does NOT diverge: 0.013 vs 0.045 (|Δ|=0.032, below the 0.08
  threshold), while set 46 is 1.00/1.00 — tag-independent, the signature of a committed architectural
  demand. The real HW diverging sets are [6,7,8,32].
- `tag_track_rc.py` selected 0x118 by raw knockout magnitude because it dropped the decoyed-gate
  restriction that the corrected `tag_track.py` added precisely to prevent this.
- The 2D tracking proves only that a tag COMPARISON gates set 52 (rules out uniform noise); it does not
  prove the comparison is speculative. An architectural MTE tag-check (fault vs no-fault on the granule-442
  demand) tracks the memory tag identically. Tracking is not the arch-vs-speculative discriminator.
- The sign is backwards for a shrink-flavor TikTag: match = 0.02 is the LOWEST residency, mismatches
  ~1.00. Genuine TikTag-v1 (155905) has the match HIGHEST (access executes -> set resident). On an
  architectural checked load, a mismatch is a tag-check fault; the device returned normal data, so the
  mechanism is architectural/fault-path, not a speculative-window tag check.
- My "tag-0 TCMA special case" note was itself wrong: base tag is 15, so delta 0 = pointer tag 15 and
  delta 1 = tag 0; the scripts exclude deltas {0,15} = tags {15,14}, not tags {0,15}.

## The actual lead for this test case (not yet characterized)
The violation's consistent signal is sets [6,7,8] (all three decoys) and set 32 (vs #8 only). These are
prefetch-stream positions of the decoyed SPECULATIVE store at 0x15c (spec-only sets [4,17,42], granules
17/69/171). A correct test sweeps seal 0x15c's pointer tag and retags granules 17/69/171 (NOT 442),
watching sets [6,7,8] / 32, with a position null-control first (input #0 alone gets the warm-up rounds
and PHR flush, so the match/decoy split is confounded with batch position). Until that gate-independent
check (gd_check.py on this violation, never run) separates tag from position, the violation is at best
INCONCLUSIVE — classified NOISE/artifact for the set-52 claim.

Raw logs retained here: TAG_TRACK_RC_021502.txt, TAG_TRACK_RC_021502_REPEAT.txt, MEMTAG_021502.txt,
TAG_TRACK_2D_021502.txt (all measured on seal 0x118 — the wrong gate).

## Device verification battery (2026-10-23, campaign paused; user-directed "do all verifications")
All candidate channels tested on the Pixel-8 X3; every one fails. FINAL: 021502 has NO genuine speculative
MTE tag-check channel. It is architectural-MTE + batch-position + prefetcher/jitter noise.
1. gd_check.py (gate-independent, 4000 reps, GD_CHECK_021502.txt): baseline vs 3 decoys + noncanon.
   set 52 NOT resident / NOT divergent (confirms the 0x118 claim dead). Apparent consistent divergence at
   sets [6,51,56,58] (genuine ~0.01 -> decoy ~0.2-0.36) and a weak set 32. BUT gd_check batches the
   variants at DIFFERENT slots (baseline slot 0, decoys slots 1-4) -> position-confounded.
2. ctrl_pos_tag.py (fixed prefix, swap only the suspect at a fixed slot, CTRL_POS_TAG_021502.txt):
   at a FIXED slot, sets [6,18,51,52,56,58] have Δtag ~= 0 (genuine == decoy) -> the gd_check divergence
   there was BATCH POSITION, not the tag. Only set 32 showed a fixed-slot Δtag (+0.37 at tail) but it was
   position-modulated (no effect at head) and the genuine baseline was unstable (0.55 vs 0.94 across runs).
3. ctrl_store_track.py (position-controlled, decoyed gate 0x15c auto-selected, retag granule 69,
   CTRL_STORE_TRACK_021502_set32_gran69.txt): sweeping the speculative STORE's pointer tag 0..15 at a
   fixed tail slot for memtags T=3/6/9/12, set 32 is FLAT at ~0.93 everywhere; dips do not match the
   predicted match deltas and do not track T. No pointer-vs-memory tag tracking -> set 32 is not genuine.
The decoyed-gate guard (now enforced) correctly selected 0x15c, and the controlled result is still
negative: this test case does not carry a detectable speculative tag-check leak.
