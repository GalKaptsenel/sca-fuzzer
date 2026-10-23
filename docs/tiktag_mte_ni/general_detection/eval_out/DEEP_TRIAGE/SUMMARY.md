# Deep triage of all collected fully-random violations (2026-10-23, user-directed)

Battery per violation: gd_check.py (gate-independent genuine-vs-decoy, 4000 reps) -> for any divergent set,
ctrl_pos_tag.py (fixed prefix, swap only the suspect at a FIXED slot: separates batch position from tag)
-> for any surviving fixed-slot tag effect, ctrl_store_track.py (position-controlled tag tracking at the
decoyed gate). Campaign paused throughout (single-access).

## Verdicts — ALL 8 fully-random violations are NOISE
| viol | suspect | gd_check | fixed-slot control | verdict |
|------|---------|----------|--------------------|---------|
| 000605 | 5 | no divergence | - | NOISE |
| 051434 | 5 | no divergence | - | NOISE |
| 004926 | 5 | set27 0.01->0.24 | Δtag -0.04, Δpos +0.18 | NOISE-ARTIFACT (position) |
| 010423 | 5 | sets1/32/53/54 | Δtag ~=0, Δpos +0.03..0.07 | NOISE-ARTIFACT (position) |
| 032456 | 2 | set49 0.02->0.72, set17 0.70->0.97 | Δtag NEG (-0.06/-0.11), Δpos +0.06..0.26 | NOISE-ARTIFACT (position) |
| 061958 | 5 | decoy0 set5/17; decoy1+noncanon EMPTY | Δtag +0.07/+0.02 sub-threshold | NOISE (inconsistent+position) |
| 062913 | 5 | sets38/50 | Δtag ~=0, Δpos +0.05/+0.08 | NOISE-ARTIFACT (position) |
| 021502 | 0 | (full battery earlier) | set52 arch; set32 no tracking | NOISE-ARTIFACT (REJECTED/) |

## Key methodological finding
gd_check.py itself is BATCH-POSITION-CONFOUNDED: it measures baseline/decoy0..2/noncanon in ONE batch at
different slots, and only the first slot gets the warm-up rounds + PHR flush. So a gd_check "gain" (even a
large one: 032456 set49 0.02->0.72) can be entirely position. The fixed-slot control (ctrl_pos_tag: fixed
prefix, swap only the suspect at one slot) is REQUIRED to separate position from tag, and here it removed
EVERY apparent divergence. The earlier "4 of 6 gain-flavor genuine" exam (004926/010423/032456/021502) was
based on gd_check-style divergence and is REFUTED — all position artifacts.

## Conclusion
Fully-random MTE non-interference on the Pixel-8 Cortex-X3 produces only prefetcher / batch-position
artifacts; no fully-random violation to date carries a genuine speculative pointer-vs-memory tag-check.
Genuine TikTag on this box has only been shown from the user_mte_skeleton template (155905) and the manual
/dev/executor PoC. Evidence: GD_ALL.txt, POS_ALL.txt (this dir); 021502 in eval_out/REJECTED/.
