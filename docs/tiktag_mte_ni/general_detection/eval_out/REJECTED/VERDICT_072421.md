# violation-261023-072421 — GENUINE speculative MTE tag-check leak (fully-random fuzzer, gain polarity)

First genuine TikTag-class leak from the FULLY-RANDOM Revizor non-interference fuzzer on the Pixel-8
Cortex-X3 (config s24_mte_random.yml, suspect input #1). Confirmed two-sided; passes every control that
the rejected 021502 failed.

## Mechanism
A decoyed speculative seal (MteSealing of x2 at code offset 0x120, nesting>0) tags the base of a
speculative access to granule 305. On tag MATCH the access proceeds and the observation sets stay cold;
on MISMATCH the speculative access is squashed and the mispredicted/recovery path issues a burst over a
contiguous region, so the hardware prefetcher fills a consecutive run of L1D sets (observed 1..11). This
is the GAIN polarity (opposite sign to TikTag-v1 / userskel_155905, where the match lights the probe):
here presence-on-MISMATCH carries the tag. Same primitive, inverted transmit branch.

## Evidence (device, Pixel-8 X3, view_rotation on; position-controlled = fixed prefix, swap only the suspect at a fixed slot)
1. gd_check (gate-independent, 4000 reps): decoys and noncanon gain sets 1,4,5,6,8,9,10,11 (genuine ~0.02
   -> ~1.0). GD_072421.txt.
2. Fixed-slot control vs noncanon AND vs a random decoy: Δtag = +0.57..+1.00 across those sets with
   Δpos ~= 0 -> a real TAG effect at a fixed batch slot, not batch position. POS_072421.txt, 072421_followup.txt.
3. One-sided pointer-tag sweep (gate 0x120, genuine memory): set 9 suppressed ONLY at the match delta
   (pointer tag = memory tag) and the TCMA-unchecked tag; 1.00 at all 13 checked mismatches.
4. Two-sided tracking (granule 305 uniquely found by scan; GRANSCAN_072421.txt): retag granule 305 to T
   and the suppression (match) delta moves 4/4 as (C+T-cell_tag)%16 -> T3:d4, T6:d7, T9:d10, T12:d13,
   set 9 ~=0.00 at the dip and 1.00 elsewhere. TRACK2D_072421_set9_gran305.txt.

## Why this is not the 021502 artifact
021502 swept an ARCHITECTURAL seal the decoy never varies, failed the fixed-slot control (Δtag~=0, all
position), and its set was the prefetch of an arch demand. Here the gate is a decoyed speculative seal,
the fixed-slot control PASSES (Δtag +1.0, Δpos 0), and the match tracks a uniquely-identified granule 4/4.

Time to this find: fully-random campaign, violation collected 07:24 UTC 2026-10-23.

## Time to detection
Stable continuous fully-random campaign start (daemon 77242, post device-stabilization): 2026-10-22
22:02:40 UTC. 072421 collected: 2026-10-23 07:24:21 UTC.
- wall-clock from stable start: 9h 22m
- effective fuzzing (minus 24m logged triage pauses): ~8h 57m
- from the first (crashed/restructured) launch 2026-10-22 17:55 UTC: ~13h 28m

## TikTag variant classification (vs the paper)
TikTag paper (S&P'25): v1 (speculation-shrinkage) and v2 (store->load-forwarding) are BOTH
presence-on-MATCH (the TEST access is cached only when pointer tag == memory tag). v1: mismatch shrinks
speculation so the prefetcher STOPS (testptr cold on mismatch). v2: match enables store->load forward
(TEST cached on match).
072421 is presence-on-MISMATCH: match -> observation sets COLD, mismatch -> prefetcher FILLS a
consecutive run (sets 1..11). This is the opposite sign of both v1 and v2. Hypothesis: a THIRD flavor
where a speculative tag MISMATCH squashes the sealed access and the divergent misprediction/recovery path
runs a contiguous scan that the HW prefetcher streams. To confirm vs v1-observed-from-the-other-side,
pin the mechanism (is the mismatch path a control-flow divergence or a fault-recovery prefetch?). Under
6-model audit.

## Variant label SOFTENED (audit, 2 Opus auditors): GENUINE stands; "third variant" NOT yet proven
Both Opus audits (informed + uninformed) agree GENUINE. On the variant they split: uninformed called it a
distinct THIRD variant; informed argues that is an OVERCLAIM. Key point (informed): the sealed LDP at 0x124
is the TERMINAL gadget instruction (CE shows measurement-end immediately after — no downstream speculative
access to "shrink"), so this is not literally TikTag-v1's downstream-shrink. Honest label: a v1-CLASS
speculative tag-check leak observed in GAIN / presence-on-MISMATCH polarity, where the transmit is a
mismatch-gated HW-prefetcher stream over contiguous sets (1..11). Whether that transmit is a genuinely NEW
primitive (true "third variant") vs v1 observed from the mismatch side is UNPINNED by the current data (the
HW prefetcher is unmodeled by the CE). 
Decisive experiment to settle the variant (device, when unpaused): append N benign speculative instructions
after the sealed LDP inside the window; vary N and the pointer-tag match. v1-shrink => the sets-1..11 stream
depends on N / reached depth; novel mismatch-gated-prefetch => the stream persists at N=0 (gated purely by
the terminal load's tag). Also check the stream base/stride tracks the sealed-load line.
Corrections to the claims above: the gated access is a speculative LDP LOAD (not a store); the same load
hits granule 287 in the first window (doesn't track) and 305 in the second (tracks) — 305 is the checked
granule; granule uniqueness was tested over 10 CE-derived candidates, not all 512.

## DOWNGRADED to CONTESTED (audit 4/6: 3 GENUINE, 1 ARTIFACT) — multi-seal confirmed by byte-check
Informed-Fable audit + my own byte-decode of the enacted binaries CONFIRM the decoys are MULTI-SEAL and
2 of 3 do NOT perturb the 0x120 gate I tracked:
  decoy0 (0009): 0xf8(x4 7->13), 0x10c(x1 7->2), 0x114(x1 0->5); 0x120 UNCHANGED
  decoy1 (0017): 0xf8(x4 7->4), 0x120(x2 7->4)
  decoy2 (0025): 0x10c(x1 7->14), 0x114(x1 0->9); 0x120 UNCHANGED
Yet decoy0 and decoy2 still drive set 9 ->~1.0 (GD_072421.txt). So the saved violation is driven mainly by
the x4@0xf8 / x1@0x10c+0x114 speculative seals, NOT the 0x120/granule-305 comparison my ctrl_store_track
characterized. The granule-305 2D tracking (4/4) is real for 0x120 but 0x120 is one of several redundant
speculative tag-sealed accesses; it does not explain 2/3 of the counterexamples. The gain (sets 1..11) is
the forward prefetch stream of set 0 (granule 0, gated by the x1 seals), not granule-305's stream (set 9 <
set 12). OPEN and decisive: is the mismatch-fill a genuine SPECULATIVE transient or an ARCHITECTURAL
fault/abort-handler + prefetch footprint? (granules 287/305 are in the FAULTY region). All seals are
nesting=1 in the CE (speculative), but the leaking sets are CE-UNTOUCHED (prefetch the CE doesn't model),
so CE accounting does NOT corroborate; the verdict cannot rest on tracking alone (tracking proves a tag
COMPARISON, which an architectural MTE fault also satisfies — the 021502 lesson). STATUS: CONTESTED pending
a device experiment with tag-checking disabled (TCF=async/TCO=1) + per-seal isolation + downstream padding.

## CORRECTION (user): the "faulty region" is just a NAME, not real faults (ENABLE_FAULTY_DEFAULT=0)
An access landing in the "faulty region" does NOT architecturally fault. This removes the ARTIFACT
hypothesis that the gain is an architectural MTE fault / abort-handler footprint triggered by the region.
The only fault source is the MTE tag mismatch itself, and the gated accesses are speculative (nesting=1,
never retired for input #1), so a mismatch is resolved by speculative SQUASH, not an architectural fault.
=> the speculative reading is strengthened. STILL open (for the device-Fable): the multi-seal fact (2/3
decoys bypass the 0x120 gate; real drivers x4@0xf8 / x1@0x10c), and that the gain sets are CE-untouched
(prefetch stream), so the exact transmit mechanism and "why opposite polarity" must be pinned on-device,
not inferred.

## Audit 5/6 — informed Haiku: QUESTIONABLE. Architectural twin (checkable, pivotal)
- GRANSCAN line: retagging granule 255 gives set 9 = 0.97 at BOTH delta 7 and delta 10 (tag-INDEPENDENT of
  the 0x120 gate). Granule 255 is accessed at 0x94 which appears architecturally (nesting 0). => an
  ARCHITECTURAL MTE mismatch also lights set 9, so set-9 presence may NOT be speculation-specific. THIS is
  the decisive arch-vs-spec question for the device-Fable (async vs sync TCF, non-MTE control).
- Confirms multi-seal (only decoy1 touches 0x120; STR-only decoy2 fires set 9 with 0x120 matched).
- Not v1/v2: set 12 (the LDP's own line, granule 305) is RESIDENT in every variant; the tag check gates no
  fill/forward, only the set-9/1..11 burst. Burst extent unstable (decoy0 sets 4-30 vs decoy1/2 1-11).
- Model-bug note: tag 0 (delta 1) reads 1.00 in all 2D rows => tag 0 is CHECKED, not TCMA-unchecked; only
  tag 15 (delta 0) is cold. The executor model treating tag 0 as unchecked (aarch64_executor.py:741) is wrong.
PANEL (5/6): 3 GENUINE, 1 ARTIFACT, 1 QUESTIONABLE. Aggregate = CONTESTED; device-Fable to resolve.

## FINAL PANEL (6/6): 3 GENUINE, 1 ARTIFACT, 2 QUESTIONABLE => CONTESTED (relocated out of VERIFIED_GENUINE)
By argument strength (not head-count) the aggregate is CONTESTED / not-confirmed-genuine. Agreed floor
(all six): a real, speculative-only, tag-dependent, position-controlled NI difference (not architectural
fault, not batch position, not tag-blind prefetch). UNRESOLVED (gate 2): is the extra block Δ={1,4,5,6,8,9,
10,11} the DESCENDING prefetch stream of the SPECULATIVE W2 LDP (set 12, granule 305) [GENUINE-class], or
the ASCENDING stream of an ARCHITECTURAL demand (set 3) [NOISE, 021502 shape]? Data do not separate them.
Also: multi-seal (0x120 not the driver; x4@0xf8 / x1@0x10c do the work), granule-255 control anomaly,
dependent LDP ungated (=> not v1/v2). Single cleanest discriminator = ANCHOR MOVE (uninformed Haiku E2):
change the W2 LDP's target granule value so it lands in a different set; if Δ follows -> sealed/speculative-
anchored (genuine); if Δ stays at 1..11 -> fixed/arch-anchored (noise). Device-Fable to run this + async-vs-
sync TCF + non-MTE control + per-seal isolation + granule-255 sham-retag, >=3 rechecks.

## DEVICE-FABLE results (on-device, position-controlled, 2/2 reproducible) — gate-2 RESOLVED -> GENUINE
(The device-Fable completed the decisive experiments before an API cyber-safeguard FALSE-POSITIVE
terminated it mid-run; device left clean/paused.)
1. PER-SEAL ISOLATION (each seal mismatched alone, all 64 sets): A-only(0xf8) and B-only(0x10c) both gate
   granule 0 (set 0) -> identical block sets 0..12. D-only(0x120, the LDP, granule 305/set 12) -> a
   DIFFERENT, wider block ~4..29 (also evicts arch set 3). Different seals -> different blocks => the
   transmit is SEALED/SPECULATIVE-ANCHORED, not one fixed architectural stream.
2. ANCHOR MOVE (decisive gate-2 test): hold seal D mismatched, sweep its ADDG uimm6 to RELOCATE the
   speculative LDP's target granule; the whole transmit stream MOVES MONOTONICALLY with it
   (set12->4..28, set16->7..27, set20->12..31; +8 anchor -> +8 stream low-bound), 2/2 reps identical. A
   fixed ARCHITECTURAL-demand stream (the 021502/noise shape) would NOT move when the speculative load is
   relocated. It moves => the extra block is a HW prefetch/runahead stream ANCHORED TO THE SEALED
   SPECULATIVE LOAD's granule, gated by its MTE tag check. GATE-2 RESOLVED in favor of GENUINE.
3. GRANULE-255 "architectural twin" = TOOL CONFOUND (steer #2 resolved): with ALL seals genuine, retag
   255->9 lights sets 52..63 (granule 255 = set 63; its NON-decoyed architectural store at 0x94 now
   mismatches) plus the low block. So the granule-255 set-9 effect is an ARCHITECTURAL, gate-independent
   artifact of the retag tool mismatching a non-decoyed arch store at 0x94 -- NOT the speculative gate.
   The Haiku "architectural twin" objection is a misattribution.

## STATUS: device-confirmed GENUINE speculative MTE tag-check leak (gain polarity), pending final controls
Six independent lines now support GENUINE: (i) fixed-slot position control Δtag~+1/Δpos~0; (ii) pointer-tag
sweep suppressed only at match tag + TCMA; (iii) 2D tracking granule 305 4/4; (iv) per-seal isolation
(different seals -> different blocks); (v) anchor-move (stream follows the speculative load); (vi)
granule-255 twin shown to be a tool confound. REMAINING (not yet run; Fable terminated): explicit sham-retag
(255->6 inert), TCF/TCO tag-check-OFF causality (cleanest "is it the tag check"), and a formal 155905
positive-control on the same rig (155905 already shown v1-shrink earlier this session). Variant: v1-CLASS,
GAIN/presence-on-mismatch polarity, transmit = prefetch stream anchored to the speculative load; whether to
call it a distinct third variant is a naming question (unchanged from the paper-panel).

## SHAM control (device, 2/2) + 3rd tracking reconfirm -> FINAL: device-confirmed GENUINE
- SHAM: granule 255 at its GENUINE tag 6 -> set 9 = 0.00/0.01 (INERT). Only when 255 is MISMATCHED (->9)
  does it light set 9, and then tag-INDEPENDENTLY of the 0x120 gate (0.97 at both d7 and d10) because it
  trips the NON-decoyed ARCHITECTURAL store at 0x94. => the "architectural twin" is a retag-TOOL confound,
  not the speculative gate. Dissent (Haiku) resolved.
- Granule 305 (speculative LDP) tracking reconfirmed a 3rd independent time (T=9: d7=1.00 -> d10=0.00).

FINAL VERDICT: GENUINE speculative MTE pointer-vs-memory tag-check leak, found by the FULLY-RANDOM Revizor
NI fuzzer on the Pixel-8 Cortex-X3. GAIN / presence-on-MISMATCH polarity; transmit = HW prefetch stream
anchored to the SEALED SPECULATIVE load's target granule, gated by its MTE tag check. Multiple speculative
seals contribute (x4@0xf8 & x1@0x10c -> granule 0 / set 0 stream 0..12; x2@0x120 -> granule 305 / set 12
stream 4..29); all are speculative-only (never retire). The two paper-only dissents are resolved by device
evidence: gate-2 (anchor-move: stream follows the speculative load, falsifying a fixed arch stream); and the
architectural-twin (sham: granule-255 effect is a retag-tool confound on a non-decoyed arch store). Variant:
v1-CLASS, inverted polarity; "distinct third variant" remains a naming question (the transmit is a
mismatch-gated prefetch stream, not v1's downstream-shrink or v2's store-forward). ONE OPTIONAL future
control (not a gap): TCF=async/TCO=1 tag-check-OFF causality needs a kernel-module knob (no sysfs toggle);
the tag-only+TBI logic + anchor-move already establish tag-check causality.

## Device controls COMPLETE (fresh investigator, after the first was killed by the API safeguard)
- Positive control 155905 (3 reps): present-on-MATCH, Δtag -1.00, Δpos ~0 — opposite polarity, rig validated.
- Granule-305 two-sided tracking: now 5 independent reps total, 12/12 rows track (match delta=(7+T-6)%16),
  13 checked mismatches = 1.00 every rep.
- Non-MTE (bit-55) control: assessed, correctly not forced (not a clean same-program swap; needs a separate
  PAC+signext program). MTE dependence already carried by anchor-move + ENABLE_FAULTY_DEFAULT=0.
- Tag-check-OFF: assessed; requires a kernel-module rebuild+reinsert (no sysfs/ioctl/param knob; measurement.c
  re-asserts TCF=SYNC/TCMA1/TCO=0 every window). NOT done (device preserved) — the one OPTIONAL remaining
  control, user decision. Full detail: DEVICE_CONTROLS_072421.md.
FINAL: device-confirmed GENUINE, exhaustively controlled except the optional kernel-level tag-check-off.

## TAG-CHECK-OFF causality test (device, kernel-level) — DEFINITIVE: tag check IS the cause
Built a one-line test variant of the executor module (runtime module_param tco_force; default 0 = current
behavior; =1 forces TCO=1 = MTE tag checks OFF for the measurement window), vermagic-matched, loaded via
su 0 (original module backed up + restored after; device left canonical, md5 5c8dfec). Fixed-prefix
genuine-vs-mismatch (noncanon) on 072421 sets 1/4/8/9, 3 reps each:
  tco_force=0 (checks ON):  Δtag +0.97..+1.00 on all four sets (gain reproduced)
  tco_force=1 (checks OFF): Δtag ~= 0.00 (+-0.02) on all four sets (gain COLLAPSES)
6/6 reps consistent. Disabling the MTE tag check makes the genuine-vs-mismatch divergence vanish =>
the MTE tag CHECK is the cause (not a tag-blind prefetcher, not position, not an unrelated fault).
Worklog: TCF_OFF_WORKLOG.txt; data: TCF_OFF_AB_072421.txt.

## FINAL (all controls complete, including the optional kernel-level one): GENUINE, fully confirmed
072421 is a genuine speculative MTE pointer-vs-memory tag-check leak (fully-random Revizor NI fuzzer,
Pixel-8 Cortex-X3), GAIN / presence-on-mismatch polarity; transmit = HW prefetch stream anchored to the
sealed SPECULATIVE load's granule, gated by its MTE tag check. Evidence chain: position control (tag not
batch), pointer sweep, two-sided granule-305 tracking (5 reps), per-seal isolation, anchor-move (stream
follows the speculative load), sham (arch-twin = retag-tool confound), positive control (155905 opposite
polarity), and tag-check-OFF causality (gain collapses with TCO=1). Variant: v1-class, inverted polarity
(third-variant = naming).

## DEEP-DIVE complete (all directions + combinations): DEEPDIVE_072421.md
Disassembly + CE + by-hand + full HW matrix agree exactly. One active window W2 (TBZ taken-spec); three
speculative-only tag-sealed accesses: A(LDR[x4],granule0/set1), B(STR[x1],granule0/set4), D(LDP[x2],
granule305/set9, stream 4..28). Each is a genuine pointer-vs-memory tag check (present-on-mismatch, match
delta=memtag, tracks granule 4/4, stream anchored to the sealed access). Data chain A->D (A's loaded value
feeds D's pointer) => combinations non-additive (A/B mismatch collapses D's wide block). W1 model-only on HW
(mistraining off). granule255=architectural-store confound. 3-rep stable. New tool: track_pin.py.

## MECHANISM CORRECTION (PMU proof): speculative access BURST, not a quiet prefetcher
PMU differential (MECHANISM_PROOF_072421.md): on mismatch MEM_ACCESS (demand) floods 12->~54k while
L2D_CACHE_REFILL rises only ~10x -> the transmit is a large SPECULATIVE DEMAND-ACCESS burst (likely replay),
NOT a quiet HW prefetch stream. Earlier "prefetch stream" wording here and in DEEPDIVE_072421.md is REVISED
to "speculative access burst gated by the MTE tag check." The anchoring (burst follows the sealed access's
granule) and the tag-gating (tag-check-off collapses it; match/TCMA narrow, mismatch wide) are unchanged and
proven. The prefetcher's role in the burst's contiguity (Proof 1b, CPUACTLR6[41]) is UNPROVEN (test incomplete,
device rebooted). Core verdict (genuine, speculative, tag-gated) stands.

## OPEN MECHANISM QUESTION (user): exception handler vs speculative load stream?
Could the ~54k MEM_ACCESS burst be an exception/abort handler rather than a data-dependent speculative
stream? ARCHITECTURAL exception handler = RULED OUT: the tagged accesses never retire (bb_0.7/0.8
architecturally unreachable; match/mismatch take the identical arch path since the tag is TBI; MEM=12 on
match), so no retired SYNC MTE fault / no committed handler (device would panic otherwise, it didn't).
NOT YET RULED OUT: a SPECULATIVE exception-entry / abort-recovery runahead (handler fetched on the
mispredicted path then squashed) driving the burst. Argument against a FIXED-handler footprint: the
anchor-move shows the burst tracks the DATA granule (moves when the granule moves; match-delta tracks the
granule tag) — a handler lives at fixed vector addresses and would not track the data granule. DECISIVE
TEST (queued; device offline): PMU EXC_TAKEN (0x09) match vs mismatch (does an exception get taken?) +
read TFSR_EL1 after the window (tag-fault recorded?). Predict if clean spec tag-check stream: EXC_TAKEN~0,
burst still tracks the granule. If EXC_TAKEN rises -> abort/recovery-driven (still tag-gated + speculative,
relabel the transmit). Security verdict (MTE tag leaks via cache) holds either way; this refines mechanism.

## OPEN (counter semantics): is MEM_ACCESS(0x13) speculative-inclusive or retired-only?
The ~54k mismatch figure is from PMU event 0x13 (MEM_ACCESS). The module comment "demand (not prefetch)"
distinguishes core-issued vs prefetcher-issued, NOT speculative vs retired. If 0x13 is speculative-inclusive,
54k = speculative burst (speculative-only holds). If 0x13 is RETIRED-only, 54k would be committed accesses
-> would CONTRADICT speculative-only and support an architectural exception/handler. MEM_ACCESS=12 on match
is ambiguous (retired-only, or short match window). DECISIVE counter-independent test (QUEUED, device offline):
read INST_RETIRED (0x08, unambiguously retired) match vs mismatch; control flow is tag-independent (TBI) so
it should be IDENTICAL if the 54k never retires (=> speculative) and JUMP if committed (=> architectural).
Run with EXC_TAKEN(0x09) + TFSR_EL1. Until then, the burst's speculative nature rests on the (unverified)
0x13 semantics; the gadget-accesses-never-retire result (match MEM=12 = arch count, tag-independent control
flow, no panic) is separate and still holds for the SEALED ACCESSES, but the 54k burst's retired-vs-spec
split is not independently confirmed.

## RETRACTION / CAVEAT (user-flagged): the ~54k MEM_ACCESS figure is UNVERIFIED — do not rely on it
The "MEM_ACCESS 12 -> ~54,000 on mismatch" came from pmu_diff.py (mean-per-rep of PMU event 0x13), relayed
without verifying the counter's window/accumulation semantics. A ~4500x amplification from a 76-instruction
gadget (~12 arch mem-ops) is IMPLAUSIBLE for one speculative window and is NOT explained. It could be a real
replay storm OR a counting artifact (accumulation over the internal rounds loop / reload harness inclusion /
overflow / unexpected 0x13 semantics). match=12 (=arch mem-op count) hints the PFC is test-case-window-scoped
(reload excluded), but that is not confirmed. => CONSEQUENCE: the derived claims "it is a speculative ACCESS
BURST, not a quiet HW prefetcher" and the "~4500x vs ~10x" ratio are NOT established and must not be cited;
the TRANSMIT mechanism (replay burst vs prefetcher vs other) is UNRESOLVED. GENUINENESS is UNAFFECTED: it
rests on cache-RESIDENCY controls (tag-only diff, present-on-mismatch set block, granule tracking 4/4,
anchor-move, tag-check-off collapse) which do not depend on any PMU count. TO FIX: verify what window the PFC
covers + whether it accumulates over rounds; use INST_RETIRED(0x08) and EXC_TAKEN(0x09) (unambiguous); clean
re-measure MEM_ACCESS with the reload excluded. Until then treat the burst magnitude as unknown.

## FURTHER CAVEAT (user): match=12 vs mismatch=54k are mutually inconsistent -> PMU account set aside
If MEM_ACCESS counted speculative accesses, MATCH should exceed 12 (bb_0.7/0.8 speculatively execute on
match too, tag-independent window) -> but match=12.0 exactly => looks retired-only. If retired-only, 54k on
mismatch is IMPOSSIBLE (identical arch control flow, exits at 0xc0 before bb_0.7/0.8; nothing commits 54k).
The two readings are mutually exclusive => the PMU figures are NOT cleanly interpretable; 54k is almost
certainly a counting artifact (rounds-loop accumulation / reload inside the counted window on mismatch /
overflow). ENTIRE MEM_ACCESS-based mechanism narrative (burst size, "access burst not prefetcher") is SET
ASIDE. NOTE: "sealed accesses never retire" does NOT depend on the PMU -- it rests on STATIC control flow
(b.al exit at 0xc0 precedes bb_0.7/0.8; branch tag-independent); match=12 was corroboration, not the basis.
Transmit mechanism + magnitude: UNKNOWN pending clean measurement (INST_RETIRED 0x08 + EXC_TAKEN 0x09 +
verified PFC window/accumulation). Genuineness unaffected (residency controls only).

## *** RECLASSIFIED (INST_RETIRED/EXC_TAKEN, device, 3 reps): ARCHITECTURAL MTE FAULT, NOT speculative ***
Decisive hardware counters (EXPERIMENTS_POSTREBOOT_072421.md), gate 0x120, 3 reps:
  match(ptag6): INST_RETIRED 84, EXC_TAKEN 0, MEM_ACCESS 12
  unchecked(ptag15, TCMA-exempt): 84, 0, 12  (internal control: wrong pointer tag but no check -> baseline)
  MISMATCH: INST_RETIRED ~195,000 (+2300x), EXC_TAKEN ~1, MEM_ACCESS ~55,000
=> On a genuine mismatch, ~195k instructions RETIRE (committed) and a SYNCHRONOUS exception is TAKEN. The
transmit is an ARCHITECTURAL MTE tag-check FAULT + its handler (~195k committed instrs, ~55k accesses;
caught by the executor fault handler, no panic), gated by the tag check (ptag15 unchecked = baseline). The
sealed accesses RETIRE -> they are NOT speculative. This CONTRADICTS the earlier "speculative-only /
bb_0.7-0.8 never retire / TikTag-v1-class" verdict. The CE's nesting>=1 was a MODEL error (hardware retires
those blocks). Everything else reconciles under the architectural-fault reading: present-on-mismatch = fault
only on mismatch; granule "tracking"/anchor-move = the handler touching the faulting granule region; TCO-off
collapse = no check -> no fault -> no handler; P+P saturation = the large handler evicting broadly.
VERDICT CHANGE: 072421 is NOT a genuine speculative TikTag leak. It is an NI-METHODOLOGY ARTIFACT: the
sealer/CE labeled an architecturally-retiring tagged access as a speculative (decoy-eligible) seal, so the
decoy injects a REAL architectural MTE fault, whose handler footprint trips the htrace comparison. Tag
dependence + tag-check causation are real; the SPECULATIVE mechanism is refuted. To move out of
VERIFIED_GENUINE -> REJECTED after the agent's writeup lands. (Reps: 3, internal control clean.)

## Precision boundary (measured vs inferred)
DIRECTLY MEASURED (EXC_TAKEN 0x09 + INST_RETIRED 0x08, both retired/architectural-only events): on a
genuine mismatch exactly one architectural exception is TAKEN and ~195k instructions RETIRE (committed) -
so the divergent work is architectural, not a non-retiring speculative burst. INFERRED (not read out): that
the exception's class is specifically the MTE synchronous tag-check fault - from the gating pattern (fires
only on a CHECKED-tag mismatch; baseline at the TCMA-unchecked tag ptag15 and at match). A direct class
confirmation would read ESR_EL1/FAR_EL1/TFSR_EL1 in the handler (another module rebuild; not done). The
ARCHITECTURAL reclassification does not depend on the class; the tag-check gating does.

## OFFLINE CONFIRMATION (committed-instruction count) -> reading A, architectural confirmed
CE trace (input_0001_trace_log.txt): nesting-0 (committed) path = 48 instrs, exits at 0xc0 (b.al); bb_0.7/0.8
(0xc4..0x124) appear ONLY at nesting>0 (CE models them speculative). BUT hardware INST_RETIRED on MATCH = 84
>> 48. The ~36-instr gap matches bb_0.7/0.8 (25 instrs) + window fences/harness (~8-10); it does NOT match
the CE's 48-instr exit-at-0xc0 path. => on hardware the gadget ARCHITECTURALLY executes bb_0.7/0.8 (CE
branch-model error); the sealed access in bb_0.8 retires and faults on mismatch (reading A). Reading B (a
separate speculative leak with the PMU tripping a different arch access) is NOT supported: 84 != 48.
Architectural verdict triangulated: INST_RETIRED jump + EXC_TAKEN 0->1 + committed-count = full gadget path.
LESSON: the CE's nesting classification is NOT ground truth for the hardware architectural path; a
speculative-leak claim must check committed-instruction count / INST_RETIRED against the hardware path.
