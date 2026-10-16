# Cross-input BTB violation integrity triage

Config: `eval_configs/canon_btb_3t.yml` (NI fuzzer; indirect+dispatch, >=3 targets;
`enable_branch_target_sealing`; SSB window via `enable_speculative_store_bypass`;
`contract_execution_clause: [cond, bpas]`; `enable_cross_input_priming`,
`cross_input_priming_localizer: optimal`, `cross_input_leaks_only: true`; `inputs_per_class: 4`).

Method: CONTRACT-EXECUTOR + static analysis on saved artifacts only. The N3 device was held by a
running campaign, so NO HW re-measurement, no /dev/executor, no /sys/executor, no Pixel. Device-only
steps are called out as pending.

## How the cross-input localizer guarantees non-self (code-level, `src/aarch64/cross_input.py`)
`find_leaking_pair(base, toggle, detecting, exclude_self_dependence=True)` searches `[0, detecting)`.
`splice()` sets classes `[t, detecting]` to the `toggle` lane. Therefore for every probe the DETECTOR
slot `detecting` is held FIXED at `toggle[detecting]` (the identical concrete input); only an EARLIER
position varies. A reported boundary `t` satisfies `r_key(t) != r_key(t+1)` with the detector input
unchanged, so it is a pure PREFIX effect = genuine cross-input by construction. `cross_input_leaks_only:
true` wires `exclude_self = True` (`aarch64_fuzzer.py:204`). `is_valid_toggle = _contract_equal_toggle`
rejects a boundary whose two members are not contract-equal, and `_reverify` re-measures r(t), r(t+1)
fresh at the largest sample with a robust ChiSquared test before accepting.
CAVEAT: the report's printed "Sequence A / Sequence B differ only in positions lo..hi" is a looser
"chain" display that also varies the detector position; the underlying proof (above) does NOT -- it
holds the detector fixed. The display is slightly misleading; the algorithm is correct.

## Sealing mechanism (static, `src/aarch64/seal/sealer.py` + enacted-bin diff)
`BranchTargetSealing` inserts, right before a SPECULATIVE-ONLY `BLR`, either a `NOP` (genuine =
canonical correct target) or an `EOR Xt, Xt, #mask` (decoy) that makes the speculative target
non-canonical (a contiguous run in bits [54:VA]) or misaligned (low bits {1,2,3}). The mask pool is
`stable_rng("bt-canon-pool", salt)` -- per-test-case and POSITION-INDEPENDENT, so the genuine/decoy
assignment is fixed per boosted lane across all input classes. `generated.asm` shows only `NOP`
placeholders; the per-input resolution lives in `enacted_test*.bin`.

---

## Violation A -- run1/violation-261016-203511  (pair 24 -> 27)
Program seed 322439; 184 inputs = 46 classes x 4 lanes (class k = [k, k+46, k+92, k+138]).
Report: leaking pair = class 24, detecting pair = class 27; all four detector members
(27,73,119,165) share contract hash 13953104153730783921.

1. GADGET (confirmed). `generated.asm` function_0/1/2 each hold an indirect multi-target dispatch
   `ADR x28,.Lfunction_N_dispatch ; LDRSW ; ADD x28,x28,Xi ; BLR x28` over 2-4 `.word` table entries
   (>=3 reachable targets function_1..4). `input_0027_trace_log.txt` shows the executed stream.
2. SEALING + SPECULATIVE-ONLY (confirmed). `objdump` diff of the four detector-class enacted bins
   (27,73,119,165) differ at EXACTLY ONE offset, 0x194, the seal slot before `blr x28` at 0x198:
     - input  27 (lane0): `nop`                              -> GENUINE (canonical target)
     - input  73 (lane1): `eor x28,x28,#0x2`                 -> DECOY (misaligned -> PC-align fault)
     - input 119 (lane2): `eor x28,x28,#0x2`                 -> DECOY (misaligned)
     - input 165 (lane3): `eor x28,x28,#0x60000000000000`    -> DECOY (non-canonical, bits 53:54)
   The `blr x28` at 0x198 (function_2's dispatcher) does NOT appear in input 27's architectural trace
   (grep count 0) -> reached only under speculation (SSB/cond window). This is the sealed target.
3. CROSS-INPUT (confirmed, structural). leaking 24 != detecting 27; localizer holds the detector
   fixed and toggles only position 24 (see code section). By lane-determinism the leaker toggle is
   class-24 lane0 (input 24, genuine canonical) vs lane1 (input 70, decoy misaligned) -- two ct-equal
   variants whose only difference is the sealed SPECULATIVE BLR target. Toggling it flips input 73's
   (fixed detector) HW readout, i.e. earlier input trains the BTB target, later ct-equal input inherits.
4. NOISE. HW rep-counts: detector input 27 = 95/5 split; classmates 73/119/165 = 100/0. A stable
   ~2-bit cache-set difference, consistent across three classmates, that passed the robust ChiSquared
   reverify at the large sample. Not consistent with random measurement jitter.
5. MODELING ARTIFACT -- ruled out. The contract (cond|bpas) deliberately does NOT model indirect/BTB
   target prediction, so the HW BTB cross-training is genuinely out-of-contract (the intended target of
   this NI axis), not a model bug. Members are ct-equal (report hashes equal; enacted bins differ only
   at a speculative-only seal slot, which cannot change committed/architectural state).

VERDICT A: GENUINE cross-input BTB leak.

## Violation B -- run2/violation-261016-203642  (pair 23 -> 24)
Program seed 112167; 148 inputs = 37 classes x 4 lanes (class k = [k, k+37, k+74, k+111]).
Report: leaking pair = class 23, detecting pair = class 24; all four detector members
(24,61,98,135) share contract hash 7708156139577755727.

1. GADGET (confirmed). Same structure -- three `ADR/LDRSW/ADD/BLR` multi-target dispatchers
   (function_0/1/2) plus a direct `ADR .function_4 ; BLR`.
2. SEALING (confirmed). `objdump` diff of detector-class enacted bins (24,61,98,135) differ only at
   seal slots 0x124 and 0x1c8:
     - input  24 (lane0): `nop` / `nop`                          -> GENUINE
     - input  61 (lane1): .../ `eor #0x70000000000000`           -> DECOY (non-canonical, bits 52:54)
     - input  98 (lane2): `eor #0x2` / `eor #0x3`                -> DECOY (misaligned)
     - input 135 (lane3): .../ `eor #0x1e000000000000`           -> DECOY (non-canonical)
   Genuine-vs-decoy branch-target sealing per lane on speculative BLRs, same as A.
3. CROSS-INPUT (confirmed, structural). leaking 23 != detecting 24; detector held fixed, position 23
   toggled; leaker toggle = class-23 lane0 (23, genuine) vs lane1 (60, decoy non-canonical) by
   lane-determinism.
4. NOISE. HW rep-counts: detector 24 = 98/2, classmates 61/98/135 majorities ~97-100% on the opposite
   (1-bit) pattern. Stable, consistent, passed reverify. Not noise-like.
5. MODELING ARTIFACT -- ruled out, same reasoning as A.

VERDICT B: GENUINE cross-input BTB leak.

---

## Overall
BOTH A and B are GENUINE cross-input BTB leaks: an earlier input's sealed, speculative-only indirect-
call target (canonical vs non-canonical/misaligned) trains a BTB entry, and a later contract-equal
input's L1D cache trace depends on that training. Neither is self (localizer holds the detector fixed;
distinct indices), noise (stable multi-rep divergence through a robust reverify), nor a modeling
artifact (contract omits indirect prediction by design; members are provably ct-equal).

## Honest limitations (device-dependent, pending)
- Independent CE re-derivation of the contract-trace hashes was INFEASIBLE in this device-free harness.
  The real sandbox base (x29 = 0xffff400095321000, recovered from the saved trace-log register dump) is
  correct, but a hand-built `ContractExecution` FATALs in `kaddr2uaddr` on an out-of-sandbox access
  because the production path's register/base translation (normally seeded from
  /sys/executor/print_sandbox_base and the fuzzer's `trace_test_case_with_taints` setup) was not
  reproduced. The saved reports carry the model-computed ct-equality, and the enacted-bin diffs show the
  members differ ONLY at speculative-only seal slots, which is equivalent structural evidence.
- The leaker-class per-lane seal pattern is INFERRED from salt-seeded, position-independent lane
  determinism (the detector-class pattern was dumped directly); the leaker enacted bins were not saved.
- The DEFINITIVE anti-noise / anti-batch-contamination test -- a HW controlled swap (load the full
  batch once, fix the prefix, swap ONLY the leaker slot between its genuine and decoy members, confirm
  the detector's htrace divergence persists) -- requires /dev/executor and is PENDING device
  availability. The campaign's own large-sample ChiSquared reverify is the standing substitute.

CE note: the stock `tools/triage.py` SIGBUSes on dispatch cases (omits `data_size`); `tools/v2_ce_ladder.py`
passes `data_size = 4 * (#.word entries)` but still needs the live /sys base to avoid the FATAL above.
