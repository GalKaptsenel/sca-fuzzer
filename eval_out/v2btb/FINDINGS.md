# Spectre-v2 (indirect branch) over the real BTB — which-target, NO sealing (N3, eval-measurements)

The paper's Fig-(a) "what priming cannot find" gadget: a target *returns* a value and the victim loads
from it, so WHICH target the BTB predicts selects the L1D set. Pure which-target leak — all targets
canonical, access no memory themselves, no branch-target sealing, no PAC. Detected by generalized
(cross-input) priming; standard priming drops it. Distinct from the canonicality-over-BTB leak (which
leaks the target *address* canonicality via the NI seal).

## The CT-SEQ insight (what made it work)
Generalized priming confirms a leak by toggling a **within-class, contract-equal** variant that changes
BTB training (`is_valid_toggle` rejects non-ct-equal boundaries). Canonicality gets this toggle from the
NI *seal* (genuine canonical vs decoy non-canonical, same arch trace). Pure v2 has no seal, so the toggle
must come from the basic fuzzer's **boosting**, which only varies **untainted** bytes (those that do not
change the contract trace).

- Under `[cond,bpas]` the bpas/SSB speculation IS modeled, so the speculative dispatch target lands in the
  contract trace → the target-selecting bits are **tainted** → boosting holds them fixed → every class
  member dispatches to the same target → no toggle → always "no leaking pair → false positive".
- Forcing **CT-SEQ** (`contract_execution_clause: [seq]`) removes all modeled speculation, so the whole
  speculative dispatch/target is beyond-contract and its selecting bits are **untainted** → boosting
  varies them → each class has members that train the BTB to DIFFERENT targets = the which-target toggle.

The HW still speculates (SSBS=1 regime, SSB/v4 window open); CT-SEQ only changes the reference model.

## Template result (templates/spectre_v2_btb.py, eval_configs/v2_btb_seq.yml)
`tfuzz -c v2_btb_seq.yml -t spectre_v2_btb.py -n1 -i100` → **Violations: 1** in 57s.
CONFIRMED cross-input leak, **leaking pair 0 → detecting pair 11**.
Report: two ct-equal sequences A=[0..11], B=[200..211]; differ only in positions 0..11; detecting pair's
HW trace differs → input 0's BTB residue changes input 11's readout. Kind: cross-input (no self).
**Deterministic reproduce 3/3** with pinned seeds (program 716170, input 2522819617) → identical pair 0→11.

Three target sets observed across the batch: 8 (f0, off 512), 16 (f1, off 1024), 32 (f2, off 2048); the
diverging lane carries an extra target set = the BTB-predicted (cross-input) target.

## Required regime (same as all cross-input BTB work)
ssbs=1, enable_pre_run_flush=0, enable_phr_flush=0, enable_view_rotation=0, unpinned (pin=-1); P+P.
fuzzer: basic (boosted random-data lanes), inputs_per_class 4, enable_cross_input_priming +
cross_input_leaks_only, indirect_call_probability 1, dispatch_call_probability 1, NO sealing.

## Random (no-template) path — IN PROGRESS
eval_configs/v2_btb_seq_rand.yml (CT-SEQ, 3 targets, no seal, low avg_mem_accesses), fuzz -i50. Running
(eval_out/v2btb/rand1). RESULT PENDING — not yet claimed until random Revizor actually finds it.

## TTD — PENDING (R=20 via eval_scripts/run_ttd_tmpl.sh on the template; random TTD from rand run).

## Random (no-template) path — mechanistic root cause + decision (2026-10-17)
Candidates ARE flagged every TC under CT-SEQ, but the localizer rejects them ("no leaking pair"). Root cause:
the basic fuzzer boosts by varying only UNTAINTED bytes, so a class's lane-members are ct-equal. For an
ARCHITECTURAL indirect dispatch, ct-equal => same target => the generalized-priming splice has no toggle to
flip => rejected. Only a SPECULATIVE-ONLY dispatch (target untainted, so boosting varies it) is localizable;
in random code that needs a data-hazard (SSB)-gated dispatch, which the template engineers but random
programs rarely produce. This is structural, not bad luck — more tries/CF-richness do not change it.
- Confirmed: template under the random config confirms (leaking 4->5, 13.6s) => config is capable.
- Confirmed: the NI-seal vehicle (controlled ct-equal toggle on each spec-only slot) finds it in 1 TC/29.8s
  (ni probe) — common, like canonicality. That is the lever; without a seal the toggle is uncontrolled.
- Plain-fuzzer random campaigns rand1-4 (ipc4-6, linear->richer CF, -i50/100): many programs, 0 confirmed.

**USER DECISION: strictly NO seal.** We do NOT add a which-target seal. The template is the reliable pure-v2
(no-seal) demonstrator; the random path stays the plain fuzzer with many tries (accepted as rare). A random
hit, if found, is a genuine chance SSB-gated spec-only dispatch.

## Random final best-effort (2026-10-17)
rand_final (rand4 richer-CF, -i50, CT-SEQ, no seal): 201 programs in 1 h, Violations=0, confirmed cross-input=0. Combined with rand1-4 => no-seal fully-random pure-v2/BTB NOT surfaced (structurally rare, root-caused). Template remains the reliable demonstrator (TTD 20/20).
