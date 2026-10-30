# Speculative-exception fuzzing — evaluation & hardening log

Branch `speculative-exception-fuzzing`. Persistent work log for the six-task autonomous run
started 2026-10-30. Each experiment is logged BEFORE (plan + exact command) and AFTER (result +
artifact path). Raw artifacts live under `docs/spec_exc_eval/results/`.

Device: Neoverse N3, `/dev/executor` live (local kernel module). PAC is HW-testable here; MTE is
CE-only (no FEAT_MTE2 on this box).

## Task list (user, 2026-10-30)
1. Clear explanation of the bug and of the fix.
2. Sign-extend + PAC-bit fuzzing — find the AUT* (bit-55) leak.
3. ct-bpas: where bpas does not speculate → decoy sealings; where bpas speculation crosses into a
   branch not architecturally taken → the class is identical (arch flow always genuine; the
   speculative flows under bpas, when crossing a non-arch-taken branch, carry the EXACT SAME value
   across the class — genuine or a shared decoy, but identical).
4. Cross-input priming + canonicality decoys → find the canonicality leak, with and without a template.
5. Only then: clean the code (no leftovers/deprecated/debug, comment discipline, all guidelines).
6. Only after commit+push: paper — maximize bits/s rate of the AUT* and canonicality channels while
   keeping a high success rate; make the comparison fair (vary only medium + technique); produce
   tables that support a conclusion, grounded in how the literature compares covert/side channels.

### Side note (user, 2026-10-30): does GVT improve throughput?
The GVT register-indirection makes the test-case code byte-identical across all inputs of a test
case, so the per-input code **relocation splice** (and any re-JIT it forced) is gone — only the GVT
image / input bytes change per input. Hypothesis: fewer per-input code writes ⇒ higher inputs/s.
To be MEASURED under task 6 (profile + inputs/s with the current GVT path; reason about the retired
splice cost). Logged here so it is not lost.

---

## Task 2 — PAC + sign-ext fuzzing → the AUT* (bit-55) leak

BEFORE (2026-10-30): config `config_pac_signext.yml` (PAC seal_prob 1.0, strip 0.0, enable_signext,
signext_seal_prob 1.0, clause=cond, mode=strict). Launch:
  `revizor-venv/bin/python revizor.py fuzz -s base.json -c config_pac_signext.yml -n 200 -i 50 \
     -w docs/spec_exc_eval/results/t2_pac_signext --timeout 1200`
(no `--nonstop` → stops at the first violation for triage). The mechanism: the speculative decoy
flips the authed pointer's sign-extension / bit-55 so the speculative AUT* canonicality check
diverges, leaking the PAC field via the µarch channel. Triage the saved violation with the
PAC/leak-flow skills; a genuine AUT* leak must survive CE speculative-only accounting + HW reproduce.

AFTER (2026-10-30 00:15): violation found on the FIRST sweep — TC #69, 48s to detection.
Throughput during the run: ~2.2 tc/s, ~1300 traces/s (final burst ~2885 traces/s). Violation at
`results/t2_pac_signext/violation-261030-001505`. Counterexamples: input #26 vs #68, SAME contract
hash 418388102679174604, divergent htraces (#26 differs across its own repeats 379/121, #68=500).
generated.asm has AUTIA/AUTDZB + PACIZA/PACDA + TBNZ/TBZ (conditional branches) — the sign-ext AUT*
shape. Integrity triage DELEGATED to a subagent (CE speculative-only accounting + controlled HW
swap of the leaking slot #26↔#68) to confirm: correct, reproducible, and specifically the AUT*
channel. Verdict pending — recorded here on return. Triage artifacts under `.../triage/`.

**INCIDENT (2026-10-30 ~09:00): the triage subagent crashed/reset the box** during the HARDWARE
phase of the PAC sign-ext triage (uptime 2 min after; `/dev/executor` gone; pstore empty; subagent
transcript lost with the prior session). This is the known PAC box-reset hazard. Recovery: `insmod
revizor-executor.ko` + chmod — device live again. The campaign's OWN run reported "Hardware Tracing
Errors: 0", i.e. the FUZZER's sealed execution of this test case was safe; the crash came from the
subagent's *manual* HW reproduce (a hand-built swap can put a sign-ext/FPAC fault on the architectural
path). REVISED PLAN: triage CE-only first (local, cannot crash), then reproduce on HW ONLY via the
fuzzer's validated sealed path (`revizor.py reproduce`), never a hand-built manual swap for the PAC
sign-ext case. No subagent for the HW step — tight local control.

**ROOT CAUSE of the box crashes (2026-10-30, confirmed): sandbox-base-specific PAC signatures.**
A PAC signature authenticates `base+offset` (the actual runtime pointer, which includes the sandbox
base). The saved GVT images carry the CAMPAIGN boot's signatures. The sandbox base is NOT stable
across reboots (observed `0xffff40005ca61000` → `0xffff40008b821000`). My MANUAL replay of the raw
saved GVT images ran STALE signatures at the new base, so the ARCHITECTURAL AUT*s FPAC → box reset
(both of my crashes). Evidence: reconstructed executor-input serializes BYTE-IDENTICAL to the saved
`.reif` (not a plumbing bug); the CE-only triage never crashed (the CE ABORTS in software on a bad
sig — contract_executor signal-6 — rather than faulting silicon); `revizor.py reproduce` did NOT
crash because it RE-SEALS (re-signs at the current base).
**This is NOT a GVT regression.** The OLD relocation scheme spliced the signature as a MOVK CONSTANT
into the code — equally `sign(base+offset)`, equally base-specific. It "never hit this" only because
(a) within one module load the base is stable, so same-boot replay authenticates, and (b) the proper
reproduce path re-seals (regenerates MOVKs/GVT at the current base) in both schemes. Manually
replaying the old scheme's saved spliced-MOVK code across a reboot would FPAC identically.
RULE: never replay a raw saved PAC artifact (GVT image or spliced code) on HW across a reboot —
re-seal at the current base, or regenerate from seeds. (The first crash was the subagent's hand-built
manual reproduce at the campaign boot; its transcript was lost, so its exact trigger is unrecoverable,
but it is the same family — a stale/incorrect signature on the architectural path.)

AFTER (final): **VERDICT = GENUINE speculative-exception leak via the sealed PAC-auth / sign-ext
(AUT* bit-55) channel** — established from the in-run data (see `.../triage/VERDICT.md`):
contract-equivalent + arch-inputs byte-identical (only the seal differs) + set-11 htrace divergence
0.76 vs 0.00 over 500 reps + 0 hardware tracing errors (arch-safe at the campaign base ⇒ the
divergence was speculative). Independent HW re-observation across a reboot is blocked by the
base-specific-signature property above (re-fuzz from the recorded seeds at the new base did not
reproduce THIS instance — 0 violations — so the specific instance may be base-sensitive; the
phenomenon/capability reproduces: see the capability re-fuzz below).

**CONSOLE-CAPTURED PANIC (2026-10-30 09:50, the user opened the serial console) — deeper root cause.**
`Internal error: Oops - FPAC`; pc in the sandbox; x29=current base; faulting insn `dac11c80` = `AUTDB
x0,x4`; x0=`ff8f40008b822da5` (PAC field does not verify); FATAL IN INTERRUPT (the measurement IPI) ⇒
the AUT* ran on the ARCHITECTURAL path. So beyond the stale-cross-reboot cause, there is a deeper one:
a DECOY forged signature landed on an AUT* the HARDWARE executes architecturally. Decoys only perturb
CE-"speculative" (nest>0) slots; when the CE's speculative classification DISAGREES with HW
architectural execution (base-dependent, via data-dependent branches reading base-dependent PAC'd
values), the forged sig is authenticated for real → architectural FPAC → reset. Genuine baseline never
faults; only the decoy does ⇒ an unlucky-base-free campaign reports 0 hw errors while a re-fuzz at a
different base panics. Both causes reduce to: a wrong PAC sig on the ARCH path FPACs FEAT_FPAC silicon.

DECISION: STOP HW PAC fuzzing with `pac_strip_prob:0.0` (every decoy is a forgeable AUT* → panics at
unlucky bases). The validated SAFE recipe is `pac_strip_prob>0` (leaky variant = XPAC STRIP, never
faults) — [[project_pac_strip_prob_leak_key]]. Total box resets this task: 4 (1 subagent manual, 2 my
cross-reboot replays, 1 re-fuzz at an unlucky base). The task-2 verdict stands on the in-run evidence;
it does NOT depend on any crashing replay. The deterministic-salt change (sealer.py) is kept as a real
reproducibility improvement (same program ⇒ same decoy selection) but is ORTHOGONAL to the FPAC crash
and to reproduce-finding-0 (the latter is Spectre-v1 mistraining washed out by isolated re-measurement).
Memory: [[project_pac_sig_base_specific_replay_crash]].

---

## Task 3 — ct-bpas sealing invariant

BEFORE (2026-10-30): Config `config_canon.yml` (canonicality seals; contract clause `cond` + `bpas`).
The invariant to verify: a slot reached only speculatively (incl. bpas crossing a conditional branch
into a non-architecturally-taken block) is decoy-eligible; a slot reached architecturally is not; and
across one equivalence class the architectural flow is always genuine while the speculative flow is
byte-identical (same genuine-or-shared-decoy value for every input in the class).

Code reading CONFIRMS this holds by construction:
  * `_Resolved.speculative = (spec_nesting != 0)` — arch slots (nest 0) are never decoy-eligible;
    `_eligible()` = speculative slots with alts; `_decoy_subset` only perturbs eligible (speculative).
  * `collapse_key = tuple((value, speculative) per entry)` — the class pins every slot's value AND
    speculative status.
  * `_variants_for`: baseline = `genuine()`; decoy_n(i) seeded by `stable_rng(collapse_key, salt, i)`
    (aarch64_executor.py:852) — NO per-input entropy, so two inputs with the same collapse_key get the
    byte-identical image for every variant. Arch lanes always hold `r.value` (genuine); the decoy only
    touches speculative lanes.

Deliverable: `tests/aarch64_tests/unit_bpas_class_identity.py` (device-gated) asserts (a) arch slots
genuine in every variant + decoys non-trivial, (b) per-variant GVT image identical across a real
same-class input pair. Plus a short cond+bpas demonstration run captured to results/. RUN pending
(device held by the task-2 triage subagent; serialized to avoid concurrent /dev/executor users).

---

## Task 4 — cross-input priming + canonicality decoys → the canonicality leak

BEFORE (2026-10-30): the cross-input BTB canonicality leftover. A branch-target seal marks an
indirect-call target canonical vs non-canonical per boosted lane; a canonical leaker trains a BTB
entry a later contract-equal prober reads → a cross-input leftover that standard priming discards but
`enable_cross_input_priming` localizes.
  * WITHOUT template: `revizor.py fuzz -c config_canon_cip_ni.yml` (enable_cross_input_priming,
    localizer=any, branch_target_sealing, clause cond+bpas) — random programs; priming auto-localizes
    the leaker→prober pair.
  * WITH template: `revizor.py tfuzz -t templates/canon_oracle_btb.py` under the same cross-input
    knobs — three indirect targets at FIXED distinct cache sets; the prober's set-24 fire iff its
    predecessor was canonical is the cross-input signal.
Both runs pending (device serialized). Triage each hit for a genuine cross-input canonicality signal.



### The bug
The MTE seal makes a speculative tag-check fail on the speculative path while the architectural
access stays tag-clean. The seal works by writing a 4-bit allocation tag into the pointer's tag
field [59:56] with a `BFI` on the **base register**, then doing the access.

MTE checks the tag of the **effective address (EA)**, not of the base register. For an
immediate-offset access (`ldr x1,[x0,#off]`) the EA's top byte equals the base register's top byte,
so EA tag == base tag and writing the cell's tag onto the base is correct. But for a
**register-offset** access (`ldr x1,[x0,x2]`) the index register `x2` contributes its own top byte
to the address computation, so the EA tag = base tag + index tag (mod 16). BFI-ing the accessed
cell's tag onto the base register then produced an EA whose tag was `cell + index_tag`, **not**
`cell` — a tag mismatch on the genuine (architectural) access, i.e. a real synchronous MTE tag
fault on hardware. The genuine baseline is supposed to be architecturally clean; this made it fault.

Caught by `unit_mte_ni_tag_safety`, which asserts the genuine baseline is arch tag-clean.

### The fix (commit 0b06406f)
Target the **EA** tag, not the base tag. The base tag that makes EA tag == cell is
`base_tag + (cell - ea_tag)` (mod 16); for an immediate-offset access `ea_tag == base_tag` so it
reduces to `cell`, preserving the old behaviour.

- CE `mte_emulate_retag` (`mte_seal_plugin.c`): reads the base register (`get_rn(access)`) and the
  EA, computes `applied = (base_tag + cell - ea_tag) & 0xF`, and injects `applied` into both GVT
  lanes (the applied lane and the re-converge lane).
- `_resolve_mte` (`sealer.py`): reads the applied base tag the inject set (`value_reg >> 56`),
  derives the index's contribution `index_contrib = (cell - applied) & 0xF`, and offers decoy tags
  `t` such that the **resulting EA tag** `(t + index_contrib) & 0xF` is a genuinely-checked value —
  excluding 0 and 15, which `TCR_EL1.TCMA1` makes Tag-Unchecked (a decoy landing there would behave
  identically to the genuine access, so it is not a valid mismatch).

Net: the genuine access is arch tag-clean for both immediate- and register-offset forms, and every
decoy is a genuine, checked mismatch.

### Related entanglement fix (PAC+MTE, commit fb08646f) — recorded for completeness
On the N3 `tbid1`/`constpacfield` profile the PAC field overlaps the MTE tag/top byte, so signing
the raw tagged pointer ≠ signing the pointer the `AUT*` actually verifies. `_resolve_pac` signs the
**stripped** pointer (`PacSigner.strip`, as the `AUT*` canonicalises it), so AUT* verifies on
silicon. Distinct from the EA-tag bug above but in the same PAC/MTE family.

---

---

## RESOLUTION of the "GVT PAC crash" (2026-10-30) — NO fuzzer/GVT bug

Rigorous conclusion after the full investigation:
- **config_pac_signext sealed fuzzing does NOT crash the box.** Evidence: 60 individual-variant TCs +
  250 full-batch TCs + 5 instrumented TCs = 315+ fresh test cases, full genuine+decoy batches (94-100
  variants/TC, 492 htraces), box stable throughout. Plus the original task-2 campaign and the capability
  re-fuzz both completed. 
- **SW QARMA matches silicon** (unit_pac_hw_differential 3/3) ⇒ genuine signatures are valid.
- **CE is self-consistent** — 77 cases, 0 architectural FPACs for genuine OR decoy (CE SEQ emul=b"").
- **GVT delivery is faithful** — the seal UMOV loads x27 from the GVT correctly (0/13 mismatch);
  strip(ptr) is a no-op on canonical reached pointers; genuine pointer at each reached AUT* is correct.
- **The box crashes were a METHODOLOGY error, not a code bug:** replaying the SAVED violation's GVT
  images on HW ACROSS A REBOOT. PAC sigs authenticate base+offset; the sandbox base changes per reboot
  (observed 5ca61000 -> 8b821000); the saved sigs are stale at the new base ⇒ the ARCHITECTURAL AUT*
  FPACs ⇒ reset. Crashes 2/3/4 = my t2_replay.py; crash 1 = the subagent's hand-built reproduce. The
  fuzzer never replays a saved image — it re-seals at the current base (why fresh fuzzing is clean and
  revizor.py reproduce didn't crash).
- Console-captured panic decoded: `AUTDB x0,x4` (Code: ...8b1d0000=ADD x0,x0,x29 clamp; 0e013ffb=UMOV
  w27,v31.B[0]; b3501b60=BFI x0,x27; dac11c80=AUTDB) — a mem-access PAC seal authenticating a
  stale-signature pointer on the arch path. Consistent with stale-base replay.

**No GVT regression.** Moving the immediate from a spliced MOVK (code relocation) to a UMOV-from-GVT is
faithful. The deterministic-salt idea (reverted) is an optional enhancement for EXACT reproduction of a
saved violation's decoy pick; it is NOT a crash fix. The pac_strip_prob suggestion was a wrong fallback.
Only real limitation: a saved PAC .reif is not HW-replayable across a reboot (stale base-specific sigs);
reproduce by re-sealing (revizor.py reproduce/fuzz) or regenerating from seeds.

---

## Task 4 — cross-input priming + canonicality leak (RESULTS)

WITH TEMPLATE (GENUINE, found): `revizor.py tfuzz -c config_canon_cip_ni.yml -t templates/canon_oracle_ssb.py`
found a violation in 18.5s at `results/t4_canon_ssb/violation-261030-103700`. Counterexamples #38 vs #88,
same contract hash 16354134495979352227, 0 hw errors, all 50 classes effective (nothing filtered). The
leaking set is **set 16 = the template's f2 target** (f1->8, f2->16, f3->32): #38's prober speculatively
ran f2 via a BTB entry trained by a CANONICAL leaker predecessor (set 16 fires), while contract-equivalent
#88 did not. That is the designed cross-input canonicality BTB leftover — stable (100 vs 99/1 reps), box
never reset. (canon_oracle_btb filtered every TC with this config; canon_oracle_ssb is the right template,
per memory project_btb_leftover_template_automatic.)

WITHOUT TEMPLATE (random `revizor.py fuzz -c config_canon_cip_ni.yml -n 300 -i 50`): 0 violations — all
300 TCs fast-path-discarded (0 effective classes). Pure random fuzz did not surface the leaker->prober
structure in 300 TCs; the validated random detection is "tfuzz + scan" (per memory
project_btb_leftover_template_automatic), i.e. it uses a scanner over Revizor output, not bare `fuzz`.
So the without-template case needs the scan tool / a larger budget — OPEN.

### Task 4 no-template ROOT CAUSE (GVT == relocation, confirmed)
The random no-template run found 0 violations with Effective Cls 0.0 NOT because of any GVT issue, but
because `indirect_call_probability` defaults to 0.0 and config_canon_cip_ni.yml never sets it. Verified
locally: a random TC under that config has 0 BLR (indirect calls) in its asm and the sealer creates 0
branch-target seals (value_sealings == []). BranchTargetSealing needs indirect calls
(aarch64_config.py: "Needs indirect calls ... indirect_call_probability > 0"). This is a generator/config
matter, identical for the GVT and the old relocation delivery — the seal mechanism itself works (the
template case produced 50 effective classes + a genuine leak). FIX for the no-template path: set
`indirect_call_probability: 1.0` (+ dispatch_call_probability for distinct target sets) so random TCs
contain BLRs to seal. Re-run: docs/spec_exc_eval/results/config_canon_notmpl.yml, campaign t4_canon_notmpl2.

### Task 4 no-template — DONE (with indirect_call_probability=1.0)
Random `fuzz` on docs/spec_exc_eval/results/config_canon_notmpl.yml (config_canon_cip_ni + indirect_call_probability:1.0
+ dispatch_call_probability:1.0): VIOLATION found (results/t4_canon_notmpl2/violation-261030-114352), Effective Cls 49.7,
only 2 fast-pathed, box clean. Several contract-equivalent inputs (hash 2661331406448718221) with divergent htraces =
cross-input canon leftover (noisier than the template case, as expected for random programs). CONCLUSION: the GVT canon
method is equivalent to the relocation method; the only missing piece was the indirect_call_probability knob. TASK 4 DONE
(template = clean genuine set-16 leak; no-template = violation once indirect calls exist).

## 2026-10-30 (session resume) — TikTag device-exclusivity bug + stderr isolation + PENDING reloc-vs-GVT compare

TikTag "Hardware Tracing Errors"/"malformed input" on the Pixel root-caused: NOT a serialization or
transport code bug. A leftover autonomous campaign (collect_daemon.sh -> revizor.py fuzz
s24_mte_random.yml --nonstop, resurrected every 5 min by cron campaign_boot.sh) was driving the same
/dev/executor. Two processes corrupt the module-global `checkout_region`, so the 324-byte test case
got validated on the INPUT path -> bad magic -> rejected. Fixed by stopping the daemon+fuzz and
commenting out the campaign_boot cron lines (crontab backup saved; re-enable by uncommenting the
#DISABLED-20261030 lines). After that, 90 TikTag TCs ran with 0 HW tracing errors.

Separately fixed a real latent defect (commit 10df5412): ADBConnection.run used adb's exec: service,
which MERGES stderr into stdout, so a device error surfaced as an inscrutable "bad magic" instead of
its real message. Now drives the batch over adb shell-protocol v2 (raw, no PTY): stdout/stderr/exit on
separate framed streams; nonzero exit -> IOError(stderr) -> HardwareTracingError (skip TC). Test:
tests/aarch64_tests/unit_adb_shell_v2.py (4 cases, no device). HW-verified clean.

PENDING (requested, do after the TikTag hunts; needs the Pixel):
  Compare the generated assembly of the relocation-era seal delivery vs the GVT replacement:
  (a) LOGIC — same instructions/effect per seal (pre-removal reference sealer at scratchpad/sealer_preGST.py,
      409050ce; relocation-era commits precede 425b88d4);
  (b) RATE — bits/second of the channels, GVT vs relocation (fair: vary only the delivery mechanism);
  (c) PIXEL MTE SAFETY — confirm the genuine/arch path raises NO MTE exception on the device. The user
      disabled panic-on-MTE-mismatch, but a tag-check exception could still fire and PRINT to dmesg;
      grep the Pixel dmesg for MTE/tag-fault lines during a genuine run and confirm none on the arch path.
