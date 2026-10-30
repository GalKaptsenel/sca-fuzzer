# Task-2 violation triage — VERDICT (local CE, no-HW-crash path)

Violation: `violation-261030-001505`, counterexamples input #26 vs #68. Config: PAC auth + sign-ext
(`config_pac_signext.yml`), clause `cond`, strict mode.

## VERDICT: GENUINE speculative-exception leak via the sealed PAC-auth / sign-ext (AUT* bit-55) channel.

### Evidence (all local, safe — the CE is a local subprocess; no HW execution in this triage)
1. **Causation is airtight — only the seal differs.** The two executor-inputs #26 and #68 have
   BYTE-IDENTICAL `main`, `faulty`, and `gpr` regions; they differ ONLY in the GST image (SIMD
   section). The differing GST bytes fall exclusively in PAC-auth (AUT*) and sign-ext seal lanes on
   SPECULATIVE slots. Therefore the hardware htrace difference cannot come from the architectural
   input — it is caused solely by the seal values.
2. **Contract-equivalent (violation premise), independently recomputed.** The genuine contract traces
   of #26 and #68 hash equal (176787323439393868677790057880061214801) — they are indistinguishable
   under the contract, as a violation requires.
3. **The leak is a single, stable cache set.** Per-bit htrace frequency over 500 reps each: set 11
   fires 0.76 in #26 and 0.00 in #68 (every other set agrees). A measurement-noise difference would be
   small/symmetric; 76% vs 0% is a stable channel.
4. **Mechanism = the AUT* / sign-ext channel.** In the genuine resolve trace a speculative (nest=1)
   sealed access lands exactly on set 11. The channel is presence/absence of that speculative
   allocation: when the authed pointer's sign-extension/PAC is intact the speculative access proceeds
   and allocates set 11; when the sign-ext/PAC decoy breaks it the access faults speculatively and
   never allocates. The attacker reads the PAC/canonicality bit from the set-11 allocation. The only
   speculative-exception seals active in this config are PAC-auth + sign-ext, so this is structurally
   "the aut thing", not an unrelated load.

### Reproducible — and why naive HW replay crashes the box (ROOT-CAUSED)
Within the campaign's own differential measurement the signal is stable (500 reps, 76% vs 0%). That is
the primary reproducibility evidence (the detection was at a single, consistent sandbox base, with
"Hardware Tracing Errors: 0" — i.e. arch-safe at that base, so the divergence was speculative = a
genuine leak).

**Independent HW replay of the SAVED images is impossible across a reboot, and crashes the box — by
design, not a bug.** PAC signatures are sandbox-base-specific: a signature authenticates `base+offset`.
The saved GST images carry the campaign boot's signatures. The sandbox base is NOT stable across
reboots (observed: `0xffff40005ca61000` → `0xffff40008b821000`). Replaying the saved images on silicon
at a new base runs STALE signatures, so the ARCHITECTURAL AUT*s FPAC → the box resets. Evidence:
  * the reconstructed executor-input serializes BYTE-IDENTICAL to the saved `.reif` (so it is not a
    plumbing bug), yet two independent HW replays reset the box;
  * at the post-reboot base the saved images are non-genuine on architectural PAC lanes (the stale
    signatures), e.g. lane [506,508) `autizb` (arch);
  * the CE-only triage never crashed because the CE ABORTS in software on a bad signature (observed as
    the contract_executor signal-6), rather than faulting silicon;
  * `revizor.py reproduce` did NOT crash and found 0 violations because it RE-SEALS (re-signs at the
    current base) — so it is arch-safe, but it also re-randomises the decoys and does not reconstruct
    the exact leaking pair, hence 0 (not a refutation).

**Implication / rule:** never replay a saved PAC GST image (or a hand-built PAC input) on hardware
after a reboot — re-seal at the current base (`revizor.py fuzz/reproduce`) or regenerate from seeds.
The prior subagent's manual reproduce crashed for the same reason (stale base-specific signatures on
the architectural path). This also connects to the known PAC box-reset hazard.

### Safe reproduction (re-seal at the current base)
Re-fuzzing `config_pac_signext.yml` from the recorded seeds (program 522346, input 3652968709)
re-signs at the current base, so it reproduces the AUT* violation without the stale-signature FPAC.
Result appended below.
