# Evaluation Results — ARM Neoverse-N3 (+ remote Pixel 8), branch eval-measurements

Machine A operator: Neoverse-N3 host (single isolated core 90, local) + remote Pixel 8 over adb (A715=cpu4,
X3=cpu8, A510=cpu0). Channels in scope: L1D and the BTB only (no TLB attack surface fuzzed). Mistraining
deprecated/unused (executor warmups OK). All numbers below trace to artifacts under eval_out/; see per-row
paths. Comparison to prior Revizor papers in COMPARISON.md; integrity audit trail in VERIFICATION.md.

## 7.1 Throughput (local N3, single isolated core, inputs/tc=100 effective)
| contract | tc/h | exec/s | traces/s | source |
|----------|------|--------|----------|--------|
| CT-SEQ (simplest)        | 7,895  | 219.3 | 4,384 | measured, 1316 tc in 600.1 s (campaign.log) |
| CT-COND                  | 12,177 | —     | 12,165 | measured |
| CT-COND-BPAS             | 7,810  | —     | 7,865  | measured |
| Heavy NI (seal+dispatch+nesting=30) | ~200–250 | — | — | ESTIMATE from ~15–17 s/tc anchor (NOT a logged run) |

## 7.2 Time-to-detection (R=20 per row, local N3)
| violation | channel | R=20 mean | note |
|-----------|---------|-----------|------|
| Spectre-v1 | L1D | within 32–80 s band | v1_l1d.yml |
| Spectre-v4 | L1D | within 32–80 s band | v4_l1d.yml |
| Spectre-v2 (std priming, L1D) | L1D | 79.9 s (max 361.9 s; one outlier) | v2_l1d.yml — DISTINCT from the pure-v2/BTB template below |
| Canonicality | cache/TLB(L1D) | 42.9 s | canon_l1d.yml (≈10 s template rediscovery cited separately) |
| AUT*/PAC sign-bit | cache/TLB(L1D) | within 32–80 s band (runs to ~144 s) | aut_signext.yml |
Row MEANS are <~90 s (not every individual run — v2 max 361.9 s, aut to ~144 s).

## 7.3 Localization cost (A6; binary search vs linear worst case)
k=12: binary 4 / linear 12; k=64: binary 7 / linear 64; k=256: binary 9 / linear 256. Measured avg via the
real exponential_search/linear_scan localizers (eval_scripts/a6_localization_cost.py).

## Spectre-v2 (indirect branch) over the real BTB — which-target, NO sealing (headline)
Paper Fig-(a) gadget generalized to 3 targets: each target RETURNS a distinct offset (no memory access in
the target, no sealing, no PAC); the victim loads from the returned value, so which target the BTB predicts
selects the L1D set. Detected by generalized (cross-input) priming, dropped by standard priming.
KEY: force CT-SEQ so the speculative dispatch/target is beyond-contract -> target-selecting bits UNTAINTED
-> boosting varies them -> the within-class which-target toggle appears. templates/spectre_v2_btb.py.
| path | result |
|------|--------|
| template (v2_btb_seq.yml, -i100) | CONFIRMED cross-input leaking 0 -> detecting 11, 1 viol/57 s |
| deterministic reproduce x3 | 3/3 identical pair 0->11 (seeds prog 716170 / input 2522819617) |
| template under random config (capability) | CONFIRMED leaking 4 -> detecting 5, 13.6 s |
| TTD (R=20, template) | 20/20 detected (100%), mean 71.3 s, range 36.9–155.5 s, distinct cross-input pair each run |
| fully-random (no template) | CLOSED — structurally rare (0 confirmed across rand1-4 + rand_final = 201 programs / 1 h, and more). Plain-fuzzer ct-equal boosting cannot toggle an ARCHITECTURAL dispatch; only a chance SSB-gated spec-only dispatch is localizable. No seal per user decision. |

## Canonicality over the BTB (for contrast — leaked axis = target canonicality, uses the seal)
Fully-random canon_btb_3t.yml (NI fuzzer + branch_target_sealing): 12/20 (60%) within cap, 2 triage-GENUINE.
Template canon_oracle_ssb.py reliable. Cache/TLB canonicality channel also confirmed (canon_cachetlb).

## Remote reproduction (Pixel 8 over adb; ABI-2 module verified, abi_version=2)
| core | cluster | rows detected | note |
|------|---------|---------------|------|
| A715 (cpu4) | mid | 4/5 at first 300 s cap; v1 on recheck | v1 missed at 107 tc, hit at 137 tc/374 s (marginal, core-dependent BPU) |
| X3 (cpu8) | prime | 4/5 | v2_l1d hit the 361 s cap (X3 BPU needs more tc) |
| A510 (cpu0) | efficiency | 5/5 | v1/v2/v4/canon(cache-TLB)/aut all detected 22–54 s |
canon cache/TLB reproduces on X3 (consistent with paper: only the BTB canonicality channel is A715-only).

## A7 end-to-end exploits (N3, isolated core 90) — fresh --save artifacts in eval_out/exploit_artifacts/
| exploit | accuracy | rate | artifact |
|---------|----------|------|----------|
| Cross-process EL0->EL0 (BTB), PMU       | 56/56 (100%)   | 51 bit/s   | exploit_artifacts/xproc_pmu.{log,save} |
| Cross-process EL0->EL0 (BTB), rootless  | 54/56 (96.4%)  | 5 bit/s    | exploit_artifacts/xproc_rootless.* (--samples 800) |
| Cross-VM guest->host (BTB)              | 70/70 (100%)   | 125 bit/s  | exploit_artifacts/xvm.* (--victim-branches 16) |
| Cross-privilege EL2->EL0 (L1 dTLB)      | 160/160 (100%) | 1240 bit/s | exploit_artifacts/xel.* (16-bit secret) |
| SMC/BTB (appendix)                      | effect 0.000   | —          | exploit_artifacts/smc.* (NOP 1.000 − SMC 1.000; CFP RCTX 0.000 flushes; self-flood 0.028 control) |
| Blind stack canary [55:0]               | v1 18/19, v4 19/19, rsb 18/19 FULL | — | exploit_canary/all.log (20 launches) |
| OpenSSL RSA-2048 ROP                    | PENDING (runs LAST) | — | --samples 1500, ~5.5 h; target ~2044/2048 |
Fresh numbers supersede earlier log assertions (xproc 63->51, rootless 100%->96.4%, xvm 126->125, xel 1242->1240) — now artifact-backed.
