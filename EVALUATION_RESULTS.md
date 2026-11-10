# Revizor AArch64, Evaluation Measurements (Machine A: N3 + Pixel 8)

**Operator scope:** Machine A only, Neoverse N3 host + remote Pixel 8 (Cortex-A715 / Cortex-X3).
Machine B (Alder Lake baseline + L1D IP-prefetcher) is owned by a **separate agent**.

**Revizor commit:** `daca5968` (branch `eval-measurements`, pre-PTE tree).
PTE WIP preserved on `ni-emulation-table-pacmte` @ `471f87e7` (nothing lost).

**Global knobs:** R = 20 for every timed quantity (fewer only where one run is very long, noted per item).
T_max = 3600 s (per-run safety cap; defines the `cross` verdict). Executor pinned to one isolated core.

---

## Environment blocks (§5)

### Machine A, Neoverse N3 host
```
Machine:        GCP VM revizor-physical (us-central1-b)
CPU/microarch:  ARM Neoverse-N3, 96 cores, 1 thread/core (no SMT), BogoMIPS 2000
Pinned core:    90 (isolcpus=90-91 nohz_full=90-91 rcu_nocbs=90-91; irqaffinity 0-89,92-95)
Freq scaling:   no cpufreq governor exposed (cloud N3, fixed frequency), record as "fixed, no governor"
PAC/MTE:        PAC present (paca pacg, QARMA); MTE ABSENT
RAM:            743 GiB
OS/build:       Debian GNU/Linux 13 (trixie) 13.7
Kernel:         6.12.111+deb13-cloud-arm64
Compiler:       gcc (Debian 14.2.0-19) 14.2.0
revizor commit: daca5968 (branch eval-measurements)
ISA spec:       base.json
T_max:          3600 s     R: 20
```

### Remote target, Pixel 8 (Cortex-A715 @ core 4 / Cortex-X3 @ core 8)
```
Device/SoC:     Google Pixel 8 (shiba), Google Tensor G3, platform zuma, arm64-v8a, 8 GB (7.19 GiB)
Cores:          0-3 Cortex-A510 0xd46 @1.70GHz | 4-7 Cortex-A715 0xd4d @2.37GHz | 8 Cortex-X3 0xd4e @2.91GHz
Pinned core:    A715 = 4  |  X3 = 8   (executor_pinned_core)
OS:             Android 14 (SDK 34), security patch 2024-09-05, build AP2A.240905.003 (user/release-keys)
                fingerprint google/shiba/shiba:14/AP2A.240905.003/12231197:user/release-keys
Kernel:         5.15.137-android14-11-gbc062a78e195-ab12057991 #1 SMP PREEMPT Mon Jul 8 2024
Root:           Magisk 28.0 (MAGISKSU, /system/bin/su) -> insmod permitted
Features:       PAC (paca pacg) + MTE (mte mte3) on both A715 & X3; ssbs, sve2, bti
Transport:      adb over SSH reverse tunnel (laptop adb server -> N3 127.0.0.1:5037), serial 47061FDJH001K3
Device staging: /data/local/tmp/executor_userland , /data/local/tmp/revizor-executor.ko
                (CURRENTLY built Sep 2-4; freshness vs Oct-12 fuzzer protocol PENDING validation)
```

---

## Config mapping, Table 7.2 rows -> config files
| Row | Config | Basis |
|-----|--------|-------|
| Spectre-v1 (cond) / L1D        | `config.yml`                   | basic fuzzer, no speculation modeled -> cond-branch leak violates |
| Spectre-v2 (indirect) / L1D    | `config_live_v2b.yml`          | contract permits v1+v4 -> remaining leak = indirect-target (v2) |
| Spectre-v2 (indirect) / BTB    | `config_spectre_v2_leftover.yml` | v2 via BTB carrier + enable_cross_input_priming (generalized-priming-only) |
| Spectre-v4 (store bypass) / L1D| `config_live_v4.yml`           | permits v1 + enable_speculative_store_bypass -> store-bypass = v4 |
| Canonicality / L1D       | `config_canon.yml`             | NI fuzzer, enable_canonicality, instrumented ordinary loads/stores (mem -> L1D/TLB) |
| Canonicality / BTB             | `config_canon_cip_ni.yml`      | branch-target-GVT access + cross-input priming (BTB leftover surfaces as an L1D cache divergence); unpinned by design |
| AUT* sign bit / L1D      | `config_pac_signext.yml`       | NI fuzzer, enable_signext |

**BTB vs cache channel selection (per Gal):**
- cache/TLB channel: `enable_cross_input_priming: true` + target functions DO access memory.
- BTB channel: `enable_cross_input_priming: true` + targets do NOT access memory (no cache signal -> BTB only).
Knobs: `enable_cross_input_priming`, `cross_input_priming_localizer: optimal`, `cross_input_leaks_only`, `enable_branch_target`.

---

## Detection landscape (probe sweep, N3, -i 50)
| Row | config | probe result | note |
|-----|--------|--------------|------|
| canon_cachetlb | config_canon.yml | **detects ~22s** | headline leak; fast |
| v4_l1d | config_live_v4.yml | **detects ~30s** (tc 26, viol 1) | store-bypass fast |
| aut_signext | config_pac_signext.yml | **detects ~28s** (tc 16, viol 1) | PAC sign-ext fast |
| v1_l1d | config.yml (seq) | **no detect in 60s** (tc 222, viol 0) | needs branch mispredict, N3 quirk |
| v2_l1d | config_live_v2b.yml | **no clean result in 90s** (timed out) | indirect+deep-nesting; slow per-tc |

v1/v2 are the branch-misprediction variants; N3 branches don't mispredict without help
(`enable_branch_mistraining`, WIP-off by default; P+P priming can train v1/v2 away). Testing
`enable_branch_mistraining: true` (configs v1_mt.yml / v2_mt.yml) as the legitimate Revizor lever.

---

## Comparability with prior Revizor (methodology guardrails)
- **Throughput headline = test cases/hour** (vs ASPLOS'22 ">200 tc/h for complex contracts"); also report
  executions/sec with `execution = (test-case x input)` stated, inputs/test-case given, and the **contract named**.
- **TTD = wall-clock mean over R>=10 seeded runs, stop-at-first-violation** (= ASPLOS'22 §6.6 Table 4, "<10 min").
  ALSO capture **test-cases-to-violation** and **inputs-to-violation** (ASPLOS §6.3 Table 5), HW-speed-independent,
  the fair ARM-vs-x86 comparand. State whether warm-up is in the wall-clock.
- **Reproduction = priming-survival (genuine) x detected-in-all-R-seeds (stable)**; state per-input n_reps
  (group gate = 5; ASPLOS = 50) and which priming strategy; report false-positive/noise rate.
- **Localization: HW Measure-calls x 2 directions x reps**, ceil(log2(k+1)) (binary) vs k (linear), k=12/64/256;
  label "input-class/boundary localization" (ours-vs-ours), distinct from prior-art program minimization.
- **Setup must state** single-core pin, SMT off (N3 1 thread/core), and **prefetcher state** (flag if left ON,  ASPLOS disabled prefetchers), inputs/tc, test-case length in instructions AND basic blocks, spec window/nesting
  depth (ASPLOS=250), input entropy, measurement channel (P+P/F+R, perf-counter), ARM knobs (QARMA3, TBI, FPAC).

## Key constraints / gotchas
- **cross-input priming is LOCAL-only** (`aarch64_fuzzer.py` asserts `enable_cross_input_priming=False` for remote)
  -> the BTB rows (v2/BTB, Canon/BTB) cannot use cross-input priming on the remote Pixel. HANDLING TBD for A715/X3 BTB cells.
- N3: no cpufreq governor (fixed), no MTE.
- Remote: fuzzer insmods its own module; device node chmod'd by setup. Device binaries built Sep 2-4, validate protocol.

---

## §5 RESULT TEMPLATES (to fill)

### Table 7.1, throughput (Machine A)
```
                      Local        Remote
Test cases / hour     ______       ______    (remote variant: Pixel / loopback-N3, label it)
Inputs / test case    ______       ______
Executions / second   ______       ______
R = ___, duration per run = ___ s, config(local)=____, config(remote)=____, contract=____
```

### Table 7.2, violations: TTD + reproduction
```
Violation / Channel              TTD (mean±sd, R)   tc-to-viol / inp-to-viol   N3    A715   X3    AlderLake
Spectre-v1 (cond) / L1D          N3: ____           ____ / ____                ____  ____   ____  (MachB)
Spectre-v2 (indirect) / L1D      N3: ____           ____ / ____                ____  ____   ____  (MachB)
Spectre-v2 (indirect) / BTB      N3: ____           ____ / ____                ____  ____   ____  (MachB)
Spectre-v4 (store bypass) / L1D  N3: ____           ____ / ____                ____  ____   ____  (MachB)
Canonicality / L1D         N3: ____           ____ / ____                ____  ____   ____  (MachB)
Canonicality / BTB               N3: ____           ____ / ____                ____  ____   X3=cross?  (MachB)
AUT* sign bit / L1D        N3: ____           ____ / ____                ____  ____   ____  n/a
(check / cross / n/a; T_max = 3600 s; R = 20)
```

### Table 7.3, localization (Machine A)
```
                         k=12   k=64   k=256
Binary search (analytic)   4      7      9
Measured avg (binary)     ___    ___    ___      (mean±sd over R runs; HW Measure-calls x2 dir x reps)
```

### Exploit numbers, §7.5 / §7.6 (Machine A, PoCs in ~/arm-disclosure/pocs)
```
Cross-process (BTB), PMU:        accuracy ____%   rate ____ bit/s
Cross-process (BTB), rootless:   accuracy ____%   rate ____ bit/s   (--samples 800)
Cross-VM (BTB):                  accuracy ____%   rate ____ bit/s
Cross-privilege (L1 dTLB):       accuracy ____%   rate ____ bit/s
Blind canary:                    v1 __/20   v4 __/20   rsb __/20
OpenSSL RSA-2048:                2035/2048 bits (99.37%)   wall-clock 5.58 h   (--samples 1500)
SMC/BTB (appendix):              SMC effect ____   CFP-RCTX control ____
```

---
## FILLED RESULTS (updated as phases complete)

### Table 7.2, N3 TTD + N3 reproduction (COMPLETE; R=20, T_max=3600s, pinned core 90, host quiet)
| Violation / Channel              | TTD N3 mean±sd (R=20) | tc-to-viol mean[min-max] | inp-to-viol (≈tc×50 gen) | N3    | A715 | X3  | AlderLake |
|----------------------------------|-----------------------|--------------------------|--------------------------|-------|------|-----|-----------|
| Spectre-v1 (cond) / L1D          | 53.9 ± 22.1 s         | 86.7 [6-221]             | ≈4,335                   | check | check| check| (Mach B) |
| Spectre-v2 (indirect target) / L1D | 79.9 ± 77.6 s       | 4.4 [1-11]               | ≈220                     | check | check| check| (Mach B) |
| Spectre-v4 (store bypass) / L1D  | 32.3 ± 9.8 s          | 30.2 [2-132]             | ≈1,510                   | check | check| check| (Mach B) |
| Canonicality / L1D               | 42.9 ± 23.1 s         | 77.8 [5-295]             | ≈3,890                   | check | check| check| (Mach B) |
| AUT* sign bit / L1D              | 54.5 ± 39.9 s         | 66.8 [1-234]             | ≈3,340                   | check | check| check| n/a      |
| Spectre-v2 (indirect) / BTB      | 71.3 ± 43.6 s (tmpl)  | 1 [1-1] (tmpl, -i100)    | ≈100 (within 1 tc)       | check | n/a  | n/a | (Mach B) |
| Canonicality / BTB               | 84.0 s (fully-random) | random (12/20 @ -i50 cap)| ≈(tc×50)                 | check | n/a  | X3=cross? | (Mach B) |

inp-to-viol = tc-to-viol × 50 GENERATED inputs/tc (the -i used in every TTD run; ×4 boosting → 200 executed);
derived from tc, not independently logged, and an upper estimate (the violating tc stops partway). Executed
inputs ≈ 4× these. The 5 L1D/PAC rows detect 20/20 on N3 (genuine, survive priming; v2 independently
CE-triaged genuine). Remote: X3 5/5, A715 5/5, A510 5/5 (L1D/PAC rows). BTB rows are local-only (cross-input
priming asserts off for remote), so A715/X3 = n/a. Spectre-v2/BTB row = the strict cross-input TEMPLATE
(v2_btb_seq.yml, TTD 20/20, 1 tc each); fully-random does NOT surface it, and existing-knob tuning cannot
force it (~0/240), a plain self-dependent v2 IS found by random fuzzing once cross_input_leaks_only=false.

### Table 7.3, localization cost (COMPLETE; device-independent, real binary_search/linear_scan localizers)
  k     analytic   measured binary (mean+-sd) [min-max]   (contrast) gallop   linear
  12       4          5.75 +- 0.43  [5-6]                      7.50            9.42
  64       7          8.02 +- 0.12  [8-9]                     12.14           35.48
  256      9         10.00 +- 0.06  [10-11]                   16.04          131.50
  512     10         11.00 +- 0.04  [11-12]                   18.02          259.50
  1024    11         12.00 +- 0.03  [12-13]                   20.01          515.50
  2048    12         13.00 +- 0.02  [13-14]                   22.01         1027.50
analytic = ceil(log2(k+1)). Measured binary = real binary_search (pure bisection over the full range),
averaged over all boundary positions; it is analytic+1 (the +1 = the 2 bracketing endpoint probes) and
FLAT in the boundary position (sd 0.43 -> 0.02 as k grows). Galloping (exp_search) ~= 2*analytic for a
uniform boundary (cheaper ONLY for a recent/near trainer, ~const 4); linear ~= k/2. probes = distinct
memoized HW batch measurements. Full HW cost = probes x reps x 2 swap-dirs + 2 reverify. eval_scripts/a6.

### Table 7.2, Canon/BTB + v2/BTB rows (cross-input, N3)
| Violation / Channel        | N3 result | detail |
|----------------------------|-----------|--------|
| Canonicality / BTB         | check (template/seed), rare (fully-random) | template canon_oracle_ssb: CONFIRMED cross-input leak ~6s, reproducible (leaker->detector, strictly cross-input); gadget seed 799067: 9s deterministic. Fully-random auto-detect: seed-dependent/rare (gadget uncommon), NOT a contract/regime bug ([seq] & [cond,bpas] both rare; regime live-verified). |
| Spectre-v2 / BTB           | check (template); random v2 abundant once filter off | same BTB cross-input carrier; leaked bit = predicted target (no GVT access). CORRECTION: fully-random "0/rare" was an artifact of cross_input_leaks_only:true discarding self-dependent leaks, same 30 seeds give 0 viol with the filter, 11 confirmed/13tc without (7×v1/2×v4/1×v2 38v88). STRICTLY cross-input pure-BTB (leaker≠prober) remains rarer. |
Fixes that restored it: aarch64_fuzzer.py (regime wraps detection, not just localization) + builder.py (re-clamp base after BL/BLR). Committed 2c3b18e3.

### Canon/BTB fully-random reproducibility (eval_configs/canon_btb_3t.yml: >=3 targets, -i 50, 180s cap)
R=20: 12/20 detected (60%) within cap; mean TTD(detectors)=84.0s. 12 DISTINCT cross-input gadgets:
  39->41 26->30 24->25 38->40 21->28 25->27 36->38 19->29 34->36 30->38 41->43 32->34.
Plus: template canon_oracle_ssb ~6s reliable; gadget seed 799067 deterministic 9s; 2 real-fuzzed
violations triage-verified GENUINE cross-input BTB (24->27, 23->24). Rate bounded by -i 50 per-tc cost
(~10 tc/180s); fewer targets (2) / fewer inputs (8) => rare. HW controlled-swap: pending (next, device free).

### Table 7.1, throughput LOCAL (N3, 1 isolated core 90, stock basic fuzzer)
| contract | reps | tc/hour (mean) | exec/s (mean) | traces/s (mean) | inputs/tc |
|----------|------|----------------|---------------|-----------------|-----------|
| CT-SEQ   | 2 (600s) | 9,151 (7895, 10408) | 254 | 4,223 | 100 (eff) |
| CT-COND  | 2 (400s) | 12,171 (12177, 12165) | 338 | 3,407 | 100 |
| CT-COND-BPAS | 2 (400s) | 7,838 (7810, 7865) | 218 | 2,190 | 100 |
| heavy-NI (GVT access+dispatch+nest30) |, | ~200-250 (anchor) | | | 100 |
Execution = (test-case x input) pair. traces/s = fuzzer-reported (x reps). Remote (3-way) + CT-COND rows pending.

Contract-vs-throughput note: traces/s is the clean monotonic contract-cost indicator (seq 4,223 >
cond 3,407 > cond+bpas 2,190 -> richer contract = more CE model time per measurement). tc/h is noisier
(seq 9,151; cond 12,171; cond+bpas 7,838) because boosting/effective-classes differ by contract. Simple
contracts ~8-12k tc/h; the heavy-NI contract (deep nesting + dispatch) is the ~40x drop to ~200 tc/h.
v2_btb_3t (pure Spectre-v2 / BTB, no GVT access) fully-random: the earlier "0/3 rare" was an artifact of
cross_input_leaks_only:true (filters self-dependent leaks); with it off, random fuzzing finds v2 readily
(1 of 11 random self-dependent violations CE-ladder-attributes to v2).

### Table 7.2, remote A4 (Cortex-A715=core4) + A5 (Cortex-X3=core8) reproduction (L1D/PAC rows)
| Violation / Channel | N3 | A715 | X3 | detail |
|---------------------|----|----|----|--------|
| Spectre-v1 / L1D    | check | check (137tc recheck; first run cap-limited @107tc) | check (44s,1tc) | v1 A715 needs more tc than X3: core-dependent cond-branch BPU, but reproduces |
| Spectre-v2 / L1D    | check | check (126s) | check (185s) | |
| Spectre-v4 / L1D    | check | check (92s)  | check (45s)  | |
| Canonicality / L1D  | check | check (96s)  | check (48s)  | |
| AUT* sign bit / L1D | check | check (213s) | check (40s)  | A715 AUT* was the protocol's open unknown -> CHECK |
X3 = 5/5 reproduce; A715 = 4/5 (v1 pending recheck). BTB rows remote: N/A (cross-input priming is local-only).
Remote per-tc ~2.8s (adb). ABI-2 Pixel module (built on N3 from GKI), verified loading, abi_version=2.

### Table 7.1, throughput COMPLETE
| variant | tc/hour | exec/s | traces/s | note |
|---------|---------|--------|----------|------|
| Local N3 (CT-SEQ) | 9,151 | 254 | 4,223 | 1 isolated core |
| Remote Pixel A715 (CT-SEQ) | 1,233 | 34.3 |, | ~7.4x slower; ~2.5s adb+device overhead/tc |
| Local CT-COND | 12,171 | 338 | 3,407 | |
| Local CT-COND-BPAS | 7,838 | 218 | 2,190 | |
| Local heavy-NI (nest30+dispatch) | ~200-250 | | | complex-contract anchor |
Remote-loopback-N3 not feasible via adb (N3=Debian, no adbd); overhead quantified via per-tc delta.

### §7.5/7.6, end-to-end exploits (N3, isolated core 90)
| exploit | accuracy | rate | note |
|---------|----------|------|------|
| Cross-process EL0->EL0 (BTB), PMU       | 100% (56/56 bits) | 51 bit/s   | 8/8 rounds, 7-bit secret (artifact xproc_pmu.save; audit-corrected 63->51) |
| Cross-process EL0->EL0 (BTB), rootless  | 96.43% (54/56)    | 5 bit/s    | 6/8 rounds, --samples 800 (artifact xproc_rootless.save; audit-corrected 100%->96.43%) |
| Cross-VM guest->host (BTB)              | 100% (70/70)      | 125 bit/s  | 10/10 rounds, --victim-branches 16 (artifact xvm.save; audit-corrected 126->125) |
| Cross-privilege EL2->EL0 (L1 dTLB)      | 100% (160/160)    | 1240 bit/s | 10/10 rounds, 16-bit secret (artifact xel.save; audit-corrected 1242->1240) |
| Blind stack canary [55:0]               | v1 18/19, v4 19/19, rsb 18/19 FULL |, | fresh random canary/launch (target v1 20/20, rsb 19/20, v4 18/20) |
| OpenSSL RSA-2048 ROP                    | 2035/2048 = 99.37% (21/32 limbs exact) | 0.102 bit/s | --samples 1500, wall 20101 s = 5.58 h (target ~2044/2048=99.8%, ~5.5h) |
| SMC/BTB (appendix)                      | SMC effect 0.000  |,          | baseline/NOP/SMC survived=1.000; self-flood control 0.028; CFP RCTX flushes (survived 0.000); MMU M-toggle no flush (artifact smc.save) |
All cross-boundary + SMC match targets. OpenSSL COMPLETE (99.37% bit accuracy, 5.58 h, matches target). All PoCs from ~/arm-disclosure/pocs (built on N3).
Note: exploit rates above are the artifact-backed, multi-model-audited numbers (VERIFICATION.md); they supersede earlier log assertions.
