# Revizor AArch64 — Evaluation Measurements (Machine A: N3 + Pixel 8)

**Operator scope:** Machine A only — Neoverse N3 host + remote Pixel 8 (Cortex-A715 / Cortex-X3).
Machine B (Alder Lake baseline + L1D IP-prefetcher) is owned by a **separate agent**.

**Revizor commit:** `daca5968` (branch `eval-measurements`, pre-PTE tree).
PTE WIP preserved on `ni-emulation-table-pacmte` @ `471f87e7` (nothing lost).

**Global knobs:** R = 20 for every timed quantity (fewer only where one run is very long — noted per item).
T_max = 3600 s (per-run safety cap; defines the `cross` verdict). Executor pinned to one isolated core.

---

## Environment blocks (§5)

### Machine A — Neoverse N3 host
```
Machine:        GCP VM revizor-physical (us-central1-b)
CPU/microarch:  ARM Neoverse-N3, 96 cores, 1 thread/core (no SMT), BogoMIPS 2000
Pinned core:    90 (isolcpus=90-91 nohz_full=90-91 rcu_nocbs=90-91; irqaffinity 0-89,92-95)
Freq scaling:   no cpufreq governor exposed (cloud N3, fixed frequency) — record as "fixed, no governor"
PAC/MTE:        PAC present (paca pacg, QARMA); MTE ABSENT
RAM:            743 GiB
OS/build:       Debian GNU/Linux 13 (trixie) 13.7
Kernel:         6.12.111+deb13-cloud-arm64
Compiler:       gcc (Debian 14.2.0-19) 14.2.0
revizor commit: daca5968 (branch eval-measurements)
ISA spec:       base.json
T_max:          3600 s     R: 20
```

### Remote target — Pixel 8 (Cortex-A715 @ core 4 / Cortex-X3 @ core 8)
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

## Config mapping — Table 7.2 rows -> config files
| Row | Config | Basis |
|-----|--------|-------|
| Spectre-v1 (cond) / L1D        | `config.yml`                   | basic fuzzer, no speculation modeled -> cond-branch leak violates |
| Spectre-v2 (indirect) / L1D    | `config_live_v2b.yml`          | contract permits v1+v4 -> remaining leak = indirect-target (v2) |
| Spectre-v2 (indirect) / BTB    | `config_spectre_v2_leftover.yml` | v2 via BTB carrier + enable_cross_input_priming (generalized-priming-only) |
| Spectre-v4 (store bypass) / L1D| `config_live_v4.yml`           | permits v1 + enable_speculative_store_bypass -> store-bypass = v4 |
| Canonicality / cache-TLB       | `config_canon.yml`             | NI fuzzer, enable_canonicality, sealed ordinary loads/stores (mem -> L1D/TLB) |
| Canonicality / BTB             | `config_canon_cip_ni.yml`      | branch-target-sealing + cross-input priming (BTB leftover surfaces as an L1D cache divergence); unpinned by design |
| AUT* sign bit / cache-TLB      | `config_pac_signext.yml`       | NI fuzzer, enable_signext |

**BTB vs cache channel selection (per Gal):**
- cache/TLB channel: `enable_cross_input_priming: true` + target functions DO access memory.
- BTB channel: `enable_cross_input_priming: true` + targets do NOT access memory (no cache signal -> BTB only).
Knobs: `enable_cross_input_priming`, `cross_input_priming_localizer: optimal`, `cross_input_leaks_only`, `enable_branch_target_sealing`.

---

## Detection landscape (probe sweep, N3, -i 50)
| Row | config | probe result | note |
|-----|--------|--------------|------|
| canon_cachetlb | config_canon.yml | **detects ~22s** | headline leak; fast |
| v4_l1d | config_live_v4.yml | **detects ~30s** (tc 26, viol 1) | store-bypass fast |
| aut_signext | config_pac_signext.yml | **detects ~28s** (tc 16, viol 1) | PAC sign-ext fast |
| v1_l1d | config.yml (seq) | **no detect in 60s** (tc 222, viol 0) | needs branch mispredict — N3 quirk |
| v2_l1d | config_live_v2b.yml | **no clean result in 90s** (timed out) | indirect+deep-nesting; slow per-tc |

v1/v2 are the branch-misprediction variants; N3 branches don't mispredict without help
(`enable_branch_mistraining`, WIP-off by default; P+P priming can train v1/v2 away). Testing
`enable_branch_mistraining: true` (configs v1_mt.yml / v2_mt.yml) as the legitimate Revizor lever.

---

## Comparability with prior Revizor (methodology guardrails)
- **Throughput headline = test cases/hour** (vs ASPLOS'22 ">200 tc/h for complex contracts"); also report
  executions/sec with `execution = (test-case x input)` stated, inputs/test-case given, and the **contract named**.
- **TTD = wall-clock mean over R>=10 seeded runs, stop-at-first-violation** (= ASPLOS'22 §6.6 Table 4, "<10 min").
  ALSO capture **test-cases-to-violation** and **inputs-to-violation** (ASPLOS §6.3 Table 5) — HW-speed-independent,
  the fair ARM-vs-x86 comparand. State whether warm-up is in the wall-clock.
- **Reproduction = priming-survival (genuine) x detected-in-all-R-seeds (stable)**; state per-input n_reps
  (group gate = 5; ASPLOS = 50) and which priming strategy; report false-positive/noise rate.
- **Localization: HW Measure-calls x 2 directions x reps**, ceil(log2(k+1)) (binary) vs k (linear), k=12/64/256;
  label "input-class/boundary localization" (ours-vs-ours), distinct from prior-art program minimization.
- **Setup must state** single-core pin, SMT off (N3 1 thread/core), and **prefetcher state** (flag if left ON —
  ASPLOS disabled prefetchers), inputs/tc, test-case length in instructions AND basic blocks, spec window/nesting
  depth (ASPLOS=250), input entropy, measurement channel (P+P/F+R, perf-counter), ARM knobs (QARMA3, TBI, FPAC).

## Key constraints / gotchas
- **cross-input priming is LOCAL-only** (`aarch64_fuzzer.py` asserts `enable_cross_input_priming=False` for remote)
  -> the BTB rows (v2/BTB, Canon/BTB) cannot use cross-input priming on the remote Pixel. HANDLING TBD for A715/X3 BTB cells.
- N3: no cpufreq governor (fixed), no MTE.
- Remote: fuzzer insmods its own module; device node chmod'd by setup. Device binaries built Sep 2-4 — validate protocol.

---

## §5 RESULT TEMPLATES (to fill)

### Table 7.1 — throughput (Machine A)
```
                      Local        Remote
Test cases / hour     ______       ______    (remote variant: Pixel / loopback-N3 — label it)
Inputs / test case    ______       ______
Executions / second   ______       ______
R = ___, duration per run = ___ s, config(local)=____, config(remote)=____, contract=____
```

### Table 7.2 — violations: TTD + reproduction
```
Violation / Channel              TTD (mean±sd, R)   tc-to-viol / inp-to-viol   N3    A715   X3    AlderLake
Spectre-v1 (cond) / L1D          N3: ____           ____ / ____                ____  ____   ____  (MachB)
Spectre-v2 (indirect) / L1D      N3: ____           ____ / ____                ____  ____   ____  (MachB)
Spectre-v2 (indirect) / BTB      N3: ____           ____ / ____                ____  ____   ____  (MachB)
Spectre-v4 (store bypass) / L1D  N3: ____           ____ / ____                ____  ____   ____  (MachB)
Canonicality / cache-TLB         N3: ____           ____ / ____                ____  ____   ____  (MachB)
Canonicality / BTB               N3: ____           ____ / ____                ____  ____   X3=cross?  (MachB)
AUT* sign bit / cache-TLB        N3: ____           ____ / ____                ____  ____   ____  n/a
(check / cross / n/a; T_max = 3600 s; R = 20)
```

### Table 7.3 — localization (Machine A)
```
                         k=12   k=64   k=256
Binary search (analytic)   4      7      9
Measured avg (binary)     ___    ___    ___      (mean±sd over R runs; HW Measure-calls x2 dir x reps)
```

### Exploit numbers — §7.5 / §7.6 (Machine A, PoCs in ~/arm-disclosure/pocs)
```
Cross-process (BTB), PMU:        accuracy ____%   rate ____ bit/s
Cross-process (BTB), rootless:   accuracy ____%   rate ____ bit/s   (--samples 800)
Cross-VM (BTB):                  accuracy ____%   rate ____ bit/s
Cross-privilege (L1 dTLB):       accuracy ____%   rate ____ bit/s
Blind canary:                    v1 __/20   v4 __/20   rsb __/20
OpenSSL RSA-2048:                ____/2048 bits   wall-clock ____ h   (--samples 1500)
SMC/BTB (appendix):              SMC effect ____   CFP-RCTX control ____
```
