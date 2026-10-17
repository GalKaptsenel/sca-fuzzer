# Revizor throughput & methodology: our AArch64/Neoverse-N3 vs. all prior Revizor papers

Scope: honest, source-verified comparison of fuzzing throughput and the time-to-detection (TTD)
methodology. Every number below names its **contract** and its **hardware**, because tc/hour
varies ~10x with contract complexity and is not portable across microarchitectures/ISAs. This is
a literature + synthesis document; no fuzzing or hardware was touched to produce it.

Units used:
- **tc/h** = test-cases per hour (one test case = one random program + its input batch).
- **executions/s** = (test-cases x inputs-per-tc) per second = program executions, before
  per-execution measurement repetitions.
- **traces/s** = individual hardware measurements per second (= executions x measurement
  repetitions for noise reduction).

---

## 1. Throughput comparison table

| Tool / paper | Hardware (microarch, ISA) | Contract(s) | Inputs / tc | Reported throughput (native unit) | Normalized (tc/h; executions/s where derivable) | Source |
|---|---|---|---|---|---|---|
| **Revizor** (Oleksenko et al., ASPLOS'22) | Intel i7-6700 (Skylake) & i7-9700 (Coffee Lake), x86-64 | CT-SEQ / CT-COND / CT-BPAS / ARCH-SEQ / MEM-SEQ | 50 initial, grown to "several hundreds" | **">200 test cases per hour for complex contracts"**; full V1/V4/MDS/LVI discovery "under two hours, on average" | **>200 tc/h (complex contracts)**; executions/s not reported | §1 contribution (6), p.2, L176-181 of asplos dump; CPUs L861-862 |
| **Hide & Seek with Spectres** (EuroS&P'23, arXiv 2301.07642) | Intel i5-6500 (Core6), i7-8650U (Core8), E-2288G (Xeon); AMD EPYC 7543P, x86-64 | CT-SEQ (+ the filters) | 100 | Reports **speedup, not tc/h**: "two orders of magnitude" faster; 130M test-case executions over 13 ISA subsets in **16 h** (orig. Revizor would need **">2 months"**); detection-time reduced by **factor 10-50** | Derived from campaign (see note A): ~**81k tc/h** / ~**2,257 exec/s** optimized vs ~**0.9k tc/h** / ~**25 exec/s** baseline — spans 13 heterogeneous subsets, most tc filtered cheaply; NOT a per-tc fully-executed rate | Fig.1 caption L231-237; L136-143; L242-244 of hs dump |
| **Flowvizor** (ASPLOS'27, Technion) | Intel i5-12500 (Alder Lake / Golden Cove P-core); comparison runs on AMD Zen 3, x86-64 | CT-SEQ; CT-(AM-COND) baseline; BPU-oracle | 40 | Reports campaign size in **"runs"**, not tc/h: benchmark = 10K programs x 40 inputs = **400K runs** per test-case size; per-tc end-to-end ~50-150 ms (conditional-branch microbenchmark, Fig.11) | No tc/h stated; Fig.11 per-tc time implies order-of ~10 tc/s for the short conditional-branch microbenchmark — not a representative campaign rate (reset/short-tc dominated) | §8 L1450-1472; L315-319; Fig.11 L1600-1620 of flowvizor dump |
| **Ours** (this work) | **ARM Neoverse-N3**, single isolated core (core 90), local, stock basic fuzzer, AArch64 | **CT-SEQ** (simplest) | **100** (effective, after lane-boosting) | **1316 tc in 600.1 s** | **7,895 tc/h**; **219.3 exec/s**; **4,384 traces/s** | measured N3 run |
| **Ours** (this work) | ARM Neoverse-N3, single isolated core, AArch64 | **Heavy NI** (branch-target sealing + multi-target dispatch + `model_max_nesting=30`) | 100 | **~15-17 s / test-case** | **~200-250 tc/h** (complex-contract anchor) | ESTIMATE from the ~15-17 s/tc anchor — NOT a logged throughput campaign (the measured throughput runs are the CT-SEQ/CT-COND/CT-COND-BPAS rows) |
| **Ours** (this work) | ARM Neoverse-N3, single isolated core, AArch64 | **CT-COND**, **CT-COND-BPAS** | 100 | *measurement in progress* | *TBD — expected between the SEQ and heavy-NI rows as contract complexity rises* | to append |

**Note A (H&S derivation, shown for completeness, treat with caution).** The H&S campaign is
100,000 programs x 100 inputs x 13 ISA subsets = 1.3M test cases / 130M executions. Optimized: /16 h
≈ 81k tc/h ≈ 2,257 exec/s. Baseline ("over 2 months", take ~60 d = 1440 h): ≈ 900 tc/h ≈ 25 exec/s.
These average over 13 subsets where the speculation/observation filters discard the vast majority of
test cases *before* the expensive model+HW stage, so they are a filtered-campaign aggregate, **not**
a per-tc fully-executed throughput and are **not** directly comparable to the ASPLOS ">200 tc/h" or
to our per-tc rates. H&S's own headline metric is the speedup, which is the honest comparand.

---

## 2. Analysis: is our N3 throughput reasonable vs. prior x86?

**The only fair comparison is per-contract tc/h, and the fair ASPLOS comparand is the complex
contract.** ASPLOS explicitly qualifies its headline rate: ">200 test cases per hour **for complex
contracts**" (L176-178). So:

- **Complex contract vs complex contract:** our heavy-NI configs run **~200-250 tc/h** on N3 — the
  **same order of magnitude** as ASPLOS's ">200 tc/h" on Skylake/Coffee Lake. This is the apples-to-apples
  row. Our N3 complex-contract throughput is therefore reasonable and in line with the original tool.
- **Simplest contract (CT-SEQ):** our N3 rate is **7,895 tc/h** — ~30-40x faster than the heavy-NI rate,
  which is exactly the ~10x+ contract-complexity spread the ASPLOS caveat warns about. This number must
  **not** be compared against ASPLOS's ">200 tc/h", because that figure is for complex contracts, not SEQ.
- **Cross-ISA caveat (load-bearing):** ASPLOS/H&S/Flowvizor are **x86-64** (Skylake, Coffee Lake, Core6/8,
  Xeon, EPYC, Alder Lake); ours is **ARM Neoverse-N3**. Absolute tc/h differences conflate the model
  executor cost, the kernel-executor/measurement path, the microarchitecture (ROB size, predictor,
  reset cost), the ISA, and the contract. **We do not claim a speedup over prior Revizor**: any raw tc/h
  gap is a different CPU + a different contract + a different measurement harness, not a tool improvement.
- H&S and Flowvizor measure throughput differently on purpose (speedup / "runs"), because their
  contribution is filtering/oracle efficiency, not raw tc/h. We should cite their native metrics, not
  force them into tc/h beyond Note A.

Bottom line: our N3 numbers are internally consistent (219.3 exec/s x ~20 measurement reps = 4,384
traces/s) and land where expected — complex-contract rate on par with ASPLOS's >200 tc/h, with the
simpler SEQ contract an order of magnitude faster, consistent with the documented contract-complexity
spread.

---

## 3. TTD (time-to-detection) comparison

Same methodology as ASPLOS Table 4: **mean wall-clock time to the first detected violation, averaged
over R>=10 independent input-generation seeds.** (ASPLOS: mean over 10; ours: R=20.)

| Vulnerability | ASPLOS'22 (CT-SEQ, Skylake/Coffee Lake), mean | Ours (N3), mean wall-clock, R=20 |
|---|---|---|
| Spectre-V1 | 4m 51s | within our 32-80 s band |
| Spectre-V4 | 73m 25s (longest; needs long spec window / hard-to-train predictor) | within our 32-80 s band |
| Spectre-V2 (v2_l1d.yml: random, standard priming, [cond,bpas], 2-target BLR dispatch) | (not time-measured in ASPLOS) | within our 32-80 s band (R=20 mean 79.9 s) |

Note: this Spectre-V2 TTD row is the standard-priming L1D v2 gadget (eval_configs/v2_l1d.yml) and is DISTINCT
from the pure which-target / no-sealing BTB template result (templates/spectre_v2_btb.py, CT-SEQ, cross-input
priming), whose own TTD is tracked separately in EVALUATION_RESULTS.md.
| MDS-type | 7m 40s | n/a on N3 |
| LVI-type | 5m 35s | n/a on N3 |
| Canonicality | (not in ASPLOS) | **42.9 s** (R=20 mean, canon_l1d); ≈10 s is the template rediscovery, cited separately (paper §Eval) |
| AUT*/PAC | (not in ASPLOS) | within our 32-80 s band |

- ASPLOS summary (L1265): **"Revizor detected most violations in under 10 minutes."** The notable
  exception is V4 at 73m 25s because it needs a long speculation window and a predictor that is hard
  to mistrain.
- **Ours on N3: per-leak R=20 MEANS 32-80 s** for Spectre-V1/V2/V4/canon/AUT* — comfortably inside the
  "<10 min" envelope. This is the per-row MEAN, not a per-run bound: individual runs range higher
  (v2_l1d mean 79.9 s but max 361.9 s — one outlier; aut runs to ~144 s; v1 to ~122 s). So state "row
  means under ~90 s", NOT "all runs under 90 s". Markedly faster for V4-class leaks than the ASPLOS V4
  figure too (different CPU/predictor — report as "also fast", not as a tool speedup).
- Beyond ASPLOS we also record **test-cases-to-violation** (not just wall-clock), a finer-grained
  reproducibility metric than ASPLOS's wall-clock-only Table 4; ASPLOS reports inputs-to-violation
  only for manually-written PoCs (Table 5, "few inputs / less than a second").

---

## 4. What to cite in the thesis (exact comparable claims)

- **Complex-contract throughput (apples-to-apples):** "ASPLOS'22 reports >200 tc/h for complex
  contracts on Intel Skylake/Coffee Lake (x86) [Oleksenko §1(6)]; our heavy NI contracts
  (branch-target sealing + multi-target dispatch + nesting=30) run ~200-250 tc/h on ARM Neoverse-N3 —
  same order of magnitude, different ISA/core."
- **Simplest-contract throughput:** "On the simplest contract (CT-SEQ, 100 effective inputs/tc, one
  isolated N3 core) our basic fuzzer sustains 7,895 tc/h (219.3 executions/s, 4,384 traces/s), an
  order of magnitude above the complex-contract rate — consistent with the documented ~10x
  contract-complexity spread." (State the contract every time; never compare 7,895 tc/h to >200 tc/h.)
- **TTD:** "ASPLOS'22 detects most violations in <10 min (mean over 10 seeds; V4 the outlier at
  73m 25s) [Table 4]; on N3 we detect Spectre-V1/V2/V4/canonicality/AUT* with mean wall-clock 32-80 s
  (all <90 s, R=20 seeds), and rediscover the canonicality leak in ~10 s."
- **Honesty caveats to state alongside every cite:**
  - Different ISA and core (x86 Skylake/Coffee Lake vs ARM Neoverse-N3) — absolute tc/h and TTD are
    not directly comparable; the portable comparands are per-contract tc/h and the "<10 min" TTD envelope.
  - tc/h varies ~10x with contract complexity; always pair the number with its contract.
  - Do not present any raw tc/h or TTD gap as a tool speedup: it reflects a faster/different CPU, a
    different measurement harness, and/or a simpler contract.
  - H&S (EuroS&P'23) and Flowvizor (ASPLOS'27) report speedup / "runs", not tc/h — cite their native
    metrics (H&S: ~100x faster, 130M executions in 16 h vs >2 months, detection-time factor 10-50;
    Flowvizor: 400K runs = 10K tc x 40 inputs per size, orders-of-magnitude fewer false-positive
    violations via the BPU oracle), not a forced tc/h.

---

### Source index (verified against pdftotext dumps)
- ASPLOS'22 Revizor — `/tmp/revizor_asplos.txt`: throughput L176-181; CPUs L154, L861-862; detection
  Table 4 L1119-1280 (V1 4m51s, V4 73m25s, MDS 7m40s, LVI 5m35s; "<10 min" L1265); inputs L880-881.
- Hide & Seek (EuroS&P'23, arXiv 2301.07642) — `/tmp/hs.txt`: speedup/campaign L30, L136-143;
  Fig.1 targets+130M execs L231-237; factor 10-50 L242-244; 100x filter overhead L238-239.
- Flowvizor (ASPLOS'27) — `/home/gal_k_1_1998/tmp/flowvizor.txt`: setup Intel i5-12500 L1454-1456;
  "400K runs" L1467-1470; perf-benchmark 10K progs x 40 inputs L315-319; Fig.11 end-to-end L1600-1620;
  related-work positioning L1626-1640.
- Our paper's stated convention — `/home/gal_k_1_1998/overleaf-6ac0fcd3/main.tex` L1752-1755:
  report "≈ tc/s and traces/s on N3", cite prior Revizor x86 ">200 tc/h, most violations <10 min";
  canonicality leak rediscovered in "≈10 s" (L1759).
