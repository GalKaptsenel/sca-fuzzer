# MANIFEST — general precondition-gated leak detection (TikTag-v1 MTE focus)

Authoritative index of everything produced. Full step-by-step evidence (hypothesis/command/verbatim
output/interpretation for every experiment) is in `LOG.md`; the narrative summary is in `RESULTS.md`.
All paths are absolute. No existing repo file was modified; all artifacts are new files under this dir
(`/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/`).

## Environment / device (for reproduction)

- Repo: `/home/gal_k_1_1998/revizor/sca-fuzzer`  (ISA spec `/home/gal_k_1_1998/revizor/sca-fuzzer/base.json`)
- venv python: `/home/gal_k_1_1998/revizor/revizor-venv/bin/python`
- Device: Pixel 8, Cortex-X3 = core 8, via adb server 127.0.0.1:5037 (ppadb). Serial 47061FDJH001K3.
- Kernel executor: `/dev/executor`; sysfs `/sys/executor`; userland `/data/local/tmp/executor_userland`;
  module `/data/local/tmp/revizor-executor.ko`.
- Sandbox base CHANGES per module load; read live each run:
  `su -c 'cat /sys/executor/print_sandbox_base'`  (was 0xffffff8873821000 this session).
- Wedge recovery: `su -c 'rmmod revizor_executor; insmod /data/local/tmp/revizor-executor.ko; chmod 777 /dev/executor; chmod -R a+rw /sys/executor'`
- htrace readout: reload_timer_shift = -1 => present bit = L1-resident; set N at printed-string index 63-N.

## The working detector (reproduce)

Reliable general MTE/TikTag-v1 detector = planted timing-critical core + random tails:
- template: `/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/general_mte_alu.asm`
- config:   `.../general_detection/s1_mte_hw_hard_noMT.yml`  (flush ON, mistraining OFF, chi2 analyser,
             inputs_per_class 4, sample [2000], cond clause, MTE+BASE categories, pinned core 8)
- run:
  `cd /home/gal_k_1_1998/revizor/sca-fuzzer && REVIZOR_LOG=0 \
   /home/gal_k_1_1998/revizor/revizor-venv/bin/python revizor.py tfuzz -s base.json \
   -t docs/tiktag_mte_ni/general_detection/general_mte_alu.asm \
   -c docs/tiktag_mte_ni/general_detection/s1_mte_hw_hard_noMT.yml -n 1 -i 8 -w <workdir>`
- Reliability: 3/3 fuzzer runs report a violation (general_mte_alu) and 3/3 (general_mte, memory tails).
- Minimal Step-1 equivalents (no random tails): `s1_tiktag_hard.asm` (det window) / `s1_tiktag_win.asm`.

## CONFIG INVARIANTS (load-bearing; a confound cost several false negatives — see LOG Exp22)

- `enable_pre_run_flush: TRUE` is REQUIRED (F+R needs the probe line flushed each run). `s1_mte.yml` has
  it FALSE and must NOT be used for HW measurement — it produced false "no divergence" results.
- `enable_branch_mistraining: FALSE`. Mistraining is deprecated; the detector uses natural cold
  misprediction of the arch-taken forward branch (body on fall-through). 3/3 with mistraining OFF.

## Templates (codes) — absolute paths + what each is

Dir prefix: `/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/`
- `general_mte_alu.asm`  — WORKING. Planted core (warm/zero guess, 24-SDIV window, arch-taken CBNZ with
  body on fall-through, guess load, AND-xzr data-dependent gate, fixed probe set 32) + random ALU tails
  (`.random_instructions.16.0.*`, n_mem=0). 3/3. Genuine by tag-sweep.
- `general_mte.asm`      — WORKING. Same core, random MEMORY tails (n_mem=8). 3/3; shrink + pollution noise.
- `s1_tiktag_win.asm` / `s1_tiktag_hard.asm` — the Step-1 planted gadget (input-dep / det window). Genuine.
- `thin_mte_randbody.asm` — window+branch planted, body 100% random (64 instr/40 mem). Fuzzer 0; direct
  measure shows genuine seal-driven but GAIN-flavor divergence (not the clean shrink).
- `thin_mte_collector.asm` — random body + a generic collector probe. No reliable signal.
- `user_mte_skeleton.asm` — the user's skeleton (random/window/random/if{random/sealed-load/random}/random).
  Fuzzer 0 (no tuned downstream).
- `user_mte_skeleton3.asm` — user's skeleton with warm/window/sealed-load planted, downstream left random
  (24 instr/12 mem). (under test, batch b78ycq7s4)
- `mte_tiktag_pytmpl.py` — Python object-template attempt. Blocked by Builder `if_` (body in TAKEN block;
  cannot place body on fall-through). Measured 0.
- `s1_dag_neverarch.asm` / `s1_dag_bodyarch.asm` — eligibility control pair (neverarch eligible, bodyarch not).
- `s1_mte_loop.asm` / `s1_mte_neverarch.asm` — the REJECTED looped templates (sealer requires a DAG CFG).
- `s2_canon*.asm`, `s3_canon_randbody.asm`, `thin_canon.asm` — canonicality (Step-2/3) templates (generalize).
- `v4_mte.asm` / `v4_mte_clean.asm` — Spectre-v4 store-bypass gadgets (no clean MTE shrink; confirmed negative).

## Configs (.yml) — absolute paths

- `s1_mte_hw_hard_noMT.yml` — THE working config (flush ON, mistraining OFF, inputs_per_class 4, sample [2000]).
- `s1_mte_hw.yml` / `s1_mte_hw_hard.yml` — flush ON, mistraining ON (earlier runs).
- `s1_mte.yml` / `s1_mte_hw_noMT.yml` — flush OFF (DO NOT USE for HW measure; confound).
- `s2_canon_hw*.yml` — canonicality NI configs. `s4_canon_random.yml` / `s4_mte_random.yml` — fully-random.
- `v4_mte.yml` — cond+bpas, SSB enabled.

## Scripts (.py) — absolute paths + purpose

- `probe_eligibility.py` — device-free CE probe: per MTE/Canon site prints (value, alts, speculative) +
  has_decoy + genuine-vs-decoy relocation diff. Presets executor._sandbox_base (no device). Handles .asm;
  measure_variants handles .py.
- `measure_variants.py` — device: measure baseline / decoy0.. / noncanon variant htraces; prints per-L1D-set
  residency and diverging sets. Handles .asm and .py templates.
- `sweep_s2.py` — device: sweep gap/ldnum/sdiv; prints probe-set-32 residency base/decoy/noncanon per input.
- `tag_sweep.py` — device: GENUINENESS control. Patches the speculative guess-retag ADDG uimm4 to each
  delta 0..15 and measures probe set 32. Genuine = present only at matching tag + TCMA-unchecked (0/15).
- `gen_s2.py` — generator for the planted-core templates (--ldnum --gap --sdiv --warm/--no-warm --det-window).

## Results (eval_out/) — absolute paths

Dir prefix: `.../general_detection/eval_out/`
- `win1/violation-261022-141431/` — first fuzzer-reported MTE violation (Step 1).
- `hard_r1..4/` — Step-1 hardened hit-rate (3/4). `noMT_r1..3/` — mistraining-OFF (3/3).
- `rel_general_mte_alu_1..3/`, `rel_general_mte_1..3/` — general-template reliability (3/3 each).
- `canon1/`, `step3canon/`, `step4canon/` — canonicality Step-2/3/4 (fully-random canon 1/40).
- `step4mte/` — fully-random MTE (0/150). `tag_sweep_win.txt` — Step-1 tag-sweep raw.
- `try_user_mte_skeleton/`, `try_user_mte_skeleton3/` — user-skeleton extended runs (batch b78ycq7s4).
- helper scripts: `hits.sh`, `genrel.sh`, `step4.sh`, `mte_random.sh`, `trymine.sh`.

## Triage verdicts (correct-vs-incorrect-tag sweeps) — see LOG Exp8, Exp24

- Step-1 gadget / general_mte_alu / general_mte: GENUINE TikTag. Probe set 32 resident ONLY at the
  matching tag (delta 0) and TCMA-unchecked tags (0b0000/0b1111); suppressed at every genuinely-checked
  mismatch. Not noise, not prefetcher, not batch contamination.
- randbody / user_skeleton: seal-driven GAIN-flavor divergence (mistag enlarges the speculative footprint),
  NOT the clean downstream-probe shrink; fuzzer-unreliable.

## Reasoning summary (full detail in LOG.md)

- MTE sealer requires a DAG CFG (`primitives.py _topo_sort`): looped templates are rejected at seal time
  (so the assumed internal-loop training is unavailable through the sealer).
- Eligibility: the gated load must be reached ONLY speculatively (CE min-nesting>0). Put the body on the
  branch's arch-NOT-taken path (fall-through, forced via `ORR x9,#1; CBNZ x9,.skip`).
- Window: natural cold misprediction of the late-resolving arch-taken forward branch (no loop, no
  mistraining). v4/store-bypass window does NOT give the MTE shrink (control-flow window required).
- Signal: MTE's tag check is post-allocation, so the checked load's OWN line fills regardless of tag; the
  leak is a DOWNSTREAM probe gated by window length. Must track a FIXED probe set at the window EDGE (gate
  length ~ window), data-dependent on the guess. This is timing-critical, so random generation almost
  never reproduces it (fully-random MTE 0/150) and it does not survive randomizing the downstream.
- Builder `if_` (Python templates): VERIFIED it places the body in the TAKEN block (fall-through is just
  `B join`); it cannot express the fall-through-body branch the cold-misprediction window needs. The CAUSE
  of the Python-template 0 is UNTESTED (consistent with fixed-single-PC no-mispredict and with an
  unverified cold-prediction-direction assumption). A fall-through-body Builder primitive would be a
  shipped-source change — DEFERRED by coordinator/user; not made.

## Exact per-run state (memory / registers / seeds) for reproduction

- Each fuzzer workdir has `rerun.yaml` with the pinned program_generator_seed + input_gen_seed, and the
  exact re-run command — rerunning reproduces the identical programs and inputs.
- Each saved violation dir (e.g. eval_out/win1/violation-261022-141431/) contains the `*.reif` input
  files = the exact initial sandbox MEMORY + register (x0..x5 + NZCV + sp) + MTE-tag state each input ran
  with, plus `input_*_trace_log.txt` (the assembled instruction stream incl. sealer-inserted ADDG retags
  and sandbox clamps), `minimize.yaml`, and `org-config.yaml`.
- Device sandbox base is NOT stored in artifacts (it changes per module load); read live with
  `su -c 'cat /sys/executor/print_sandbox_base'` before replaying by hand.
- Direct measurements (measure_variants/tag_sweep/sweep_s2) print input seeds in their output; set the
  same seed path via the input generator to reproduce a specific input's residency.

## Addendum — user-skeleton triage + new triage tool

- `triage_sweep.py` — DECISIVE per-violation 16-tag sweep on the REAL saved input: loads generated.asm +
  input_<idx>.reif, knocks out each sealed-load retag ADDG to find the load feeding the diverging set,
  then sweeps that load's tag 0..15 at the diverging set. Usage:
  `triage_sweep.py <cfg> <violation-dir> <suspect_idx> <diverging_set> [--reps N]`
  (Replaces the stale verify-mte-tag-leak accounting.py S2 / ctrl_sweep.py which expect legacy .bin and
  a removed `_pac_keys` attribute.) Note: .reif->.bin symlinks were added in the 3 violation dirs so the
  skill's S1 (tag-only diff) could run.
- user_mte_skeleton.asm violations: eval_out/try_user_mte_skeleton/violation-261022-{155905,160128,160705}.
  VERDICT (16-tag sweep on input #5, 3 passes): 155905 GENUINE (only match tag leaks, 1.00 vs ~0);
  160128 GENUINE (match + unchecked-0 high, checked mismatches ~0); 160705 NOISE (uniform).
  => a mostly-random skeleton (window + one sealed load planted, rest random) DOES auto-detect genuine
  TikTag at a low rate (2 genuine / 40 fills). Corrects the earlier single-run dismissal.
- helper: eval_out/triage3.sh (sweep all 3), eval_out/recheck.sh (stability passes).
