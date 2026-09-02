# MTE-NI campaign triage log (Pixel 8, core 8 / Cortex-X3, config_mte_sb1)

Method: verify-mte-tag-leak skill. Local accounting first; arch-prefetch-dominated divergence => noise
(no device); spec-demand-concentrated => controlled tag sweep (repeated). See memory
[[mte-decoy-nonrevert-and-scatter]] for the overall conclusion (no genuine TikTag-v1 found; violations
are prefetcher/measurement noise).

## Standing conclusion
Two suspects proven noise via controlled tag sweep (MATCH tag never separates): inp14 (143526, x0) and
inp0 (154155, x4). A speculative tag mismatch does NOT cause v1 speculation-shrinkage here. Definitive
next step (not done): official TikTag PoC as a positive control.

## Per-violation verdicts
| violation | counterexamples | perturbed reg | verdict | basis |
|---|---|---|---|---|
| 143526 | #14 / #93 (same inp14) | x0 | NOISE | controlled tag sweep: MATCH not separated (uniform) |
| 154155 | #0 / #100 (same inp0) | x4 | NOISE | controlled tag sweep: MATCH sum 0.21 vs mismatch mean 0.22 |
| 163059 | #32 / #125 (same inp32) | x4 | NOISE | local: 14/15 diverging sets are ARCH-demand prefetch streams (tag-independent) |
| 170010 | #1 / #101 (same inp1) | x2 | NOISE | local: all 5 diverging sets are arch demands / arch-prefetch streams (spec demands 50,51 untouched) |
| 174457 | #52 / #147 (same inp52) | x4 | NOISE | local: both diverging sets {19,21} are prefetch of arch demand 16 (tag-independent); spec demands untouched |

(cross-input data-leak false positives 142642/143053 from the pre-fix run are eliminated by the
same-input NI class-key fix.)

## Operational notes
- 2026-09-02 ~17:0x: campaign died mid-run from the bad-magic transport bug (run_batch decode) even at
  superbatch_size=1 — a TC's slow-path sample-500 over ~200 boosted inputs makes the device response
  large enough to desync. This recurs. Mitigation: auto-restart runner (campaign_runner.sh) that reloads
  the executor module and restarts the campaign on each crash, with -i lowered to 60 to shrink batches.
  The health monitor is keyed on the runner PID (survives per-campaign restarts). Real fix still pending:
  the remote batch transport (see memory superbatch-badmagic-remote).

## A715 middle core (cpu6) run
| violation | counterexamples | reg | verdict | basis |
|---|---|---|---|---|
| 175332 | #71 / #171 (same inp71) | x1 | NOISE | local: diverging sets are arch-prefetch streams + unaccounted; CE spec-only EMPTY (perturbed x1 access unreached). Same prefetcher-noise as X3. |

## 2026-09-02 — KASAN "bad magic" bug FIXED (page_kasan_tag_reset)

**Root cause** (see memory kasan-mte-kernel-badmagic): the Pixel 8 kernel is a userdebug build with
CONFIG_KASAN_HW_TAGS. The page allocator MTE-tags the sandbox pages; KASAN then reports the executor's
own per-input STG retags and deliberate decoy tag-mismatch accesses as `BUG: KASAN: invalid-access`,
which aborted the measurement -> malformed batch response -> "not a batch response (bad magic)".
Content-dependent -> intermittent; superbatch=100 hit a triggering input on the first window every time.

**Fix**: `mte_alloc_tagged_region` now calls `page_kasan_tag_reset()` on every page of the region under
`#ifdef CONFIG_KASAN_HW_TAGS`, marking the sandbox match-all so KASAN stops intercepting the executor's
MTE tag checks. The fuzzer fully owns the region's tags again.

**Empirical proof** (built on host B, pushed to device, reloaded):
- Deterministic repro config docs/tiktag_mte_ni/config_mte_kasan_repro.yml (seeds program:212904
  input:939371555, superbatch=100) that ALWAYS bad-magic-crashed before -> now completes.
- 3 full runs: rc=0, Hardware Tracing Errors: 0, bad-magic/Traceback lines: 0, module stays loaded.
- Device dmesg: run1 had a single transient KASAN advisory (ptr8/mem6, before the region settled),
  run2 & run3 had ZERO KASAN BUGs. Steady state clean.

**Consequence for the TikTag hunt**: the earlier "experiment invalid on this KASAN kernel" worry is
resolved. With the region excluded from KASAN, the executor's MTE tag checks run natively (KASAN no
longer hijacks the exact tag faults we measure). The big superbatch=100 TikTag campaign can now run
headless on the real device. Enables config_mte_pixel_tiktag.yml (X3 core 8, cond+bpas, F+R).
