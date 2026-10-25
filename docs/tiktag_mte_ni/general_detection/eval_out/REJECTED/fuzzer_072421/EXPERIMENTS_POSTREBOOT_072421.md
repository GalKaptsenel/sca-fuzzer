# 072421 post-reboot experiments — architectural MTE tag-check fault (not speculative)

Device: Pixel-8 Cortex-X3, module md5 5c8dfec, abi 2, F+R, view_rotation on, pinned core 8.
All three experiments ran after the GCP-box reboot, device left canonical, campaign paused.

## EXP1 — re-confirm across reboot (ctrl_pos_tag, F+R, sets 1/4/8/9, noncanon, 3000 reps x3)
Survived. match 0.00-0.07, mismatch 0.99-1.00, Δtag +0.93..+1.00, Δpos ~0. 3/3.
Raw: POSTREBOOT_RECONFIRM_072421.txt

## EXP2 — Prime and Probe (3 reps)
Same direction as F+R. Saturation: every probed set (0..63) is 1.00 on mismatch; match lights only
the baseline sets 0,2,3,7,12,56,63. Consistent with a committed exception-handler footprint, not a
single speculative demand. Raw: POSTREBOOT_PP_072421.txt

## EXP3 — PMU (handler-safe module: only PMEVTYPER1/2/3 reprogrammed; no CPUACTLR/TCO/prefetcher writes)
Events: INST_RETIRED 0x08=pfc0, EXC_TAKEN 0x09=pfc1, MEM_ACCESS 0x13=pfc2. vermagic-matched,
restored canonical and reverified after.
- gate 0x120 (seal D), 3 reps: match and unchecked = 84 / 0 / 12 ; mismatch = ~194k-197k / ~1 / ~55k.
- gate 0xf8 (seal A), 16-delta sweep: all 14 genuine mismatches ~195k / ~1 / ~55k ; only ptag6 (match)
  and ptag15 (unchecked) stay at baseline 84 / 0 / 12 ; ptag0 jumps (tag 0 is checked on this HW).
- canonical reverify (BR_MIS_PRED back on pfc2): match 8.70/12/5.0, mismatch 93.6/54568/100.8.
Raw: POSTREBOOT_PMU_072421.txt

## Conclusion — architectural, confirmed
INST_RETIRED (0x08) and EXC_TAKEN (0x09) are retired/architectural-only; a squashed speculative access
increments neither. On a mismatch both jump (EXC_TAKEN 0->~1, INST_RETIRED 84->~195k), gated by the tag
check at the seals (baseline only at the matching tag and the single unchecked tag ptag15). The module
configures synchronous tag checking (TCF=sync), under which an MTE tag-check fault is precise and taken
only when the faulting access retires. So the sealed access retired and faulted architecturally.

The "speculative TikTag" label came from the CE per-instruction trace (input_0001_trace_log.txt), which
places the three sealed accesses at speculation nesting 1 and exits the architectural path at 0x0c0. The
hardware committed-instruction count (match=84) exceeds the CE committed path (48) by the size of
bb_0.7/0.8 plus the window fences, so on silicon the architectural path enters bb_0.7/0.8. The CE branch
resolution (the TBZ) diverges from hardware. Verdict triangulated: INST_RETIRED jump, EXC_TAKEN 0->1,
and committed-count = full gadget path.

What holds throughout: a genuine tag-controlled cache leak whose cause is the MTE tag check
(tag-check-off collapse holds). What is overturned is the mechanism label: on silicon the transmit is
architectural, not speculative.

## Residual (optional, does not change the verdict)
Reading ESR_EL1/ELR_EL1/FAR_EL1 in the measurement fault handler on a mismatch would name the faulting
instruction at the ISA level (expect ELR = the seal PC, EC = MTE tag-check). Needs one handler-only
module build (no dangerous register writes). Test-module source staged at
/home/gal_k_1_1998/pixel-mod-build/executor-src-pmutest (only pmu.c's three event selectors differ from
canonical).
