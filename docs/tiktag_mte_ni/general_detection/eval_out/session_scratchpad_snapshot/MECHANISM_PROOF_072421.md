# Mechanism of violation fuzzer_072421 — on-device measurement

Pixel 8, Cortex-X3 (core 8, MIDR 0x411fd4e0). Executor module md5 5c8dfec, ABI 2, Flush and Reload,
view_rotation on. Config docs/tiktag_mte_ni/general_detection/s24_mte_random.yml. Suspect input #1.

Measurement protocol: a fixed prefix of 14 genuine variants of other inputs, the suspect at the tail
slot, and only the suspect seal's ADDG pointer-tag field (uimm4 at the gate offset) changed between runs.
Each reported line is the mean over 2000 repetitions at the tail, and every condition was taken at 3
independent repetitions. Per-measurement PMU counts read from the victim window: pfc0 = L2D_CACHE_REFILL,
pfc1 = MEM_ACCESS (demand accesses), pfc2 = BR_MIS_PRED_RETIRED. The memory tag is 6, the genuine uimm4 is
7, so base pointer tag is 15 and pointer tag = (15 + uimm4) mod 16. The access matches at pointer tag 6
(uimm4 7). Only tag 15 is Unchecked here (TCMA1), so pointer tag 15 (uimm4 0) is the one unchecked value
and every other tag is a genuine mismatch. Seals: A = ADDG@0xf8 (x4), B = ADDG@0x10c (x1), D = ADDG@0x120 (x2).

## 1. What the stream is

### 1a. PMU differential (no kernel change)

Seal D @0x120, probe sets 1,3,4,9,12,20,28,40:

| uimm4 | ptag | class    | block width | set3 | L2D_CACHE_REFILL | MEM_ACCESS | BR_MIS_PRED |
|-------|------|----------|-------------|------|------------------|------------|-------------|
| 7     | 6    | match    | 9           | 1.0  | 8.7              | 12.0       | 5.0         |
| 0     | 15   | unchecked| 9           | 1.0  | 8.5              | 12.0       | 5.0         |
| 8     | 7    | mismatch | 28          | 0.0  | 91.5             | 54273      | 104         |

Seals A and B at a mismatch give the same picture: block width 17, L2D_CACHE_REFILL 83 to 87,
MEM_ACCESS 53900 to 55100, BR_MIS_PRED 102 to 108.

Prediction, if the block were a hardware prefetch stream: on a mismatch L2D_CACHE_REFILL rises while
MEM_ACCESS (demand) does not rise with it, because prefetch fills are not demand accesses.

Measured: on a mismatch MEM_ACCESS rises from 12 to about 54000 (a factor of ~4500), while
L2D_CACHE_REFILL rises from 8.7 to about 90 (a factor of ~10). The match MEM_ACCESS of 12 equals the test
case's architectural memory-operation count, which fixes the counter mapping and shows the bb_0.7/0.8
block does not execute on a match. The ~54000 is near constant across the three seals and across both
block widths, so it does not scale with the block. This is the opposite of the prefetch signature: the
block is accompanied by a large access burst that is almost entirely L1-resident (few L2 refills), not by
refills in excess of demand. PMU event 0x13 alone therefore does not show a prefetcher. It shows a
speculative access burst that appears exactly with the block.

### 1b. Prefetcher disable (CPUECTLR / CPUACTLR)

On the Cortex-X3 the data prefetcher control is CPUACTLR6_EL1 (S3_0_C15_C8_1) bit 41, the bit the Arm
erratum 2641945 and CVE-2024-7881 workarounds set to turn off the affected L1 data prefetcher. It is not
CPUECTLR_EL1. A test build with these accessors and a pinned-core probe was prepared and loaded. The
direct disable measurement is not included here: an exception taken inside a sysfs handler of an
out-of-tree module on this kernel leaves the sysfs node permanently referenced and wedges module unload,
so the bit-41 measurement is not completable on this device within the single-device safety budget. The
prefetcher's role in shaping the block into a contiguous run therefore remains inferential (see section 4).

## 2. Set 3 — direction and cause

### 2a. Direction (16 tag deltas, seal D)

Set 3 is resident only at pointer tag 6 (match, uimm4 7) and pointer tag 15 (unchecked, uimm4 0). At the
14 genuine mismatches set 3 is evicted. Set-3 eviction coincides one to one with the wide block (width 28,
sets 2 to ~30) and the ~54000 MEM_ACCESS burst. At the two resident deltas the footprint is width 9 and
MEM_ACCESS is 12. Direction: resident at match, evicted at mismatch.

### 2b. Cause (granule retag)

Retagging seal D's own granule (granule 305, the LDP target) to tag 9 moves set 3's resident delta with
it: set 3 reads 0.00 at pointer tag 6 and 1.00 at pointer tag 9. Retagging set 3's own granule (granule
15, the STP@0x58 target) to tag 9 does not reproduce this move. Set 3's own granule tag is constant across
the entire 16-delta sweep, yet set 3's state follows seal D's tag. Set 3 is not itself tag-checked. It is
evicted as collateral of seal D's speculative stream, which is gated by seal D's check on granule 305.

### 2c. Under the prefetcher disable

Not measured, for the reason in 1b.

## 3. Why a mismatch triggers the stream

view_rotation off (enable_pre_run_flush 0, confirmed on device), seal D:

| condition                         | match (uimm4 7)        | mismatch (uimm4 8)                 |
|-----------------------------------|------------------------|------------------------------------|
| view_rotation on                  | width 9, MEM 12, BR 5.0| width 28, MEM 54k, BR 104          |
| view_rotation off                 | width 9, MEM 12, BR 0.7| width 28, MEM 54k, BR ~80          |
| view_rotation off + no mistraining| width 9, MEM 12        | width 28, MEM 54k                  |

Turning view_rotation off, and additionally stripping the per-input branch mistraining, does not remove
the divergence. The window into bb_0.7/0.8 is present on both the match and the mismatch (the match
MEM_ACCESS of 12 shows bb_0.7/0.8 never retire, so they run only under speculation in both cases). The
trigger is the tag mismatch inside that speculative window, not the view rotation. The only effect of the
window knob is on the match path's retired mispredicts, which fall from 5.0 to 0.7 when the branch
predictor is no longer rotated. BR_MIS_PRED_RETIRED on the mismatch path (about 100) stays far above the
match path, and the mismatch path issues the ~54000-access burst, so the mismatch path is much longer.

## 4. Combined picture

Stream present = wide contiguous block in the Flush and Reload trace (seal D width 28, seals A and B width 17).

| match/mismatch | window | tag check | stream | L2D_REFILL | MEM_ACCESS | BR_MIS_PRED |
|----------------|--------|-----------|--------|------------|------------|-------------|
| match          | on     | on        | no     | 8.7        | 12         | 5.0         |
| unchecked (t15)| on     | n/a       | no     | 8.5        | 12         | 5.0         |
| mismatch       | on     | on        | yes    | 90         | 54000      | 104         |
| match          | off    | on        | no     | 8.8        | 12         | 0.7         |
| mismatch       | off    | on        | yes    | 85         | 54000      | ~80         |
| match/mismatch | on     | off (TCO) | no     | -          | -          | -           |
| any            | prefetcher off | -  | not measured | - | - | - |

The tag-check-off row is the prior TCO result (tag checks disabled collapses the divergence). The
prefetcher-off row is not measured (section 1b).

Proven mechanism: the three seals' pointer-tag checks run inside one speculative window (bb_0.7/0.8, reached
only by the mispredicted TBZ, never retired). On a tag match, or at the one Unchecked tag, the window is
short and touches a width-9 footprint with 12 demand accesses. On a tag mismatch the same window becomes
long: it issues about 54000 speculative accesses and about 100 retired mispredicts, and fills a contiguous
block of L1D sets whose span is set by which seal mismatched (seal D the widest). The divergence is strictly
tag-gated: it appears at every genuine mismatch and is absent at the match and at the Unchecked tag, it
tracks the checked granule (seal D's granule 305, shown by retag), and the prior tag-check-off result
removes it. It is not gated by view_rotation. The architectural demand at set 3 is evicted as collateral of
this burst, not by any check of its own granule. Inferential: whether the hardware data prefetcher is what
turns the mismatch burst into a single contiguous run of sets, as opposed to the burst's own access
pattern, is not settled here. The L2D_CACHE_REFILL-to-MEM_ACCESS ratio (refills far below demand accesses)
is not a prefetch-dominated pattern, which argues the contiguity is largely the burst itself, but the
direct CPUACTLR6_EL1 bit-41 disable that would settle it is not completable on this device (section 1b).
