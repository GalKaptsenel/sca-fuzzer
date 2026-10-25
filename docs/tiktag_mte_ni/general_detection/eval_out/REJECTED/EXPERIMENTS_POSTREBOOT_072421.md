# Post-reboot experiments on violation fuzzer_072421

Pixel 8, Cortex-X3 (core 8, MIDR 0x411fd4e0). Executor module md5 5c8dfec, ABI 2, view_rotation
on. Config docs/tiktag_mte_ni/general_detection/s24_mte_random.yml, suspect input #1. These runs
are the first measurements of this violation after the device reboot and a fresh module load.

Protocol for every measurement: a fixed prefix of 14 genuine variants of other inputs, the
suspect at the tail slot, and only the suspect seal's pointer tag changed between runs. Each
reported value is the mean over the stated repetitions at the tail. The memory tag is 6, the
genuine uimm4 is 7, so the base pointer tag is 15 and pointer tag = (15 + uimm4) mod 16. The
access matches at pointer tag 6 (uimm4 7). Tag 15 (uimm4 0) is the one unchecked value under
TCMA1. Seals: A = ADDG@0xf8 (x4), B = ADDG@0x10c (x1), D = ADDG@0x120 (x2, LDP target granule 305).

## 1. Re-confirmation across the reboot (Flush and Reload)

ctrl_pos_tag on suspect 1, sets 1, 4, 8, 9, decoy noncanon, 3000 repetitions, prefix 14, three
independent repetitions. gen@tail is the match residency at the fixed tail slot, dec@tail the
noncanon mismatch at the same slot.

| set | rep 1 gen/dec (Δtag) | rep 2 gen/dec (Δtag) | rep 3 gen/dec (Δtag) | Δpos |
|-----|----------------------|----------------------|----------------------|------|
| 1   | 0.00 / 0.99 (+0.99)  | 0.00 / 0.99 (+0.99)  | 0.00 / 1.00 (+1.00)  | ~0   |
| 4   | 0.07 / 1.00 (+0.93)  | 0.02 / 1.00 (+0.98)  | 0.05 / 1.00 (+0.95)  | ~0   |
| 8   | 0.05 / 1.00 (+0.95)  | 0.01 / 1.00 (+0.99)  | 0.01 / 1.00 (+0.99)  | ~0   |
| 9   | 0.02 / 1.00 (+0.98)  | 0.00 / 1.00 (+1.00)  | 0.00 / 1.00 (+1.00)  | ~0   |

The present-on-mismatch block reproduces on the fresh boot and the fresh module load. The match
path is cold at these sets and the mismatch path reaches full residency. The head versus tail
difference for the genuine variant is at most 0.07, so the divergence is held by the tag at a
fixed batch position. These values match the pre-reboot run in POS_072421.txt.

## 2. Prime and Probe

Same comparison under Prime and Probe (executor_mode P+P, which rebuilds the harness for the new
template). A set bit here means the primed set was evicted during the victim. Three repetitions,
a wide probe list to show the eviction extent. measurement_mode was returned to Flush and Reload
after this section.

On the mismatch every probed set reads 1.00 in all three repetitions, from set 0 through set 63,
including the far sets 16, 20, 24, 28, 32, 40, and 48. The mismatch evicts the whole probed cache.

On the match the probed sets split into two groups, stable across the three repetitions:

- Evicted only on the mismatch (cold on the match): 1, 4, 5, 6, 8, 9, 10, 11, 16, 20, 24, 28, 32,
  40, 48. Δtag at these sets is +0.83 to +1.00.
- Evicted on both the match and the mismatch: 0, 2, 3, 7, 12, 56, 63. These carry the baseline
  footprint of the test case (the contract executor places architectural accesses at 3, 7, 63 and
  speculative accesses at 0, 12).

For the four sets of section 1 the match residencies are set 1 around 0.15, set 4 around 0.03,
set 8 around 0.01, set 9 around 0.10, against 1.00 on the mismatch. The mismatch channel is
present under Prime and Probe with the same present-on-mismatch direction as Flush and Reload.
The mismatch eviction is broad rather than confined to the block seen under Flush and Reload.

## 3. Performance counters on the match and the mismatch

A test module was built that reprograms three event selectors and nothing else. The reload hit
test counter (PMEVTYPER0) is unchanged, so Flush and Reload continues to work. The three reported
counters become PMEVTYPER1 = INST_RETIRED (0x08) read as pfc[0], PMEVTYPER2 = EXC_TAKEN (0x09)
read as pfc[1], PMEVTYPER3 = MEM_ACCESS (0x13) read as pfc[2]. The build matches the device
vermagic 5.15.137-android14-11-gbc062a78e195. The canonical module was backed up on the device,
the test module was loaded from a separate path, and the canonical module was restored afterward.

Predicted, before measuring: INST_RETIRED identical on the match and the mismatch if the mismatch
work does not retire, EXC_TAKEN near zero on both if no exception is taken, MEM_ACCESS near 12 on
the match.

### 3a. Seal D, gate 0x120, memtag 6, 2000 repetitions, three repetitions each

| uimm4 | ptag | class     | INST_RETIRED (pfc0) | EXC_TAKEN (pfc1) | MEM_ACCESS (pfc2) |
|-------|------|-----------|---------------------|------------------|-------------------|
| 7     | 6    | match     | 84.00 / 84.00 / 84.00 | 0.00 / 0.00 / 0.00 | 12.00 / 12.00 / 12.00 |
| 0     | 15   | unchecked | 84.00 / 84.00 / 84.00 | 0.00 / 0.00 / 0.00 | 12.00 / 12.00 / 12.00 |
| 8     | 7    | mismatch  | 193878 / 194702 / 197227 | 1.01 / 1.02 / 1.00 | 54777 / 54968 / 55499 |

### 3b. Seal A, gate 0xf8, sixteen pointer tag deltas, 2000 repetitions, one repetition

Two deltas read the baseline 84 INST_RETIRED, 0.00 EXC_TAKEN, 12.00 MEM_ACCESS, width 9: pointer
tag 6 (the match) and pointer tag 15 (the one unchecked tag). The other fourteen deltas read
INST_RETIRED 189936 to 199194, EXC_TAKEN 1.00 to 1.01, MEM_ACCESS 53976 to 56050, width 17.
Pointer tag 0 reads the high values, so tag 0 is checked on this hardware.

### 3c. What the counters show

MEM_ACCESS on the match is 12, equal to the test case's architectural memory-operation count, so
the counter window covers one pass of the test case body and the reload is outside it. The
mismatch MEM_ACCESS near 55000 reproduces the earlier figure under a different counter, so that
figure is not a counter-index artifact.

On the match and at the unchecked tag the window retires 84 instructions, takes no exception, and
issues 12 memory accesses. On every genuine mismatch EXC_TAKEN rises to one, INST_RETIRED rises to
about 195000, and MEM_ACCESS rises to about 55000. The unchecked tag carries a pointer tag that
differs from the memory tag yet is exempt from the check, and it stays at the baseline. The rise
therefore follows the tag check, not the pointer value.

INST_RETIRED counts architecturally retired instructions and EXC_TAKEN counts exceptions taken.
On the mismatch both rise together with MEM_ACCESS. The divergent work is on the architecturally
retired path, entered through one synchronous exception. The exception is the MTE synchronous tag
check fault on the sealed tagged access, taken into the executor's fault handler, and the handler
retires about 195000 instructions and issues about 55000 memory accesses before the window ends.
The contiguous block of filled sets is the memory footprint of that retired path.

This revises the earlier description of the transmit as a speculative access burst that never
retires. The retirement and the exception are tag check gated. They appear at every genuine
mismatch and are absent at the match and at the unchecked tag, in the same pattern at seal A and
seal D. The tag dependence and the tag check causality established earlier by the residency
controls and the tag-check-off collapse are unchanged. The counter evidence places the mechanism
on the architecturally retired fault path rather than a non-retiring speculative window.

## 4. Device state after the experiments

The canonical module was reloaded from /data/local/tmp/revizor-executor.ko. The file reads md5
5c8dfec2646f075c0328a6de2fa77602, /dev/executor is present, the ABI version is 2, measurement_mode
is Flush and Reload, and view_rotation is on. A canonical run at gate 0x120 reads match
L2D_CACHE_REFILL 8.70, MEM_ACCESS 12.00, BR_MIS_PRED 5.00 and mismatch L2D_CACHE_REFILL 93.64,
MEM_ACCESS 54568, BR_MIS_PRED 100.82, which is the canonical event programming with BR_MIS_PRED
back on pfc[2]. The test module and the on-device backup copy were removed.
