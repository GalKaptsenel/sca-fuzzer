# Verification of the TikTag-v1 executor detector

Verdict: the detection is genuine and the Revizor executor reads the hardware trace correctly. The
reported residency is a real MTE tag-check speculative leak on the Cortex-X3, not a measurement
artifact, prefetcher stream, batch contamination, or an htrace misinterpretation.

This document records the code audit and the on-device checks that establish that verdict, with the
commands and numbers behind each one. All runs are on core 8 (Cortex-X3) of the Pixel 8 at sandbox
base `0xffffff8873821000`.

## 1. Gadget audit (tiktag_v1_executor.py)

The test case is one program with an internal training loop. `gadget_asm` emits:

```
mov x3, #iters
.Lloop:
subs x3, x3, #1          # Z set on the final (attack) iteration
csel x6, x5, x1, eq      # guess pointer: attack = swept-tag x5, training = granule-tag x1
csel x10, x2, x4, eq     # probe pointer: attack = read set x2, training = throwaway set x4
cset x7, ne              # x7 = 1 on training, 0 on attack
ldr  x12, [x1]           # warm the guess granule through its own (matching) tag, architectural
str  x7, [x0]            # slow-load value = training flag
dc civac, x0 / x2 / x4   # flush the slow line and both probe sets
dsb sy ; isb
ldr  x8, [x0]            # slow load: training -> 1, attack -> 0
cbz  x8, .Lskip          # forward branch, trained not-taken by the training iterations
ldr  x9, [x6]   (xldnum) # tag-checked guess load(s)
orr  x10, x10, x9 (xgap) # serial latency chain feeding the probe address
ldr  x11, [x10]          # probe
.Lskip:
cbnz x3, .Lloop
```

Findings, each checked against the instruction stream:

- Train/attack schedule is correct. `subs` sets Z only on the last iteration, so `cbz x8` falls
  through on every training iteration (x8 = 1) and is architecturally taken to `.Lskip` on the attack
  (x8 = 0). The branch is trained to the not-taken direction and the attack speculates down the
  fall-through body. The window is widened by flushing the slow line each iteration so the branch
  resolves late.

- The swept (mismatched) load is speculative only and cannot fault. `csel x6, x5, x1, eq` selects the
  swept pointer x5 only on the attack iteration, and on that iteration the branch is architecturally
  taken to `.Lskip`, so the body that contains `ldr x9, [x6]` runs only on the mispredicted path. On
  every training iteration x6 is the matching-tag pointer x1 and the load is architectural and never
  faults. A mismatched tag is therefore never dereferenced architecturally.

- Match-during-training versus swept-on-attack split is clean. Training uses x1 (tag T) for the guess
  and x4 (throwaway set) for the probe. The attack uses x5 (swept tag) for the guess and x2 (read set)
  for the probe. The read set is written only by the attack iteration's speculation.

- The guess load's cache behavior is identical for match and mismatch, so only the tag check differs.
  x1 and x5 share the same address bits and differ only in the tag field in bits 56 to 59, which does
  not participate in cache indexing. Both point at the same physical line, warmed each iteration by
  `ldr x12, [x1]`. The guess load is an L1 hit in both cases. The sole difference between a matching
  and a mismatching sweep value is whether the MTE tag check passes, which is what gates the length of
  the speculative window. This rules out a cache-address explanation for the separation.

- The latency gate is a true data dependency. `orr x10, x10, x9` repeated `gap` times chains x10 (the
  probe address) through x9 (the guess-load result, which is zero), so the probe cannot issue until the
  chain retires. The probe issues inside the speculative window only when the window is long enough,
  which is the tag-dependent quantity.

- The probe-free separation is sound. Training probes the throwaway set (x4), the attack probes the
  read set (x2), and the two are distinct L1D sets. The read set carries only the attack access.

- The tag-sweep verdict is correct and its exclusion of tag 15 is load-bearing and justified. Tag 15
  (0b1111) is left unchecked by TCMA1 on the sandbox region, so its guess load is never tag-checked and
  its read set is always resident. The verdict computes the winner over the checked tags plus the
  sealed tag and requires the sealed tag to beat every checked mismatch by `--margin`. Excluding 15
  from the winner contest is necessary, because tag 15 would otherwise outscore the sealed tag. The
  genuineness of the leak does not rest on that exclusion: the decisive evidence (Section 3) is that a
  single checked tag survives and tracks the seal, which no uniform artifact can do.

Input construction (`make_input`) matches the gadget: x1 and x5 carry tag T and the swept tag in bits
56 to 59, `mte_tags` seals granule 0 to T and leaves every other granule at the unchecked tag 15.

## 2. Htrace interpretation

Reload semantics are confirmed from `src/aarch64/executor/templates_jit.c` (`reload`) and
`executor.h`:

- `reload_timer_shift < 0` (the default is -1, set by `set_regime`) selects the
  L1D_CACHE_REFILL hit test. The reload loads each recorded set, reads counter0 (event 0x03,
  L1D_CACHE_REFILL) before and after, and sets the set's bit iff the delta is zero. A zero delta means
  the load caused no L1 refill, so the line was already L1-resident. "present" = 1 means L1-resident.
- `reload_timer_shift >= 0` instead reads PMCCNTR and sets the bit iff the cycle delta is below
  `1 << shift`, a graded cycle threshold.

Bit to set mapping: the reload sets bit `idx` of the destination for set `idx`. Printed most
significant bit first as a 64-bit string, set N sits at string index `63 - N`, which is exactly what
`residency_rate` reads (`index = 63 - read_set`).

The reach/skip/spec control confirms the htrace resolves a reached versus not-reached probe and that a
purely speculative probe registers (read set 32, core 8):

```
reach  (architectural probe to set 32): residency 1.00
skip   (no probe):                      residency 0.03
spec   (match tag 5, probe speculative-only): residency 0.68
spec   (mismatch tag 0):                        residency 0.00
```

An architectural probe reads 1.00 and no probe reads 0.03, so the F+R readout cleanly separates the
two states. In the real gadget the probe is only ever on the mispredicted path, yet the matching-tag
run reads 0.68, which establishes that a speculative-only probe leaves an F+R trace and that a
mismatching tag suppresses it entirely (0.00).

## 3. Tag-tracking control (the decisive check)

Re-running the shipped detector for sealed tags 5, 9, 10:

```
python docs/tiktag_mte_ni/tiktag_v1_executor.py --repo /home/gal_k_1_1998/revizor/sca-fuzzer \
  --adb-host 127.0.0.1 --adb-port 5037 --core 8 --iters 512 --ldnum 16 --gap 80 \
  --read-set 32 --warm-set 48 --reps 60 --seal-tags 5,9,10
```

```
sealed granule tag 5
  0:0.02 1:0.02 2:0.02 3:0.00 4:0.03 5:0.62 6:0.05 7:0.00 8:0.02 9:0.00 10:0.02 11:0.02 12:0.00 13:0.00 14:0.00 15:0.77
  matching tag 5 residency 0.62  checked-mismatch max 0.05 mean 0.01  separation +0.57  DETECTED
sealed granule tag 9
  0:0.02 1:0.00 2:0.00 3:0.00 4:0.02 5:0.00 6:0.02 7:0.03 8:0.03 9:0.70 10:0.02 11:0.02 12:0.02 13:0.00 14:0.02 15:0.72
  matching tag 9 residency 0.70  checked-mismatch max 0.03 mean 0.01  separation +0.67  DETECTED
sealed granule tag 10
  0:0.00 1:0.00 2:0.02 3:0.02 4:0.02 5:0.02 6:0.00 7:0.02 8:0.03 9:0.02 10:0.77 11:0.00 12:0.00 13:0.03 14:0.00 15:0.70
  matching tag 10 residency 0.77  checked-mismatch max 0.03 mean 0.01  separation +0.73  DETECTED
```

The surviving checked tag is 5, then 9, then 10, following the sealed granule tag across reseals. Every
other checked tag stays at or below 0.05. Tag 15 stays high throughout, as expected for the
TCMA1-unchecked tag. A fixed measurement artifact, a prefetcher stream into set 32, or batch
contamination would light the same tags on every seal. A signal that moves with the sealed tag can only
come from the tag check, so the leak is genuine.

## 4. Independent soundness checks

### Read-set independence

Repeating the match-versus-mismatch sweep at read set 24 instead of 32:

```
0:0.05 3:0.00 5:0.65 9:0.00 15:0.80  (sealed T=5)
match 5=0.65  checked-mm max=0.05  tcma15=0.80  sep=+0.60
```

The surviving checked tag is still the sealed tag 5 and the signal now appears at set 24. The leak
follows the programmed read set rather than sitting at a fixed set 32, which rules out a set-local
artifact.

### Graded refill readout

Dumping `/sys/executor/reload_refills` for set 32 after a single measurement (columns are set,
L1 main, L1 faulty, L2 main, L2 faulty refill deltas, zero means an L1 hit):

```
match T=5    set32 = 32 0 1 0 0
mismatch 0   set32 = 32 3 1 1 1
tcma 15      set32 = 32 0 1 0 0
```

On a matching tag the read set is L1-resident (L1 delta 0). On a mismatch the read set misses L1 (L1
delta nonzero) and also refills L2 (L2 delta nonzero), so the probe genuinely did not run and the line
sits below L1. The binary present or absent readout faithfully reflects a real L1 versus deeper
distinction rather than a timing nuance.

### Clock resolution

The coordinator's lever is the reload clock. Repeating the sweep with a cycle-threshold reload instead
of the L1-residency reload:

```
shift=7 (<128 cyc = hit): match=0.85 mismatch0=0.12 tcma15=0.80
shift=8 (<256 cyc = hit): match=1.00 mismatch0=0.65 tcma15=1.00
shift=9 (<512 cyc = hit): match=1.00 mismatch0=0.82 tcma15=1.00
```

The L1-residency readout (`reload_timer_shift = -1`) gives the cleanest separation. A coarser cycle
threshold counts an L2-resident line as a hit, so the mismatch rate climbs as the threshold widens and
the match and mismatch cases converge. For this leak the probe either fills L1 or stays below it, so
L1-residency is the correct resolution. A finer cycle clock would matter for a leak that only shifts
latency within the cache hierarchy rather than gating the probe outright, which is a knob for the
general design but not a correction to this detector.

## 5. Weaknesses and caveats

- The match residency is roughly 0.6 to 0.8 rather than 1.0, because the speculative window covers the
  full chain and probe only on a fraction of runs. The separation from the checked mismatches
  (about +0.57 to +0.73) is far above the 0.3 margin, so this does not affect the verdict. It does mean
  the channel is probabilistic per run and needs the repeat loop.

- The verdict's winner contest excludes tag 15 on the strength of the TCMA1-unchecked fact. The
  tag-tracking control makes the conclusion independent of that fact, because only the sealed checked
  tag survives and it tracks the seal.

- The gadget flushes its own lines with `dc civac` and runs its own training loop, so it is a
  raw-executor harness rather than a Revizor template. The executor is used only as the measurement
  harness here. This is a property of the harness, not a defect, and the executor's F+R path reads it
  correctly as shown above.

- All measurements are single-core on the X3 with the device used exclusively. The channel is specific
  to the control-flow speculation window on this core.
