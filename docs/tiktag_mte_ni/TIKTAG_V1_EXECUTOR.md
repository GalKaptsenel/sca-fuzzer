# TikTag-v1 detection through the Revizor executor (Pixel 8 Cortex-X3)

`tiktag_v1_executor.py` detects the TikTag-v1 speculative MTE tag leak on the Cortex-X3 using the
Revizor kernel executor as the measurement harness. It loads one self-contained test case that carries
the whole measurement and reads the leaking cache set from the executor's Flush+Reload htrace.

## Gadget

The test case is a training loop whose final iteration is the attack. Each iteration:

- writes a training flag to the slow-load granule and flushes that line, the read set, and a throwaway
  set, so the forward conditional branch resolves late and the read set starts cold;
- on the training iterations the slow load returns one, the branch falls through, and the body runs
  architecturally with the granule's own tag, which trains the branch to the not-taken (fall-through)
  direction;
- on the attack iteration the slow load returns zero, so the branch is architecturally taken but is
  still predicted not-taken and speculates down the fall-through path.

On the speculated path a tag-checked load uses a pointer whose tag is swept, followed by a serial
`orr` chain that feeds the probe address and then the probe load. The chain is a pure latency
dependency: the probe cannot issue until the chain retires. A matching tag keeps the speculative
window open long enough for the probe to issue and fill its L1 line; a mismatching tag shrinks the
window so the probe never issues and its line stays absent. The probe address on the training
iterations points at a throwaway set, so the read set is touched only by the attack iteration's
speculation.

## Detection

Sweeping the pointer tag over all sixteen values, the read set stays L1-resident for the tag that
matches the sealed granule and is absent for every other checked tag. The tag `0b1111` is unchecked
under TCMA1, so its access is never tag-checked and it is excluded from the mismatch set. Resealing
the granule to a different tag moves the surviving checked tag with it. A sealed tag is reported
detected when its residency exceeds every checked mismatch by at least `--margin`.

Measured on the Cortex-X3 (core 8), sealing tags 5, 9, and 10 in turn:

```
sealed granule tag 5
  matching tag 5 residency 0.70   checked-mismatch mean 0.01   separation +0.67   DETECTED
sealed granule tag 9
  matching tag 9 residency 0.72   checked-mismatch mean 0.01   separation +0.68   DETECTED
sealed granule tag 10
  matching tag 10 residency 0.73  checked-mismatch mean 0.02   separation +0.70   DETECTED
```

## Essential parameters

- `--core 8` selects the Cortex-X3; TikTag-v1 is the X3's speculation-shrinkage channel.
- `--iters 512` sets the training length. The leak appears once the branch is trained over roughly 128
  or more iterations and is absent below 64, so the training loop lives inside the test case.
- `--gap 80` sets the serial `orr` chain that paces the probe against the window.
- `--ldnum 16` sets the speculative tag-checked loads per iteration; more checks deepen the window
  shrink on a mismatch, which separates the matching tag from the rest at the X3 timing.
- `--read-set` and `--warm-set` are distinct L1D sets so the read set carries only the attack access.

## Running

The executor module must be loaded on the device, reachable over an adb server. The script pins the
core, sets Flush+Reload with L1-residency reload timing, and leaves the executor's own flush and
branch-history knobs off because the gadget flushes its own lines.

```
python docs/tiktag_mte_ni/tiktag_v1_executor.py \
  --repo ~/revizor/sca-fuzzer --adb-host 127.0.0.1 --adb-port 5037 \
  --core 8 --iters 512 --ldnum 16 --gap 80 --read-set 32 --warm-set 48 \
  --reps 60 --seal-tags 5,9,10
```

It exits zero when every sealed tag is detected. `run_tag.sh` runs each input's repeat loop on the
device so the host issues one adb call per tag.

## Relationship to the Revizor template path

This is a raw-executor harness, not a Revizor `Template`. The object-template API emits forward-only
straight-line code with no loops and no cache-maintenance instructions, and the non-interference
measurement runs every variant under a single tag for the whole program. The gadget needs a backward
training loop, per-iteration line flushes, and a split between a training tag that falls through
architecturally and an attack tag that is read only on the speculated path, none of which the template
API expresses. The contract executor models the data-load tag check (MteSealing) but not the training
schedule and cycle distance that bring the window shrink above the measurement noise on the X3.
