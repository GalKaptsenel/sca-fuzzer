# Which flush knob is the source of the MTE/TikTag window: `view_rotation`

`enable_pre_run_flush` is a legacy combined sysfs knob. Its store handler
(`src/aarch64/executor/sysfs.c` `enable_pre_run_flush_store`) sets BOTH independent kernel knobs:

```c
executor.config.pre_run_flush = value;
executor.config.phr_flush     = value;   // zero the branch-history register once per trace
executor.config.view_rotation = value;   // serve next code view (invalidate_bpu_entries) per run
```

`measurement.c` reads `phr_flush` and `view_rotation` independently (the standalone `pre_run_flush`
field is not read). Controlled isolation on the known-good `s1_tiktag_hard.asm` (set 32, 2000 reps, two
inputs), toggling the two sub-knobs directly via `executor_regime`:

```
(phr_flush, view_rotation) -> match(genuine)/mismatch(noncanon)/gap
 1,1  both (==enable_pre_run_flush:1)  0.70/0.01/+0.69   0.55/0.06/+0.49
 1,0  phr_flush ONLY                   0.01/0.01/+0.00   0.00/0.01/-0.00
 0,1  view_rotation ONLY               0.58/0.00/+0.58   0.57/0.01/+0.55
 0,0  neither (==flush:0)              0.00/0.00/-0.00   0.06/0.05/+0.01
```

**Source = `view_rotation`.** The match/mismatch separation is present whenever `view_rotation` is on
(with or without `phr_flush`) and absent whenever it is off (even with `phr_flush` on). `phr_flush`
alone contributes nothing.

Mechanism: `view_rotation` serves the next code view through `invalidate_bpu_entries()` before each run,
forcing a tagged-BTB miss so the always-architecturally-taken forward branch cold-mispredicts and the
speculative fall-through window (where the sealed guess load sits) opens. Without it, the resolved BTB
entry predicts the branch correctly after the first rep and the window closes — no leak. `phr_flush`
(zeroing the PHR) does not invalidate the resolved BTB entry, so it cannot keep the branch cold.

Correction to earlier notes: `enable_pre_run_flush` matters ONLY because it enables `view_rotation`; it
is not a probe-line data-cache flush (the F+R reload flush is unconditional in `templates_jit.c`), and
`phr_flush` is not the responsible sub-knob.
