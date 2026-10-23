# auto_triage: violation-261022-155905

config: /home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/try_user_mte_skeleton/violation-261022-155905/org-config.yaml
suspect input: #5
reps: 2000

## Step 1+2 (accounting)
```
counterexamples: [5, 15, 25, 35]  (same input iff they pair as N and N+K)
  diff @188: ADDG x1 tag=7 -> ADDG x1 tag=13
  diff @220: ADDG x5 tag=7 -> ADDG x5 tag=4
  diff @252: ADDG x1 tag=7 -> ADDG x1 tag=3
  diff @260: ADDG x1 tag=0 -> ADDG x1 tag=4
  diff @272: ADDG x5 tag=7 -> ADDG x5 tag=12
  diff @280: ADDG x5 tag=0 -> ADDG x5 tag=11
  diff @292: ADDG x2 tag=7 -> ADDG x2 tag=3
  diff @336: ADDG x1 tag=7 -> ADDG x1 tag=12
  diff @344: ADDG x1 tag=0 -> ADDG x1 tag=11
  diff @356: ADDG x0 tag=7 -> ADDG x0 tag=5
  diff @364: ADDG x0 tag=0 -> ADDG x0 tag=2
  diff @376: ADDG x4 tag=7 -> ADDG x4 tag=5
  diff @400: ADDG x4 tag=7 -> ADDG x4 tag=8
  diff @408: ADDG x4 tag=0 -> ADDG x4 tag=15
  diff @220: ADDG x5 tag=7 -> ADDG x5 tag=9
  diff @292: ADDG x2 tag=7 -> ADDG x2 tag=4
  diff @336: ADDG x1 tag=7 -> ADDG x1 tag=10
  diff @344: ADDG x1 tag=0 -> ADDG x1 tag=13
  diff @376: ADDG x4 tag=7 -> ADDG x4 tag=13
  diff @400: ADDG x4 tag=7 -> ADDG x4 tag=15
  diff @408: ADDG x4 tag=0 -> ADDG x4 tag=8
  diff @456: ADDG x2 tag=7 -> ADDG x2 tag=15
  diff @252: ADDG x1 tag=7 -> ADDG x1 tag=12
  diff @260: ADDG x1 tag=0 -> ADDG x1 tag=11
  diff @292: ADDG x2 tag=7 -> ADDG x2 tag=15
  diff @336: ADDG x1 tag=7 -> ADDG x1 tag=9
  diff @344: ADDG x1 tag=0 -> ADDG x1 tag=14
  diff @356: ADDG x0 tag=7 -> ADDG x0 tag=14
  diff @364: ADDG x0 tag=0 -> ADDG x0 tag=9
  diff @400: ADDG x4 tag=7 -> ADDG x4 tag=6
  diff @408: ADDG x4 tag=0 -> ADDG x4 tag=1
  diff @436: ADDG x0 tag=7 -> ADDG x0 tag=11
  diff @444: ADDG x0 tag=0 -> ADDG x0 tag=12
  diff @188: ADDG x1 tag=13 -> ADDG x1 tag=7
  diff @220: ADDG x5 tag=4 -> ADDG x5 tag=9
  diff @252: ADDG x1 tag=3 -> ADDG x1 tag=7
  diff @260: ADDG x1 tag=4 -> ADDG x1 tag=0
  diff @272: ADDG x5 tag=12 -> ADDG x5 tag=7
  diff @280: ADDG x5 tag=11 -> ADDG x5 tag=0
  diff @292: ADDG x2 tag=3 -> ADDG x2 tag=4
  diff @336: ADDG x1 tag=12 -> ADDG x1 tag=10
  diff @344: ADDG x1 tag=11 -> ADDG x1 tag=13
  diff @356: ADDG x0 tag=5 -> ADDG x0 tag=7
  diff @364: ADDG x0 tag=2 -> ADDG x0 tag=0
  diff @376: ADDG x4 tag=5 -> ADDG x4 tag=13
  diff @400: ADDG x4 tag=8 -> ADDG x4 tag=15
  diff @408: ADDG x4 tag=15 -> ADDG x4 tag=8
  diff @456: ADDG x2 tag=7 -> ADDG x2 tag=15
  diff @188: ADDG x1 tag=13 -> ADDG x1 tag=7
  diff @220: ADDG x5 tag=4 -> ADDG x5 tag=7
  diff @252: ADDG x1 tag=3 -> ADDG x1 tag=12
  diff @260: ADDG x1 tag=4 -> ADDG x1 tag=11
  diff @272: ADDG x5 tag=12 -> ADDG x5 tag=7
  diff @280: ADDG x5 tag=11 -> ADDG x5 tag=0
  diff @292: ADDG x2 tag=3 -> ADDG x2 tag=15
  diff @336: ADDG x1 tag=12 -> ADDG x1 tag=9
  diff @344: ADDG x1 tag=11 -> ADDG x1 tag=14
  diff @356: ADDG x0 tag=5 -> ADDG x0 tag=14
  diff @364: ADDG x0 tag=2 -> ADDG x0 tag=9
  diff @376: ADDG x4 tag=5 -> ADDG x4 tag=7
  diff @400: ADDG x4 tag=8 -> ADDG x4 tag=6
  diff @408: ADDG x4 tag=15 -> ADDG x4 tag=1
  diff @436: ADDG x0 tag=7 -> ADDG x0 tag=11
  diff @444: ADDG x0 tag=0 -> ADDG x0 tag=12
  diff @220: ADDG x5 tag=9 -> ADDG x5 tag=7
  diff @252: ADDG x1 tag=7 -> ADDG x1 tag=12
  diff @260: ADDG x1 tag=0 -> ADDG x1 tag=11
  diff @292: ADDG x2 tag=4 -> ADDG x2 tag=15
  diff @336: ADDG x1 tag=10 -> ADDG x1 tag=9
  diff @344: ADDG x1 tag=13 -> ADDG x1 tag=14
  diff @356: ADDG x0 tag=7 -> ADDG x0 tag=14
  diff @364: ADDG x0 tag=0 -> ADDG x0 tag=9
  diff @376: ADDG x4 tag=13 -> ADDG x4 tag=7
  diff @400: ADDG x4 tag=15 -> ADDG x4 tag=6
  diff @408: ADDG x4 tag=8 -> ADDG x4 tag=1
  diff @436: ADDG x0 tag=7 -> ADDG x0 tag=11
  diff @444: ADDG x0 tag=0 -> ADDG x0 tag=12
  diff @456: ADDG x2 tag=15 -> ADDG x2 tag=7
TAG-ONLY: True
HW diverging sets (|Δ|>=0.08) between #5 and #15: [9]

WARNING: [fuzzer] Running in non-interference mode.

CE arch demand sets: [2, 33, 42]
CE spec-only sets   : [0, 5, 7, 9, 26, 27, 28, 38, 43, 44, 46, 47, 56]  (leak-relevant)

diverging set -> trainer (leak is prefetch of a SPEC tag-sealed access):
  set  9 -> IS spec demand   (base x1)

eligible tag-sealed regs: ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']
=> feed the sweep:  REG in ['x1']   LEAK_SETS = [9]
```

## Step 4 (device tag-sweep on spec-relevant diverging sets)
### diverging set 9 (base x1)
```
WARNING: [fuzzer] Running in non-interference mode.
violation-261022-155905 suspect#5 divset=9 reps=2000  genuine baseline res=1.00
  25 ADDG retag relocs
  top knockouts (drop, off, genuine_delta, res):
    off=0xa0 dgen=7 res=0.02 drop=+0.98
    off=0xdc dgen=7 res=0.02 drop=+0.98
    off=0x80 dgen=7 res=0.75 drop=+0.25
    off=0x1e0 dgen=7 res=0.80 drop=+0.20
  SWEEP leaking off=0xa0 (genuine delta=7): divset 9 residency by delta 0..15:
   0:0.02 1:0.00 2:0.00 3:0.01 4:0.00 5:0.02 6:0.01 7:1.00 8:0.04 9:0.00 10:0.02 11:0.00 12:0.00 13:0.00 14:0.00 15:0.00
```

## VERDICT: GENUINE
set 9 (base x1): match delta 7=1.00 dominates checked mismatches (max 0.04)

(spec-only CE sets=[0, 5, 7, 9, 26, 27, 28, 38, 43, 44, 46, 47, 56]; spec-relevant diverging sets=[9]; eligible MTE regs=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
