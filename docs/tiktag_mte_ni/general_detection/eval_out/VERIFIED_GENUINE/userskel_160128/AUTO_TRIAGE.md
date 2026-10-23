# auto_triage: violation-261022-160128

config: /home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/try_user_mte_skeleton/violation-261022-160128/org-config.yaml
suspect input: #5
reps: 2000

## Step 1+2 (accounting)
```
counterexamples: [5, 15, 25, 35]  (same input iff they pair as N and N+K)
  diff @196: ADDG x5 tag=7 -> ADDG x5 tag=8
  diff @228: ADDG x2 tag=7 -> ADDG x2 tag=11
  diff @236: ADDG x2 tag=0 -> ADDG x2 tag=12
  diff @256: ADDG x0 tag=7 -> ADDG x0 tag=14
  diff @276: ADDG x5 tag=7 -> ADDG x5 tag=6
  diff @284: ADDG x5 tag=0 -> ADDG x5 tag=1
  diff @336: ADDG x3 tag=7 -> ADDG x3 tag=13
  diff @404: ADDG x3 tag=7 -> ADDG x3 tag=8
  diff @412: ADDG x3 tag=0 -> ADDG x3 tag=15
  diff @228: ADDG x2 tag=7 -> ADDG x2 tag=12
  diff @236: ADDG x2 tag=0 -> ADDG x2 tag=11
  diff @256: ADDG x0 tag=7 -> ADDG x0 tag=6
  diff @276: ADDG x5 tag=7 -> ADDG x5 tag=5
  diff @284: ADDG x5 tag=0 -> ADDG x5 tag=2
  diff @312: ADDG x1 tag=7 -> ADDG x1 tag=4
  diff @336: ADDG x3 tag=7 -> ADDG x3 tag=4
  diff @276: ADDG x5 tag=7 -> ADDG x5 tag=13
  diff @284: ADDG x5 tag=0 -> ADDG x5 tag=10
  diff @312: ADDG x1 tag=7 -> ADDG x1 tag=4
  diff @404: ADDG x3 tag=7 -> ADDG x3 tag=2
  diff @412: ADDG x3 tag=0 -> ADDG x3 tag=5
  diff @196: ADDG x5 tag=8 -> ADDG x5 tag=7
  diff @228: ADDG x2 tag=11 -> ADDG x2 tag=12
  diff @236: ADDG x2 tag=12 -> ADDG x2 tag=11
  diff @256: ADDG x0 tag=14 -> ADDG x0 tag=6
  diff @276: ADDG x5 tag=6 -> ADDG x5 tag=5
  diff @284: ADDG x5 tag=1 -> ADDG x5 tag=2
  diff @312: ADDG x1 tag=7 -> ADDG x1 tag=4
  diff @336: ADDG x3 tag=13 -> ADDG x3 tag=4
  diff @404: ADDG x3 tag=8 -> ADDG x3 tag=7
  diff @412: ADDG x3 tag=15 -> ADDG x3 tag=0
  diff @196: ADDG x5 tag=8 -> ADDG x5 tag=7
  diff @228: ADDG x2 tag=11 -> ADDG x2 tag=7
  diff @236: ADDG x2 tag=12 -> ADDG x2 tag=0
  diff @256: ADDG x0 tag=14 -> ADDG x0 tag=7
  diff @276: ADDG x5 tag=6 -> ADDG x5 tag=13
  diff @284: ADDG x5 tag=1 -> ADDG x5 tag=10
  diff @312: ADDG x1 tag=7 -> ADDG x1 tag=4
  diff @336: ADDG x3 tag=13 -> ADDG x3 tag=7
  diff @404: ADDG x3 tag=8 -> ADDG x3 tag=2
  diff @412: ADDG x3 tag=15 -> ADDG x3 tag=5
  diff @228: ADDG x2 tag=12 -> ADDG x2 tag=7
  diff @236: ADDG x2 tag=11 -> ADDG x2 tag=0
  diff @256: ADDG x0 tag=6 -> ADDG x0 tag=7
  diff @276: ADDG x5 tag=5 -> ADDG x5 tag=13
  diff @284: ADDG x5 tag=2 -> ADDG x5 tag=10
  diff @336: ADDG x3 tag=4 -> ADDG x3 tag=7
  diff @404: ADDG x3 tag=7 -> ADDG x3 tag=2
  diff @412: ADDG x3 tag=0 -> ADDG x3 tag=5
TAG-ONLY: True
HW diverging sets (|Δ|>=0.08) between #5 and #15: [39]

WARNING: [fuzzer] Running in non-interference mode.

CE arch demand sets: [0, 3, 8, 37, 42, 47, 63]
CE spec-only sets   : [14, 32, 38, 39, 61]  (leak-relevant)

diverging set -> trainer (leak is prefetch of a SPEC tag-sealed access):
  set 39 -> IS spec demand   (base x5)

eligible tag-sealed regs: ['x0', 'x1', 'x2', 'x3', 'x5']
=> feed the sweep:  REG in ['x5']   LEAK_SETS = [39]
```

## Step 4 (device tag-sweep on spec-relevant diverging sets)
### diverging set 39 (base x5)
```
WARNING: [fuzzer] Running in non-interference mode.
violation-261022-160128 suspect#5 divset=39 reps=2000  genuine baseline res=0.54
  24 ADDG retag relocs
  top knockouts (drop, off, genuine_delta, res):
    off=0x1c dgen=7 res=0.00 drop=+0.54
    off=0x38 dgen=7 res=0.12 drop=+0.42
    off=0xa0 dgen=7 res=0.23 drop=+0.31
    off=0x1dc dgen=7 res=0.28 drop=+0.26
  SWEEP leaking off=0x1c (genuine delta=7): divset 39 residency by delta 0..15:
   0:0.56 1:0.00 2:0.00 3:0.01 4:0.00 5:0.01 6:0.04 7:0.58 8:0.00 9:0.00 10:0.01 11:0.06 12:0.10 13:0.01 14:0.00 15:0.00
```

## VERDICT: GENUINE
set 39 (base x5): match delta 7=0.58 dominates checked mismatches (max 0.10)

(spec-only CE sets=[14, 32, 38, 39, 61]; spec-relevant diverging sets=[39]; eligible MTE regs=['x0', 'x1', 'x2', 'x3', 'x5'])
