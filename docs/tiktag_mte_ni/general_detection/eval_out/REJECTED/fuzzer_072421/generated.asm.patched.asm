.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
b.nv .bb_0.1
.bb_0.1:
ccmn  x0, #13, #11, le
b.nv .bb_0.2
.bb_0.2:
bics  w1, w2, w1
str  w5, [x2, x1]
and  w2, w2, #33553408
csel  x5, x1, x2, eq
b .bb_0.3
.bb_0.3:
csinv  x2, x0, x2, pl
cbnz  w3, .bb_0.4
b.nv .bb_0.7
.bb_0.4:
ccmn  x1, #28, #4, eq
eor  w4, w0, w0
stp  w4, w5, [x2], #-76
subg  x1, x5, #16, #8
stzg  x3, [x0, #2960]
b .bb_0.5
.bb_0.5:
b.nv .bb_0.6
.bb_0.6:
str  x5, [x0], #157
str  w3, [x1], #78
stz2g  x2, [x2]
str  w5, [x0, x1]
tbz  x0, #58, .bb_0.7
b.al .exit_0
.bb_0.7:
csinv  w2, w1, w1, ge
ldg  x5, [x1, #-4048]
ldg  x1, [x3]
udiv  w1, w2, w1
eor  x5, x3, x0
b .bb_0.8
.bb_0.8:
ldr  w2, [x4], #222
ccmn  x4, x4, #11, pl
str  x2, [x1]
ldp  w2, w1, [x2]
.exit_0:
.macro.measurement_end: nop
b .test_case_exit
.section .data.main
.test_case_exit:
