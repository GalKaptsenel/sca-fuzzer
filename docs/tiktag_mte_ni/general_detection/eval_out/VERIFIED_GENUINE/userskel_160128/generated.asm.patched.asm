.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
ands  x5, x0, x0, lsr #59
ands  w5, w5, w4, asr #6
sdiv  w0, w4, w4
adds  w1, w2, w2
ldr  x0, [x0]
str  w1, [x4, #10112]
csel  x1, x1, x1, mi
ldp  w5, w2, [x3]
sdiv  x9, x1, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
eor  x3, x0, #8589934080
subp  x3, x5, x0
ccmp  w4, w5, #6, lt
subps  x1, x5, x5
adds  w3, w2, w3
ccmn  w0, #16, #10, cs
stp  w5, w0, [x0, #196]
csinv  x0, x3, x0, eq
orr  x9, x9, #1
cbnz  x9, .bb_0.2
.bb_0.1:
str  x3, [x5, #-47]!
subs  w1, w1, #3154
subs  x0, x4, x4
ccmp  w1, #5, #6, nv
stp  w1, w1, [x2, #-164]
ccmn  w0, w2, #13, ge
ldr  x2, [x0, #179]!
subs  w0, w5, w0, asr #3
ldr  x12, [x5]
stg  x2, [x2, #-1632]
stp  w5, w0, [x1], #-224
adds  x2, x2, x1
ldr  x3, [x3, #2912]
stz2g  x0, [x4], #-192
sdiv  w2, w4, w3
csel  x5, x0, x2, mi
ccmn  x1, #29, #0, gt
ldr  w4, [x5]
orr  w5, w0, #66847740
udiv  w5, w1, w4
ldr  w2, [x3]
sdiv  w1, w2, w1
subg  x2, x0, #48, #1
and  x0, x3, x1, lsr #43
adds  w3, w1, #2064
.bb_0.2:
stp  w1, w1, [x4]
adds  w4, w5, #228
ccmn  w0, w3, #9, ge
ands  w1, w4, w2
str  x1, [x0, #3368]
adds  w5, w2, #1512
adds  x3, x0, #3573
ldr  x1, [x2], #118
.macro.measurement_end: nop
b  .test_case_exit
.exit_0:
.section .data.main
.test_case_exit:
