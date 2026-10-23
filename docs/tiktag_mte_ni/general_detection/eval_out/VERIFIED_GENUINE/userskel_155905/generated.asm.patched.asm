.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
subs  x3, x3, x3
addg  x4, x0, #416, #11
and  w2, w0, #2147483903
and  w1, w5, w5, asr #30
bics  x4, x3, x0, lsl #52
csinc  x4, x3, x2, hi
st2g  x5, [x2]
stp  x5, x2, [x2]
sdiv  x9, x1, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  x9, x9, x2
sdiv  w4, w5, w2
bics  x1, x2, x0
stp  w1, w1, [x2], #-128
madd  w4, w4, w5, w5
ccmn  w1, #15, #6, nv
and  w3, w1, #4294967171
stp  x1, x1, [x5], #-408
msub  w1, w3, w2, w5
orr  x9, x9, #1
cbnz  x9, .bb_0.2
.bb_0.1:
stp  w2, w1, [x3, #-164]
orr  x4, x3, #68718952448
str  w3, [x1, #-242]!
bics  x5, x2, x0
bics  w4, w4, w0
and  w0, w2, w1, ror #0
str  w3, [x5, #-39]!
str  w0, [x1, #14560]
ldr  x12, [x5]
ldp  w0, w4, [x2], #-92
udiv  x0, x1, x0
csel  w1, w1, w1, pl
csneg  x3, x3, x3, pl
ldg  x4, [x1, #2976]
stp  w1, w3, [x1]
stp  x1, x4, [x0]
ldr  x2, [x4], #114
adds  x0, x5, x3, lsr #38
ands  x5, x3, #13835075646394470399
ldp  x3, x1, [x4]
adds  w2, w4, w5
adds  w4, w2, w3, lsl #19
ccmp  x5, x4, #2, ne
str  w2, [x0, x2]
ldr  x3, [x2], #247
.bb_0.2:
gmi  x1, x2, x5
msub  x0, x4, x1, x3
ldp  x2, x3, [x4], #-16
udiv  w2, w2, w4
and  x3, x5, x4, ror #14
subs  w5, w3, w5
adds  x0, x4, x3
ldp  x1, x2, [x0], #176
.macro.measurement_end: nop
b  .test_case_exit
.exit_0:
.section .data.main
.test_case_exit:
