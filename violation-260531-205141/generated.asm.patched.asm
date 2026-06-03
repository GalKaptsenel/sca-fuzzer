.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
ldr  x2, [x3], #104
ldr  x3, [x4, #9592]
and  x2, x0, x0, ror #22
cbz  x3, .bb_0.1
b .bb_0.2
.bb_0.1:
subs  w5, w2, #1663, lsl #0
subs  w0, w1, w3
csneg  w0, w3, w0, vc
str  w2, [x0, x4]
b .bb_0.2
.bb_0.2:
str  x1, [x5]
ldr  w2, [x5], #88
csneg  x5, x3, x5, vs
ldr  w5, [x0]
b .bb_0.3
.bb_0.3:
subs  w3, w4, #794, lsl #12
ldr  w4, [x2], #104
str  x2, [x5, #3960]
eor  w4, w4, w3, lsl #14
orr  w3, w5, #4294959135
ccmp  x3, x3, #1, mi
b .bb_0.4
.bb_0.4:
orr  x1, x3, x2, lsr #53
adds  w1, w1, #2460, lsl #12
sdiv  w5, w4, w3
str  w2, [x1, #-35]!
ldr  x4, [x5], #140
adds  w4, w5, w2, asr #6
orr  x1, x4, x3, lsl #51
.exit_0:
.macro.measurement_end: nop
b .test_case_exit
.section .data.main
.test_case_exit:
