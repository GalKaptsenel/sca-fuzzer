.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
csel  w1, w2, w3, ls
adds  w0, w2, w0
orr  w3, w5, w5, ror #4
adds  x2, x2, x3, lsl #56
orr  x5, x1, #17146314752
str  w1, [x5], #-196
csinv  x4, x1, x0, ne
b .bb_0.1
.bb_0.1:
str  x1, [x3], #60
ldr  w3, [x5, x0]
orr  w1, w5, #50331648
udiv  x5, x5, x2
ands  w4, w3, #4227858439
b.ls  .bb_0.2
b .bb_0.4
.bb_0.2:
orr  w0, w0, w1
csinv  x1, x3, x2, le
adds  x5, x0, #2429, lsl #12
str  w5, [x2], #241
cbz  x4, .bb_0.3
b .exit_0
.bb_0.3:
adds  w4, w1, #3442, lsl #12
ldr  x2, [x5, #-161]!
ldr  x1, [x4], #-198
str  x3, [x1, x4]
ldr  w3, [x1], #-105
adds  w2, w2, #3100, lsl #0
b .bb_0.4
.bb_0.4:
str  x4, [x0, x1]
orr  w0, w1, #471604252
.exit_0:
.macro.measurement_end: nop
b .test_case_exit
.section .data.main
.test_case_exit:
