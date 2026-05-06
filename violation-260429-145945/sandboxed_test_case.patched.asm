.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
udiv  x0, x3, x0
and x1, x1, 0b1111111111111 // instrumentation
add x1, x1, x29 // instrumentation
sub x1, x1, 29 // instrumentation
str  x4, [x1, #29]!
ands  x2, x5, x0, lsr #53
cbnz  x4, .bb_0.1
b .bb_0.4
.bb_0.1:
csel  w3, w5, w0, vs
eor  w4, w5, w0, asr #5
csneg  x3, x4, x1, gt
subs  x0, x0, x2, asr #53
sdiv  x0, x0, x5
csinc  w0, w4, w1, lt
b .bb_0.2
.bb_0.2:
orr  w5, w1, #3237986559
csinv  w2, w2, w0, cc
sdiv  x5, x3, x4
ands  x5, x3, #18446743936271122431
cbz  w5, .bb_0.3
b .bb_0.4
.bb_0.3:
csel  w3, w2, w0, cc
and x3, x3, 0b1111111111111 // instrumentation
add x3, x3, x29 // instrumentation
sub x3, x3, x4 // instrumentation
str  x1, [x3, x4]
sdiv  x4, x5, x2
eor  x2, x4, #18446743523953738751
and x0, x0, 0b1111111111111 // instrumentation
add x0, x0, x29 // instrumentation
str  x5, [x0]
and x2, x2, 0b1111111111111 // instrumentation
add x2, x2, x29 // instrumentation
sub x2, x2, -247 // instrumentation
ldr  w1, [x2, #-247]!
and x2, x2, 0b1111111111111 // instrumentation
add x2, x2, x29 // instrumentation
sub x2, x2, 4095 // instrumentation
sub x2, x2, 4095 // instrumentation
sub x2, x2, 4095 // instrumentation
sub x2, x2, 4095 // instrumentation
sub x2, x2, 4095 // instrumentation
sub x2, x2, 1917 // instrumentation
ldr  x4, [x2, #22392]
adds  x2, x5, x1
and x4, x4, 0b1111111111111 // instrumentation
add x4, x4, x29 // instrumentation
sub x4, x4, x1 // instrumentation
ldr  x5, [x4, x1]
b.le  .bb_0.4
b .exit_0
.bb_0.4:
and x1, x1, 0b1111111111111 // instrumentation
add x1, x1, x29 // instrumentation
ldr  x0, [x1], #100
ccmn  w5, w3, #0, mi
.exit_0:
.macro.measurement_end: nop
b .test_case_exit
.section .data.main
.test_case_exit:
