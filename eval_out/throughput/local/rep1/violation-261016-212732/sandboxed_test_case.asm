.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDR  x4, [x1], #104 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDR  w2, [x1], #-189 
CBNZ  x4, .bb_0.1 
B .exit_0 
.bb_0.1:
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
SUB x0, x0, x2 // instrumentation
LDR  x3, [x0, x2] 
BICS  w0, w1, w4 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, x1 // instrumentation
STR  w4, [x2, x1] 
TBZ  x3, #2, .bb_0.2 
B.nv .exit_0 
.bb_0.2:
ORR  x1, x1, #18410732868868247551 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDP  w2, w3, [x4], #-48 
ADDS  w2, w5, w1 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
SUB x3, x3, x5 // instrumentation
STR  w2, [x3, x5] 
SUBS  x1, x0, #2835, lsl #12 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #108 // instrumentation
STP  w5, w5, [x4, #108] 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
SUB x3, x3, #135 // instrumentation
STR  w1, [x3, #135]! 
REV16  w5, w2 
BICS  w4, w1, w2, ror #23 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
SUB x3, x3, x0 // instrumentation
LDR  x4, [x3, x0] 
B.al .bb_0.3 
.bb_0.3:
REV32  x0, x2 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #80 // instrumentation
LDP  x4, x0, [x2, #80] 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
SUB x0, x0, #4095 // instrumentation
SUB x0, x0, #4095 // instrumentation
SUB x0, x0, #4095 // instrumentation
SUB x0, x0, #4095 // instrumentation
SUB x0, x0, #4095 // instrumentation
SUB x0, x0, #4095 // instrumentation
SUB x0, x0, #4095 // instrumentation
SUB x0, x0, #4023 // instrumentation
STR  x5, [x0, #32688] 
REV16  x1, x3 
ORR  x5, x5, x1, asr #2 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDP  x3, x1, [x2] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDR  x0, [x2], #-48 
CRC32CH  w5, w4, w3 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
STP  w3, w2, [x5], #-160 
.exit_0:
.macro.measurement_end: NOP
B.nv .test_case_exit 
.section .data.main
.test_case_exit:
