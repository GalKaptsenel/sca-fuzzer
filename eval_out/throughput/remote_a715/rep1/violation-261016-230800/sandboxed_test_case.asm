.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
ADD x3, x3, #25 // instrumentation
STR  w1, [x3, #-25]! 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #3202 // instrumentation
STR  x0, [x5, #11392] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDP  w1, w0, [x2] 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #725 // instrumentation
LDR  x4, [x1, #21200] 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
STP  x2, x3, [x4] 
SDIV  w3, w4, w3 
B.al .bb_0.1 
.bb_0.1:
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
ADD x1, x1, #208 // instrumentation
LDP  x5, x2, [x1, #-208] 
MSUB  x1, x1, x4, x1 
EOR  x5, x2, #72057576858058752 
ADDS  w1, w3, #1039, lsl #0 
ORR  w0, w1, w5, asr #26 
ANDS  x3, x4, #18230567993010552063 
REV16  w1, w0 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDP  x5, x4, [x2], #-72 
TBZ  w5, #13, .bb_0.2 
B.nv .exit_0 
.bb_0.2:
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDP  w5, w0, [x1] 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDP  w3, w1, [x5], #196 
REV16  x0, x5 
SUBS  w0, w3, #924, lsl #0 
B.al .bb_0.3 
.bb_0.3:
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #217 // instrumentation
LDR  w4, [x2, #217]! 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
ADD x4, x4, #132 // instrumentation
LDP  w4, w0, [x4, #-132] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, x5 // instrumentation
STR  w4, [x2, x5] 
CCMN  w3, w3, #7, ne 
AND  w5, w3, w1 
CCMN  x1, x2, #1, cs 
.exit_0:
.macro.measurement_end: NOP
B.nv .test_case_exit 
.section .data.main
.test_case_exit:
