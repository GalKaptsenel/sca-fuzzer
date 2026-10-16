.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
ADD x0, x0, #24 // instrumentation
LDR  w1, [x0, #-24]! 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
LDP  x2, x1, [x0], #-384 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #178 // instrumentation
STR  w1, [x4, #178]! 
CSNEG  w3, w1, w5, lt 
EOR  w0, w2, w5 
B.al .bb_0.1 
.bb_0.1:
CRC32CX  w5, w1, x4 
SUBS  x0, x5, #365, lsl #12 
REV  x4, x2 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
SUB x3, x3, #4095 // instrumentation
SUB x3, x3, #4095 // instrumentation
SUB x3, x3, #4095 // instrumentation
SUB x3, x3, #2523 // instrumentation
STR  w0, [x3, #14808] 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
STR  x5, [x3] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #20 // instrumentation
STP  w0, w2, [x2, #20] 
B .bb_0.2 
.bb_0.2:
CRC32W  w3, w0, w1 
CLZ  w5, w2 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, x1 // instrumentation
STR  x1, [x5, x1] 
CBNZ  x3, .bb_0.3 
B .bb_0.4 
.bb_0.3:
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDP  x2, x5, [x3], #368 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, x5 // instrumentation
LDR  w5, [x1, x5] 
ADDS  x5, x4, #1902, lsl #0 
UDIV  w1, w5, w0 
CRC32CB  w1, w5, w0 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDP  w0, w5, [x2], #-112 
TBZ  w5, #9, .bb_0.4 
B.al .exit_0 
.bb_0.4:
CLZ  w0, w4 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #224 // instrumentation
LDP  w3, w2, [x4, #224] 
RMIF  x0, #61, #8 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
SUB x3, x3, #4095 // instrumentation
SUB x3, x3, #1465 // instrumentation
LDR  w3, [x3, #5560] 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
