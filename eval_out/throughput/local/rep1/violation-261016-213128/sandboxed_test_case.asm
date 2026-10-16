.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
ADD x1, x1, #109 // instrumentation
STR  x2, [x1, #-109]! 
SUBS  w5, w1, #750, lsl #0 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #3474 // instrumentation
LDR  x0, [x5, #11664] 
CRC32X  w5, w5, x4 
TBZ  w0, #16, .bb_0.1 
B.nv .exit_0 
.bb_0.1:
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDR  w0, [x5], #-136 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDR  x4, [x1] 
TBNZ  w1, #25, .bb_0.2 
B.nv .bb_0.3 
.bb_0.2:
ADDS  w3, w3, #3632, lsl #12 
ADDS  w2, w4, #1888, lsl #12 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDP  w3, w5, [x2], #-60 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
STP  w0, w4, [x2], #92 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDR  w5, [x3] 
CSINV  x5, x2, x1, ls 
B.al .bb_0.3 
.bb_0.3:
CSINV  x0, x1, x0, cs 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
SUB x0, x0, #30 // instrumentation
LDR  x2, [x0, #30]! 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
ADD x5, x5, #146 // instrumentation
LDR  w2, [x5, #-146]! 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, #192 // instrumentation
STP  x2, x4, [x1, #192] 
CCMN  w0, w2, #0, gt 
B .bb_0.4 
.bb_0.4:
REV16  w4, w1 
BICS  w5, w0, w3 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #352 // instrumentation
LDR  x2, [x4, #352] 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #3115 // instrumentation
LDR  x0, [x5, #15400] 
ANDS  w4, w5, w2, asr #6 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDP  w3, w4, [x2], #-240 
BICS  w4, w2, w3 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
