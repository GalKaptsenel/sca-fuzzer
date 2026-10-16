.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
CRC32B  w0, w3, w2 
B .bb_0.1 
.bb_0.1:
CCMN  w2, #23, #13, ls 
SUBS  w2, w2, w4 
ORR  x3, x3, x2, ror #12 
ORR  w0, w5, w4, lsl #25 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
SUB x0, x0, x4 // instrumentation
LDR  x2, [x0, x4] 
TBZ  w2, #7, .bb_0.2 
B.al .exit_0 
.bb_0.2:
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
LDR  x4, [x0], #75 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #1942 // instrumentation
STR  w2, [x5, #10132] 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
STR  w1, [x5] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #144 // instrumentation
STP  w2, w2, [x2, #144] 
SUBS  x4, x1, x1 
B.al .bb_0.3 
.bb_0.3:
CRC32CW  w3, w0, w5 
AND  x5, x3, #65534 
SUBS  x0, x2, x0, asr #27 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDR  x2, [x3] 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
ADD x1, x1, #114 // instrumentation
STR  w5, [x1, #-114]! 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
ADD x1, x1, #19 // instrumentation
LDR  w0, [x1, #-19]! 
CCMP  w2, w5, #2, cs 
TBNZ  w5, #22, .bb_0.4 
B.al .exit_0 
.bb_0.4:
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDR  w5, [x4], #-179 
ANDS  w0, w1, w0 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, #140 // instrumentation
STP  w1, w0, [x1, #140] 
CRC32B  w4, w5, w0 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, x1 // instrumentation
LDR  w4, [x4, x1] 
CLZ  w4, w5 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
