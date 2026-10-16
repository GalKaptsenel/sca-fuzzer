.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, x2 // instrumentation
STR  w0, [x4, x2] 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, x0 // instrumentation
LDR  x3, [x4, x0] 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, #392 // instrumentation
LDP  x4, x5, [x5, #392] 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDR  x1, [x4], #125 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDP  x1, x3, [x4], #-248 
CCMN  x1, #27, #13, vc 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDR  w4, [x2] 
B .bb_0.1 
.bb_0.1:
CCMP  w2, w2, #8, gt 
EOR  w1, w3, w2 
SUBS  w4, w0, w3 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #4095 // instrumentation
SUB x4, x4, #73 // instrumentation
STR  w3, [x4, #4168] 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #750 // instrumentation
LDR  x0, [x1, #25320] 
CSINV  w1, w4, w1, lt 
BICS  w4, w5, w4, ror #29 
CBNZ  x3, .bb_0.2 
B.nv .exit_0 
.bb_0.2:
BICS  w3, w0, w0, lsl #31 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #163 // instrumentation
LDR  w4, [x2, #163]! 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
ADD x0, x0, #179 // instrumentation
STR  w3, [x0, #-179]! 
TBNZ  w4, #20, .bb_0.3 
B .bb_0.4 
.bb_0.3:
CRC32W  w5, w1, w0 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
SUB x3, x3, #304 // instrumentation
STP  x1, x2, [x3, #304] 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDR  x5, [x3] 
TBNZ  x2, #38, .bb_0.4 
B.al .exit_0 
.bb_0.4:
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
ADD x5, x5, #296 // instrumentation
STP  x2, x1, [x5, #-296] 
EOR  x2, x1, #4503565267632128 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDR  x4, [x1] 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, x0 // instrumentation
LDR  x1, [x4, x0] 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
