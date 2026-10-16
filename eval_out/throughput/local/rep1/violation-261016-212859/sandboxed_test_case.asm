.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
SUBS  w3, w3, w1, lsl #21 
CBZ  w5, .bb_0.1 
B .bb_0.2 
.bb_0.1:
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, x3 // instrumentation
LDR  x2, [x1, x3] 
CLZ  x3, x1 
MADD  w3, w0, w1, w0 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
ADD x3, x3, #10 // instrumentation
LDR  x1, [x3, #-10]! 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDR  x1, [x2] 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDP  w5, w4, [x3], #120 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDR  x1, [x3] 
BICS  w5, w1, w0, asr #19 
CBNZ  w5, .bb_0.2 
B.nv .exit_0 
.bb_0.2:
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #4095 // instrumentation
SUB x4, x4, #89 // instrumentation
LDR  x3, [x4, #4184] 
CLZ  w4, w1 
EOR  w0, w4, w1, ror #0 
EOR  w1, w4, w1, ror #18 
CCMN  w3, w1, #15, hi 
CBNZ  w3, .bb_0.3 
B .exit_0 
.bb_0.3:
SUBS  x5, x3, #1443 
BICS  w0, w5, w5, lsr #23 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDP  x2, x5, [x2] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #1700 // instrumentation
LDR  x4, [x2, #18080] 
AND  x1, x0, x1, asr #40 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDR  w2, [x1] 
EOR  x0, x1, x4, lsr #27 
TBNZ  x5, #26, .bb_0.4 
B .exit_0 
.bb_0.4:
SETF8  w3 
ANDS  w1, w3, w2, ror #7 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
LDR  x2, [x0], #51 
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit 
.section .data.main
.test_case_exit:
