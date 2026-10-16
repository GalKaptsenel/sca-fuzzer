.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
REV16  x5, x3 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, x0 // instrumentation
LDR  w1, [x1, x0] 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDP  x0, x4, [x5], #-304 
B.vc  .bb_0.1 
B.nv .bb_0.3 
.bb_0.1:
BICS  x0, x2, x1, asr #8 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDP  x4, x2, [x1] 
CSEL  w3, w0, w0, mi 
BICS  x5, x4, x0, lsl #57 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDR  x3, [x2], #-66 
B .bb_0.2 
.bb_0.2:
CSEL  w5, w0, w3, nv 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
STR  w3, [x0], #-125 
ADDS  x0, x5, #3430, lsl #0 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDR  x5, [x4] 
CCMP  x5, #21, #0, ls 
TBNZ  x0, #2, .bb_0.3 
B.nv .exit_0 
.bb_0.3:
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDP  x3, x0, [x1], #64 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
STP  w0, w4, [x0] 
EOR  x5, x2, #18446744073701294079 
AND  x5, x4, #18446744073701163007 
CBNZ  x5, .bb_0.4 
B .exit_0 
.bb_0.4:
SUBS  w5, w2, w0 
REV  w3, w1 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDP  w2, w3, [x5], #-64 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
LDR  w2, [x0] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
ADD x2, x2, #221 // instrumentation
STR  x4, [x2, #-221]! 
SUBS  w3, w0, #3912, lsl #0 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
STP  x1, x5, [x1] 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
