.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
REV  w0, w4 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
STP  x4, x3, [x0], #48 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDR  w4, [x2] 
CRC32W  w2, w0, w4 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDP  x4, x5, [x2], #304 
EOR  x3, x2, x5, asr #59 
TBNZ  w3, #9, .bb_0.1 
B .bb_0.2 
.bb_0.1:
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDP  x0, x2, [x1], #208 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
STR  x3, [x2], #-9 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDP  x4, x2, [x3], #72 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
STR  w3, [x5], #-74 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
STR  w2, [x1], #-47 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, x1 // instrumentation
LDR  x0, [x2, x1] 
ADDS  w0, w4, #403, lsl #0 
SUBS  x1, x0, #2510 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDP  x2, x5, [x1], #480 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #233 // instrumentation
STR  w0, [x4, #233]! 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
ADD x5, x5, #64 // instrumentation
LDP  w0, w5, [x5, #-64] 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
SUB x0, x0, #4095 // instrumentation
SUB x0, x0, #4095 // instrumentation
SUB x0, x0, #3766 // instrumentation
STR  w2, [x0, #11956] 
B .bb_0.2 
.bb_0.2:
CRC32CB  w5, w2, w0 
CCMN  x5, x1, #12, eq 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, x1 // instrumentation
LDR  x1, [x5, x1] 
CLZ  x5, x5 
ADDS  w4, w5, w5 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
ADD x3, x3, #112 // instrumentation
LDP  x0, x2, [x3, #-112] 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
