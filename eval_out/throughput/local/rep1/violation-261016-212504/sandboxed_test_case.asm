.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDP  x2, x5, [x5] 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #224 // instrumentation
STP  w2, w3, [x4, #224] 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, x0 // instrumentation
STR  x5, [x4, x0] 
ADDS  x0, x4, x5 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
STP  w3, w3, [x0], #240 
CRC32W  w2, w2, w0 
CCMN  x1, #27, #10, ls 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDR  x0, [x5], #99 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #3451 // instrumentation
LDR  x5, [x2, #15736] 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, #568 // instrumentation
LDR  x5, [x5, #568] 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #4095 // instrumentation
SUB x1, x1, #7 // instrumentation
LDR  x5, [x1, #28672] 
CBNZ  x5, .bb_0.1 
B.nv .bb_0.2 
.bb_0.1:
ORR  x5, x5, x5, asr #27 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDR  x5, [x4], #180 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDP  x5, x1, [x5] 
SUBS  x2, x5, x2 
TBZ  x5, #46, .bb_0.2 
B .exit_0 
.bb_0.2:
CCMP  x3, #3, #15, nv 
CCMN  x0, #0, #1, ge 
EOR  x0, x2, x5 
ADDS  x4, x4, #723 
CSNEG  x3, x5, x1, mi 
CRC32CW  w4, w1, w0 
CRC32CH  w3, w5, w5 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDP  w3, w5, [x4] 
SUBS  w4, w1, #2713, lsl #0 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
