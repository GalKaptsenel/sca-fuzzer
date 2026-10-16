.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
ADDS  x2, x2, #3737 
CRC32H  w4, w0, w4 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
STP  w1, w0, [x3], #-164 
MADD  x1, x2, x1, x1 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
STP  x3, x4, [x0], #-32 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, x4 // instrumentation
STR  w0, [x2, x4] 
RBIT  x2, x1 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDP  x1, x0, [x3], #-424 
EOR  x3, x4, x1 
SUBS  w2, w0, #3711, lsl #0 
CRC32CH  w0, w3, w4 
TBZ  w0, #29, .bb_0.1 
B.nv .exit_0 
.bb_0.1:
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDP  w0, w2, [x3], #132 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDP  x0, x3, [x4], #-416 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDP  x2, x4, [x5] 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
SUB x3, x3, #80 // instrumentation
LDP  w0, w4, [x3, #80] 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
STP  w0, w1, [x5], #-212 
CCMN  x1, #11, #12, al 
CCMP  x3, x1, #9, pl 
CCMP  x2, #24, #1, lt 
CSINV  x3, x0, x2, vs 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
STP  w2, w0, [x5] 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #96 // instrumentation
LDP  w1, w3, [x4, #96] 
EOR  w1, w5, #4236247039 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, x5 // instrumentation
STR  x0, [x1, x5] 
.exit_0:
.macro.measurement_end: NOP
B.nv .test_case_exit 
.section .data.main
.test_case_exit:
