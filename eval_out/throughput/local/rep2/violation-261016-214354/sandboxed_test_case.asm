.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
REV16  w0, w1 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDP  w2, w1, [x4], #252 
B.nv .bb_0.1 
.bb_0.1:
ADDS  w5, w1, w5, lsl #5 
CRC32H  w4, w5, w2 
TBNZ  w4, #0, .bb_0.2 
B .bb_0.4 
.bb_0.2:
CLS  w2, w4 
SETF16  w1 
ADDS  x1, x2, #121, lsl #0 
ADDS  x2, x0, x3 
ADDS  x5, x2, #3794 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, x0 // instrumentation
STR  x5, [x1, x0] 
CBZ  x1, .bb_0.3 
B.nv .exit_0 
.bb_0.3:
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
ADD x1, x1, #144 // instrumentation
STP  x5, x4, [x1, #-144] 
SUBS  w4, w3, #842 
CCMP  w1, #14, #10, eq 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, #14 // instrumentation
STR  x4, [x5, #14]! 
SUBS  x5, x3, x4, lsl #4 
B.al .bb_0.4 
.bb_0.4:
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
LDP  w1, w5, [x0], #20 
ADDS  x3, x2, #3291 
RBIT  w5, w4 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
STR  w1, [x3], #-151 
AND  x5, x4, x2 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
LDR  x5, [x3], #212 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
ADD x1, x1, #22 // instrumentation
STR  x4, [x1, #-22]! 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
STP  x0, x5, [x4], #-24 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #4095 // instrumentation
SUB x2, x2, #1244 // instrumentation
LDR  x1, [x2, #17624] 
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit 
.section .data.main
.test_case_exit:
