.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
ADD x3, x3, #133 // instrumentation
STR  x5, [x3, #-133]! 
SETF8  w1 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
ADD x3, x3, #97 // instrumentation
LDR  x1, [x3, #-97]! 
SUBS  x4, x1, x4 
REV  x1, x5 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, #4095 // instrumentation
SUB x5, x5, #3469 // instrumentation
STR  w4, [x5, #7564] 
ORR  w4, w5, #67108352 
CSEL  x4, x5, x1, ne 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
LDP  x3, x5, [x0] 
B.hi  .bb_0.1 
B.al .exit_0 
.bb_0.1:
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #128 // instrumentation
LDP  w3, w5, [x2, #128] 
AND x3, x3, #0x1fff // instrumentation
ADD x3, x3, x29 // instrumentation
STP  x5, x5, [x3] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #98 // instrumentation
LDR  x4, [x2, #98]! 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
LDR  x2, [x0], #53 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
SUB x5, x5, #5 // instrumentation
LDR  x0, [x5, #5]! 
ADDS  w1, w2, w1 
CSNEG  x0, x5, x4, hi 
TBZ  w5, #0, .bb_0.2 
B.al .exit_0 
.bb_0.2:
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
ADD x2, x2, #128 // instrumentation
LDP  w5, w1, [x2, #-128] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, x0 // instrumentation
LDR  w1, [x2, x0] 
AND  x0, x0, #72057576874835964 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
SUB x2, x2, #99 // instrumentation
LDR  w1, [x2, #99]! 
CRC32CH  w0, w5, w0 
CLS  w3, w4 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, #92 // instrumentation
STR  w4, [x1, #92]! 
AND  x5, x4, x4 
.exit_0:
.macro.measurement_end: NOP
B.nv .test_case_exit 
.section .data.main
.test_case_exit:
