.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
LDP  x0, x3, [x4], #-200 
SDIV  x2, x2, x0 
TBNZ  w0, #2, .bb_0.1 
B.nv .bb_0.3 
.bb_0.1:
BICS  x1, x1, x2 
CRC32W  w5, w2, w4 
EOR  x2, x2, x1, lsr #42 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDR  x4, [x1] 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, #208 // instrumentation
STP  w4, w0, [x1, #208] 
B.hi  .bb_0.2 
B.nv .exit_0 
.bb_0.2:
MSUB  w1, w3, w0, w5 
AND  w3, w5, #2096640 
ORR  w4, w4, w3, lsl #23 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDR  x4, [x2], #-163 
ANDS  w0, w0, w4 
TBNZ  x4, #21, .bb_0.3 
B.al .exit_0 
.bb_0.3:
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDP  x4, x2, [x5], #-176 
AND x4, x4, #0x1fff // instrumentation
ADD x4, x4, x29 // instrumentation
SUB x4, x4, #4095 // instrumentation
SUB x4, x4, #4095 // instrumentation
SUB x4, x4, #4095 // instrumentation
SUB x4, x4, #4095 // instrumentation
SUB x4, x4, #4095 // instrumentation
SUB x4, x4, #869 // instrumentation
STR  x5, [x4, #21344] 
AND x2, x2, #0x1fff // instrumentation
ADD x2, x2, x29 // instrumentation
LDP  x5, x4, [x2], #24 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
LDR  w4, [x1] 
REV  x1, x0 
CRC32B  w0, w1, w0 
B.al .bb_0.4 
.bb_0.4:
ADDS  w5, w0, w5 
AND x1, x1, #0x1fff // instrumentation
ADD x1, x1, x29 // instrumentation
SUB x1, x1, #128 // instrumentation
STP  w5, w5, [x1, #128] 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
SUB x0, x0, #280 // instrumentation
STP  x2, x1, [x0, #280] 
UDIV  x0, x2, x2 
AND x0, x0, #0x1fff // instrumentation
ADD x0, x0, x29 // instrumentation
SUB x0, x0, #159 // instrumentation
LDR  x4, [x0, #159]! 
AND x5, x5, #0x1fff // instrumentation
ADD x5, x5, x29 // instrumentation
LDR  x5, [x5] 
.exit_0:
.macro.measurement_end: NOP
B.nv .test_case_exit 
.section .data.main
.test_case_exit:
