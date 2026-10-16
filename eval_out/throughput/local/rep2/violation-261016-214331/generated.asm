.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
LDP  x0, x3, [x4], #-200 
SDIV  x2, x2, x0 
TBNZ  w0, #2, .bb_0.1 
B.nv .bb_0.3 
.bb_0.1:
BICS  x1, x1, x2 
CRC32W  w5, w2, w4 
EOR  x2, x2, x1, lsr #42 
LDR  x4, [x1] 
STP  w4, w0, [x1, #208] 
B.hi  .bb_0.2 
B.nv .exit_0 
.bb_0.2:
MSUB  w1, w3, w0, w5 
AND  w3, w5, #2096640 
ORR  w4, w4, w3, lsl #23 
LDR  x4, [x2], #-163 
ANDS  w0, w0, w4 
TBNZ  x4, #21, .bb_0.3 
B.al .exit_0 
.bb_0.3:
LDP  x4, x2, [x5], #-176 
STR  x5, [x4, #21344] 
LDP  x5, x4, [x2], #24 
LDR  w4, [x1] 
REV  x1, x0 
CRC32B  w0, w1, w0 
B.al .bb_0.4 
.bb_0.4:
ADDS  w5, w0, w5 
STP  w5, w5, [x1, #128] 
STP  x2, x1, [x0, #280] 
UDIV  x0, x2, x2 
LDR  x4, [x0, #159]! 
LDR  x5, [x5] 
.exit_0:
.macro.measurement_end: NOP
B.nv .test_case_exit 
.section .data.main
.test_case_exit:
