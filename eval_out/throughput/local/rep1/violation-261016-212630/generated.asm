.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
LDR  w1, [x0, #-24]! 
LDP  x2, x1, [x0], #-384 
STR  w1, [x4, #178]! 
CSNEG  w3, w1, w5, lt 
EOR  w0, w2, w5 
B.al .bb_0.1 
.bb_0.1:
CRC32CX  w5, w1, x4 
SUBS  x0, x5, #365, lsl #12 
REV  x4, x2 
STR  w0, [x3, #14808] 
STR  x5, [x3] 
STP  w0, w2, [x2, #20] 
B .bb_0.2 
.bb_0.2:
CRC32W  w3, w0, w1 
CLZ  w5, w2 
STR  x1, [x5, x1] 
CBNZ  x3, .bb_0.3 
B .bb_0.4 
.bb_0.3:
LDP  x2, x5, [x3], #368 
LDR  w5, [x1, x5] 
ADDS  x5, x4, #1902, lsl #0 
UDIV  w1, w5, w0 
CRC32CB  w1, w5, w0 
LDP  w0, w5, [x2], #-112 
TBZ  w5, #9, .bb_0.4 
B.al .exit_0 
.bb_0.4:
CLZ  w0, w4 
LDP  w3, w2, [x4, #224] 
RMIF  x0, #61, #8 
LDR  w3, [x3, #5560] 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
