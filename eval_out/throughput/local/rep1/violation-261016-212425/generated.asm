.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
CRC32W  w4, w2, w3 
CSNEG  x2, x4, x4, eq 
CRC32B  w0, w2, w0 
CRC32H  w0, w5, w4 
LDP  w1, w4, [x5, #40] 
B .bb_0.1 
.bb_0.1:
LDR  x4, [x1, x2] 
SUBS  w1, w1, w4, lsl #27 
REV16  w1, w1 
ANDS  x0, x1, x2 
BICS  w2, w2, w5 
CRC32B  w4, w5, w1 
BICS  w1, w5, w3, lsl #12 
TBZ  x0, #30, .bb_0.2 
B.nv .exit_0 
.bb_0.2:
CRC32X  w0, w0, x0 
SUBS  w0, w2, w2 
LDR  w2, [x3, #2968] 
LDR  w0, [x2, x3] 
EOR  x3, x1, x1 
B.al .bb_0.3 
.bb_0.3:
LDR  x1, [x3], #18 
STR  x5, [x2] 
SUBS  x4, x5, #214, lsl #0 
CSNEG  x4, x4, x2, nv 
B .bb_0.4 
.bb_0.4:
CLZ  w4, w1 
ORR  w5, w0, w1 
STP  x3, x3, [x2], #-184 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
