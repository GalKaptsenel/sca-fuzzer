.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
STR  x5, [x3, #-133]! 
SETF8  w1 
LDR  x1, [x3, #-97]! 
SUBS  x4, x1, x4 
REV  x1, x5 
STR  w4, [x5, #7564] 
ORR  w4, w5, #67108352 
CSEL  x4, x5, x1, ne 
LDP  x3, x5, [x0] 
B.hi  .bb_0.1 
B.al .exit_0 
.bb_0.1:
LDP  w3, w5, [x2, #128] 
STP  x5, x5, [x3] 
LDR  x4, [x2, #98]! 
LDR  x2, [x0], #53 
LDR  x0, [x5, #5]! 
ADDS  w1, w2, w1 
CSNEG  x0, x5, x4, hi 
TBZ  w5, #0, .bb_0.2 
B.al .exit_0 
.bb_0.2:
LDP  w5, w1, [x2, #-128] 
LDR  w1, [x2, x0] 
AND  x0, x0, #72057576874835964 
LDR  w1, [x2, #99]! 
CRC32CH  w0, w5, w0 
CLS  w3, w4 
STR  w4, [x1, #92]! 
AND  x5, x4, x4 
.exit_0:
.macro.measurement_end: NOP
B.nv .test_case_exit 
.section .data.main
.test_case_exit:
