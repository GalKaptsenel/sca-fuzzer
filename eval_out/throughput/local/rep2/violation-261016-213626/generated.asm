.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
CCMN  w1, #21, #13, cs 
SUBS  w3, w2, w0 
B.al .bb_0.1 
.bb_0.1:
BICS  w5, w5, w2 
LDR  x2, [x5] 
REV16  x4, x3 
STP  x5, x2, [x3, #344] 
TBZ  w2, #11, .bb_0.2 
B .exit_0 
.bb_0.2:
SDIV  w2, w0, w1 
LDR  x4, [x1] 
STR  w4, [x3, x5] 
CCMN  x0, x1, #9, gt 
B.nv .bb_0.3 
.bb_0.3:
STR  w3, [x0, #-134]! 
LDR  w5, [x0, #13624] 
CRC32CW  w1, w5, w4 
STR  x2, [x5, x0] 
SDIV  x5, x0, x5 
B.al .bb_0.4 
.bb_0.4:
STR  x5, [x5, x4] 
ANDS  w0, w3, w2 
LDP  w1, w2, [x3, #248] 
UDIV  w5, w0, w4 
LDP  x3, x4, [x5] 
LDR  w1, [x1, x0] 
STR  x2, [x4], #140 
ADDS  w3, w2, #2605, lsl #12 
STR  x2, [x5, x2] 
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit 
.section .data.main
.test_case_exit:
