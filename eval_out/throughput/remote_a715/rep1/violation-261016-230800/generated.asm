.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
STR  w1, [x3, #-25]! 
STR  x0, [x5, #11392] 
LDP  w1, w0, [x2] 
LDR  x4, [x1, #21200] 
STP  x2, x3, [x4] 
SDIV  w3, w4, w3 
B.al .bb_0.1 
.bb_0.1:
LDP  x5, x2, [x1, #-208] 
MSUB  x1, x1, x4, x1 
EOR  x5, x2, #72057576858058752 
ADDS  w1, w3, #1039, lsl #0 
ORR  w0, w1, w5, asr #26 
ANDS  x3, x4, #18230567993010552063 
REV16  w1, w0 
LDP  x5, x4, [x2], #-72 
TBZ  w5, #13, .bb_0.2 
B.nv .exit_0 
.bb_0.2:
LDP  w5, w0, [x1] 
LDP  w3, w1, [x5], #196 
REV16  x0, x5 
SUBS  w0, w3, #924, lsl #0 
B.al .bb_0.3 
.bb_0.3:
LDR  w4, [x2, #217]! 
LDP  w4, w0, [x4, #-132] 
STR  w4, [x2, x5] 
CCMN  w3, w3, #7, ne 
AND  w5, w3, w1 
CCMN  x1, x2, #1, cs 
.exit_0:
.macro.measurement_end: NOP
B.nv .test_case_exit 
.section .data.main
.test_case_exit:
