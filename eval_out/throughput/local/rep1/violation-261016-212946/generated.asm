.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
CRC32B  w0, w3, w2 
B .bb_0.1 
.bb_0.1:
CCMN  w2, #23, #13, ls 
SUBS  w2, w2, w4 
ORR  x3, x3, x2, ror #12 
ORR  w0, w5, w4, lsl #25 
LDR  x2, [x0, x4] 
TBZ  w2, #7, .bb_0.2 
B.al .exit_0 
.bb_0.2:
LDR  x4, [x0], #75 
STR  w2, [x5, #10132] 
STR  w1, [x5] 
STP  w2, w2, [x2, #144] 
SUBS  x4, x1, x1 
B.al .bb_0.3 
.bb_0.3:
CRC32CW  w3, w0, w5 
AND  x5, x3, #65534 
SUBS  x0, x2, x0, asr #27 
LDR  x2, [x3] 
STR  w5, [x1, #-114]! 
LDR  w0, [x1, #-19]! 
CCMP  w2, w5, #2, cs 
TBNZ  w5, #22, .bb_0.4 
B.al .exit_0 
.bb_0.4:
LDR  w5, [x4], #-179 
ANDS  w0, w1, w0 
STP  w1, w0, [x1, #140] 
CRC32B  w4, w5, w0 
LDR  w4, [x4, x1] 
CLZ  w4, w5 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
