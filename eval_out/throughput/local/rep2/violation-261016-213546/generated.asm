.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
EOR  x1, x2, x2 
STR  x3, [x0, #-26]! 
LDR  x5, [x1], #99 
CCMN  x0, #21, #0, vc 
CBNZ  w5, .bb_0.1 
B.al .bb_0.3 
.bb_0.1:
LDR  x2, [x1] 
ANDS  w0, w0, w0 
AND  w2, w1, #4294967167 
LDR  w2, [x0, #51]! 
LDR  w4, [x1] 
AND  w4, w2, #4294938623 
CRC32X  w1, w2, x4 
CBZ  w3, .bb_0.2 
B .exit_0 
.bb_0.2:
ADDS  w1, w1, w4 
CLZ  x4, x2 
CLZ  w1, w0 
LDP  w4, w2, [x1] 
ORR  x5, x0, x4, asr #29 
TBZ  x1, #19, .bb_0.3 
B.al .exit_0 
.bb_0.3:
ADDS  w4, w5, #1071, lsl #12 
LDP  w4, w0, [x1] 
LDP  w5, w1, [x0], #144 
AND  w4, w1, w1 
LDR  w3, [x3, #1500] 
CLZ  w1, w2 
ORR  w1, w0, w5, lsr #10 
UDIV  x3, x3, x2 
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit 
.section .data.main
.test_case_exit:
