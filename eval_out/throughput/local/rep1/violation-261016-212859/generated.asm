.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
SUBS  w3, w3, w1, lsl #21 
CBZ  w5, .bb_0.1 
B .bb_0.2 
.bb_0.1:
LDR  x2, [x1, x3] 
CLZ  x3, x1 
MADD  w3, w0, w1, w0 
LDR  x1, [x3, #-10]! 
LDR  x1, [x2] 
LDP  w5, w4, [x3], #120 
LDR  x1, [x3] 
BICS  w5, w1, w0, asr #19 
CBNZ  w5, .bb_0.2 
B.nv .exit_0 
.bb_0.2:
LDR  x3, [x4, #4184] 
CLZ  w4, w1 
EOR  w0, w4, w1, ror #0 
EOR  w1, w4, w1, ror #18 
CCMN  w3, w1, #15, hi 
CBNZ  w3, .bb_0.3 
B .exit_0 
.bb_0.3:
SUBS  x5, x3, #1443 
BICS  w0, w5, w5, lsr #23 
LDP  x2, x5, [x2] 
LDR  x4, [x2, #18080] 
AND  x1, x0, x1, asr #40 
LDR  w2, [x1] 
EOR  x0, x1, x4, lsr #27 
TBNZ  x5, #26, .bb_0.4 
B .exit_0 
.bb_0.4:
SETF8  w3 
ANDS  w1, w3, w2, ror #7 
LDR  x2, [x0], #51 
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit 
.section .data.main
.test_case_exit:
