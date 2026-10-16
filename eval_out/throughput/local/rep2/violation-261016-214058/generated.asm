.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
LDR  x1, [x4, #14]! 
ANDS  w4, w3, w5, lsr #5 
STR  x4, [x5] 
CBNZ  w1, .bb_0.1 
B .exit_0 
.bb_0.1:
SUBS  w1, w5, w4, lsl #11 
AND  x2, x3, #18442240474090569727 
STP  w1, w1, [x0, #-148] 
ORR  w4, w1, w0 
STP  x1, x0, [x4, #-192] 
STR  w0, [x1], #-141 
LDP  x1, x4, [x0], #-72 
STR  w2, [x3, #15024] 
CLS  x0, x5 
SUBS  x2, x3, #2242, lsl #0 
CBNZ  w4, .bb_0.2 
B.nv .exit_0 
.bb_0.2:
ADDS  w3, w2, #1744, lsl #0 
STP  x2, x2, [x5, #-184] 
CSNEG  x4, x0, x0, gt 
CRC32CW  w4, w4, w4 
SDIV  w4, w0, w2 
EOR  w4, w3, #4290772995 
CLS  w0, w5 
LDR  x3, [x4, x2] 
CCMP  w1, #21, #7, cs 
RBIT  x3, x2 
STP  w5, w0, [x1, #-132] 
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit 
.section .data.main
.test_case_exit:
