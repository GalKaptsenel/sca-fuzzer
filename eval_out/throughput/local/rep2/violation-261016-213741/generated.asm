.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
REV16  x5, x3 
LDR  w1, [x1, x0] 
LDP  x0, x4, [x5], #-304 
B.vc  .bb_0.1 
B.nv .bb_0.3 
.bb_0.1:
BICS  x0, x2, x1, asr #8 
LDP  x4, x2, [x1] 
CSEL  w3, w0, w0, mi 
BICS  x5, x4, x0, lsl #57 
LDR  x3, [x2], #-66 
B .bb_0.2 
.bb_0.2:
CSEL  w5, w0, w3, nv 
STR  w3, [x0], #-125 
ADDS  x0, x5, #3430, lsl #0 
LDR  x5, [x4] 
CCMP  x5, #21, #0, ls 
TBNZ  x0, #2, .bb_0.3 
B.nv .exit_0 
.bb_0.3:
LDP  x3, x0, [x1], #64 
STP  w0, w4, [x0] 
EOR  x5, x2, #18446744073701294079 
AND  x5, x4, #18446744073701163007 
CBNZ  x5, .bb_0.4 
B .exit_0 
.bb_0.4:
SUBS  w5, w2, w0 
REV  w3, w1 
LDP  w2, w3, [x5], #-64 
LDR  w2, [x0] 
STR  x4, [x2, #-221]! 
SUBS  w3, w0, #3912, lsl #0 
STP  x1, x5, [x1] 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
