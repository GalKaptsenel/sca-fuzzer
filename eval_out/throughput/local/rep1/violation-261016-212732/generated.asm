.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
LDR  x4, [x1], #104 
LDR  w2, [x1], #-189 
CBNZ  x4, .bb_0.1 
B .exit_0 
.bb_0.1:
LDR  x3, [x0, x2] 
BICS  w0, w1, w4 
STR  w4, [x2, x1] 
TBZ  x3, #2, .bb_0.2 
B.nv .exit_0 
.bb_0.2:
ORR  x1, x1, #18410732868868247551 
LDP  w2, w3, [x4], #-48 
ADDS  w2, w5, w1 
STR  w2, [x3, x5] 
SUBS  x1, x0, #2835, lsl #12 
STP  w5, w5, [x4, #108] 
STR  w1, [x3, #135]! 
REV16  w5, w2 
BICS  w4, w1, w2, ror #23 
LDR  x4, [x3, x0] 
B.al .bb_0.3 
.bb_0.3:
REV32  x0, x2 
LDP  x4, x0, [x2, #80] 
STR  x5, [x0, #32688] 
REV16  x1, x3 
ORR  x5, x5, x1, asr #2 
LDP  x3, x1, [x2] 
LDR  x0, [x2], #-48 
CRC32CH  w5, w4, w3 
STP  w3, w2, [x5], #-160 
.exit_0:
.macro.measurement_end: NOP
B.nv .test_case_exit 
.section .data.main
.test_case_exit:
