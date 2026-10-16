.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
REV  w0, w4 
STP  x4, x3, [x0], #48 
LDR  w4, [x2] 
CRC32W  w2, w0, w4 
LDP  x4, x5, [x2], #304 
EOR  x3, x2, x5, asr #59 
TBNZ  w3, #9, .bb_0.1 
B .bb_0.2 
.bb_0.1:
LDP  x0, x2, [x1], #208 
STR  x3, [x2], #-9 
LDP  x4, x2, [x3], #72 
STR  w3, [x5], #-74 
STR  w2, [x1], #-47 
LDR  x0, [x2, x1] 
ADDS  w0, w4, #403, lsl #0 
SUBS  x1, x0, #2510 
LDP  x2, x5, [x1], #480 
STR  w0, [x4, #233]! 
LDP  w0, w5, [x5, #-64] 
STR  w2, [x0, #11956] 
B .bb_0.2 
.bb_0.2:
CRC32CB  w5, w2, w0 
CCMN  x5, x1, #12, eq 
LDR  x1, [x5, x1] 
CLZ  x5, x5 
ADDS  w4, w5, w5 
LDP  x0, x2, [x3, #-112] 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
