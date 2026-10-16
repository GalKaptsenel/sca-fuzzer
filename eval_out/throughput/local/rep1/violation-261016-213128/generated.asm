.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
STR  x2, [x1, #-109]! 
SUBS  w5, w1, #750, lsl #0 
LDR  x0, [x5, #11664] 
CRC32X  w5, w5, x4 
TBZ  w0, #16, .bb_0.1 
B.nv .exit_0 
.bb_0.1:
LDR  w0, [x5], #-136 
LDR  x4, [x1] 
TBNZ  w1, #25, .bb_0.2 
B.nv .bb_0.3 
.bb_0.2:
ADDS  w3, w3, #3632, lsl #12 
ADDS  w2, w4, #1888, lsl #12 
LDP  w3, w5, [x2], #-60 
STP  w0, w4, [x2], #92 
LDR  w5, [x3] 
CSINV  x5, x2, x1, ls 
B.al .bb_0.3 
.bb_0.3:
CSINV  x0, x1, x0, cs 
LDR  x2, [x0, #30]! 
LDR  w2, [x5, #-146]! 
STP  x2, x4, [x1, #192] 
CCMN  w0, w2, #0, gt 
B .bb_0.4 
.bb_0.4:
REV16  w4, w1 
BICS  w5, w0, w3 
LDR  x2, [x4, #352] 
LDR  x0, [x5, #15400] 
ANDS  w4, w5, w2, asr #6 
LDP  w3, w4, [x2], #-240 
BICS  w4, w2, w3 
.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit 
.section .data.main
.test_case_exit:
