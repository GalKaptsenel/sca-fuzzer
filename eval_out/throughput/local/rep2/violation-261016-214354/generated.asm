.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
REV16  w0, w1 
LDP  w2, w1, [x4], #252 
B.nv .bb_0.1 
.bb_0.1:
ADDS  w5, w1, w5, lsl #5 
CRC32H  w4, w5, w2 
TBNZ  w4, #0, .bb_0.2 
B .bb_0.4 
.bb_0.2:
CLS  w2, w4 
SETF16  w1 
ADDS  x1, x2, #121, lsl #0 
ADDS  x2, x0, x3 
ADDS  x5, x2, #3794 
STR  x5, [x1, x0] 
CBZ  x1, .bb_0.3 
B.nv .exit_0 
.bb_0.3:
STP  x5, x4, [x1, #-144] 
SUBS  w4, w3, #842 
CCMP  w1, #14, #10, eq 
STR  x4, [x5, #14]! 
SUBS  x5, x3, x4, lsl #4 
B.al .bb_0.4 
.bb_0.4:
LDP  w1, w5, [x0], #20 
ADDS  x3, x2, #3291 
RBIT  w5, w4 
STR  w1, [x3], #-151 
AND  x5, x4, x2 
LDR  x5, [x3], #212 
STR  x4, [x1, #-22]! 
STP  x0, x5, [x4], #-24 
LDR  x1, [x2, #17624] 
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit 
.section .data.main
.test_case_exit:
