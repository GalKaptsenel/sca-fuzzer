.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
LDR  w4, [x0], #244 
CSNEG  x4, x2, x1, gt 
AND  w0, w1, w1, lsl #15 
LDR  w3, [x5, #15092] 
LDR  w4, [x0, #212]! 
LDR  w3, [x2] 
CBNZ  x1, .bb_0.1 
B .bb_0.4 
.bb_0.1:
LDR  w2, [x1] 
B .bb_0.2 
.bb_0.2:
EOR  w1, w3, #8387584 
STR  w0, [x5, #12408] 
AND  w1, w4, #4286578695 
CSINV  x1, x4, x5, eq 
SUBS  w1, w4, w2, lsl #29 
EOR  x5, x1, x2 
CBNZ  x2, .bb_0.3 
B .bb_0.4 
.bb_0.3:
LDR  x2, [x0] 
LDR  x4, [x5], #20 
STR  x5, [x4] 
CSNEG  x4, x2, x0, ge 
LDR  w1, [x2, x4] 
CBZ  w2, .bb_0.4 
B .exit_0 
.bb_0.4:
ADDS  x5, x0, x0 
STR  w0, [x5, x3] 
LDR  w4, [x2, x1] 
LDR  w5, [x2] 
ADDS  w0, w1, #912 
LDR  x1, [x2], #176 
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit 
.section .data.main
.test_case_exit:
