.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
ORR x3, xzr, #8
.bb_0.1:
SUBS x3, x3, #1
SDIV x0, x1, x2
SDIV x0, x0, x2
SUBS xzr, x0, #0
B.ne .bb_0.3
.bb_0.2:
LDR x9, [x1]
LDR x11, [x2]
.bb_0.3:
CBNZ x3, .bb_0.1
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
