.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
ORR x3, xzr, #16
ORR x10, xzr, #2048
.bb_0.1:
SUBS x3, x3, #1
SDIV x9, x1, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
MADD x11, x9, x3, xzr
CBZ x11, .bb_0.3
.bb_0.2:
LDR x12, [x5]
ORR x10, x10, x12
ORR x10, x10, x12
ORR x10, x10, x12
ORR x10, x10, x12
ORR x10, x10, x12
ORR x10, x10, x12
ORR x10, x10, x12
ORR x10, x10, x12
LDR x13, [x10]
.bb_0.3:
CBNZ x3, .bb_0.1
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
