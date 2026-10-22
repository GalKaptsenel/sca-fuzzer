.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
STR xzr, [x5]
LDR x12, [x5]
ORR x10, xzr, #2048
SDIV x9, x1, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
ORR x9, x9, #1
CBNZ x9, .bb_0.2
.bb_0.1:
LDR x12, [x5]
AND x11, x12, xzr
ORR x10, x10, x11
ORR x10, x10, x11
ORR x10, x10, x11
ORR x10, x10, x11
ORR x10, x10, x11
ORR x10, x10, x11
ORR x10, x10, x11
ORR x10, x10, x11
LDR x13, [x10]
.macro.random_instructions.16.8.0:
.bb_0.2:
.macro.random_instructions.16.8.1:
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
