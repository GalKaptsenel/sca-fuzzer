.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
ORR x10, xzr, #2048
SDIV x9, x1, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
ORR x9, x9, #1
CBZ x9, .bb_0.2
.bb_0.1:
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
.bb_0.2:
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
