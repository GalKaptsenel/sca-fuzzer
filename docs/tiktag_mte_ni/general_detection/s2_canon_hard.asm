.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
ORR x10, xzr, #2048
ORR x6, xzr, #7
ORR x9, xzr, #65535
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
SDIV x9, x9, x6
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
.bb_0.2:
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
