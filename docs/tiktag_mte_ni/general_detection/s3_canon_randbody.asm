.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
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
ORR x9, x9, #1
CBNZ x9, .bb_0.2
.bb_0.1:
.macro.random_instructions.32.8.0:
.bb_0.2:
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
