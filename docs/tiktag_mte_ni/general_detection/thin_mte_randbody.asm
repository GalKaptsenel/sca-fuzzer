.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
SDIV x9, x1, x2
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
.macro.random_instructions.64.40.0:
.bb_0.2:
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
