.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
str xzr, [x5]
ldr x12, [x5]
orr x10, xzr, #2048
sdiv x9, x1, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
sdiv x9, x9, x2
orr x9, x9, #1
cbnz x9, .bb_0.2
.bb_0.1:
ldr x12, [x5]
.macro.random_instructions.24.12.0: nop
.bb_0.2:
.macro.random_instructions.8.0.1: nop
.macro.measurement_end: nop
b .test_case_exit
.section .data.main
.test_case_exit:
