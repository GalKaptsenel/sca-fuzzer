.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
str xzr, [x5]
ldr x12, [x5]
orr x10, xzr, #2048
orr x6, xzr, #7
orr x9, xzr, #65535
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
sdiv x9, x9, x6
orr x9, x9, #1
cbnz x9, .bb_0.2
.bb_0.1:
ldr x12, [x5]
and x11, x12, xzr
orr x10, x10, x11
orr x10, x10, x11
orr x10, x10, x11
orr x10, x10, x11
orr x10, x10, x11
orr x10, x10, x11
orr x10, x10, x11
orr x10, x10, x11
ldr x13, [x10]
.bb_0.2:
.macro.measurement_end: nop
b .test_case_exit
.section .data.main
.test_case_exit:
