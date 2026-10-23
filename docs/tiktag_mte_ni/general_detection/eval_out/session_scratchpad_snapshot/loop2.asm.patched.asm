.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
orr x3, xzr, #8
.bb_0.1:
subs x3, x3, #1
sdiv x0, x1, x2
sdiv x0, x0, x2
subs xzr, x0, #0
b.ne .bb_0.3
.bb_0.2:
ldr x9, [x1]
ldr x11, [x2]
.bb_0.3:
cbnz x3, .bb_0.1
.exit_0:
.macro.measurement_end: nop
b .test_case_exit
.section .data.main
.test_case_exit:
