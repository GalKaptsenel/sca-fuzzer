.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
ldr x3, [x2]
ldr x2, [x3]
ldr x0, [x2]
cbnz x0, .bb_0.2
b .bb_0.1
.bb_0.1:
ldr x2, [x1]
b .exit_0
.bb_0.2:
b .exit_0
.exit_0:
.macro.measurement_end: nop
b .test_case_exit
.section .data.main
.test_case_exit:
