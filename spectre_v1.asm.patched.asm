.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop
and x0, x0, #0x40
and x1, x1, #0xfc0
b.ge .bb_0.2
b .bb_0.1

.bb_0.1:
ldr x3, [x1]
b .exit_0

.bb_0.2:

.exit_0:
.macro.measurement_end: nop
b .test_case_exit
.section .data.main
.test_case_exit:
