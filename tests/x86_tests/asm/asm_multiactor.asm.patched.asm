.intel_syntax noprefix
.test_case_enter:
.section .data.main

.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
nop
nop

.section .data.guest_1
.function_1:
nop

.section .data.main
.function_2:
.bb0:
nop

# .section exit
.section .data.main
.function_end:
.macro.measurement_end: nop qword ptr [rax + 0xff]
.test_case_exit:nop
