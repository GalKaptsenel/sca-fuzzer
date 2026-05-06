.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
AND x0, x0, #0x40
AND x1, x1, #0xFC0
b.ge .bb_0.2
B .bb_0.1

.bb_0.1:
LDR x3, [x1]
B .exit_0

.bb_0.2:

.exit_0:
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
