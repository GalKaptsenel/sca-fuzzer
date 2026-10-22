.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
STR xzr, [x5]
LDR x12, [x5]
ORR x10, xzr, #2048
LDR x7, [x1]
UDIV x6, x1, x1
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
UDIV x6, x6, x6
MADD x4, x1, x6, xzr
STR xzr, [x4]
LDR x8, [x1]
CBZ x8, .bb_0.2
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
