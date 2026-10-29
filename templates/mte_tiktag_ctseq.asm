.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP
STG x1, [x1]
ORR x3, xzr, #16
.bb_0.1:
SUBS x3, x3, #1
CSET x8, ne
AND x5, x4, #0xf
LSL x5, x5, #56
AND x6, x1, #0xf0ffffffffffffff
ORR x6, x6, x5
CSEL x7, x6, x1, eq
SDIV x9, x1, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
SDIV x9, x9, x2
MADD x10, x9, x8, xzr
CBZ x10, .bb_0.3
.bb_0.2:
LDR x11, [x7]
LDR x11, [x7]
LDR x11, [x7]
LDR x11, [x7]
LDR x11, [x7]
LDR x11, [x7]
LDR x11, [x7]
LDR x11, [x7]
ORR x14, xzr, xzr
ORR x15, xzr, #2048
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
ORR x15, x15, x14
LDR x13, [x15]
.bb_0.3:
CBNZ x3, .bb_0.1
.exit_0:
.macro.measurement_end: NOP
B .test_case_exit
.section .data.main
.test_case_exit:
