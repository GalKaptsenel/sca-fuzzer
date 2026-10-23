.section .data.main
.macro.measurement_start:
    nop
.function_main:
.macro.measurement_start:
    nop
.bb_main.0:
.macro.measurement_start:
    nop
mov x3, #8
.macro.measurement_start:
    nop
.bb_main.loop:
.macro.measurement_start:
    nop
subs x3, x3, #1
.macro.measurement_start:
    nop
dc civac, x0
.macro.measurement_start:
    nop
dsb sy
.macro.measurement_start:
    nop
isb
.macro.measurement_start:
    nop
ldr x8, [x0]
.macro.measurement_start:
    nop
cbz x8, .bb_main.skip
.macro.measurement_start:
    nop
ldr x9, [x1]
.macro.measurement_start:
    nop
ldr x11, [x2]
.macro.measurement_start:
    nop
.bb_main.skip:
.macro.measurement_start:
    nop
cbnz x3, .bb_main.loop
.macro.measurement_start:
    nop
.bb_main.exit:
.macro.measurement_start:
    nop
