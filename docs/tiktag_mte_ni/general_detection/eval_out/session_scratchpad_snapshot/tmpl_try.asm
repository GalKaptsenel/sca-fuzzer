.section .data.main
.function_main:
.bb_main.0:
mov x3, #8
.bb_main.loop:
subs x3, x3, #1
dc civac, x0
dsb sy
isb
ldr x8, [x0]
cbz x8, .bb_main.skip
ldr x9, [x1]
ldr x11, [x2]
.bb_main.skip:
cbnz x3, .bb_main.loop
.bb_main.exit:
