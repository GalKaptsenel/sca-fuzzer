.test_case_enter:
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: NOP

add x1, x1, x29
dsb sy
isb
ssbb
pssbb
sb
csdb
add x0, x0, x29
dsb sy
isb
ssbb
pssbb
sb
csdb

mov x2, #20
dsb sy
isb
ssbb
pssbb
sb
csdb
mov x3, #0
dsb sy
isb
ssbb
pssbb
sb
csdb
Lloop:

ldr x4, [x1]
dsb sy
isb
ssbb
pssbb
sb
csdb
cbnz x3, Ldone
dsb sy
isb
ssbb
pssbb
sb
csdb
ldr x5, [x0]
dsb sy
isb
ssbb
pssbb
sb
csdb
dc civac, x0
dsb sy
isb
ssbb
pssbb
sb
csdb
dc civac, x1
dsb sy
isb
ssbb
pssbb
sb
csdb
sub x2, x2, #1
dsb sy
isb
ssbb
pssbb
sb
csdb
cbnz x2, Lloop
dsb sy
isb
ssbb
pssbb
sb
csdb
mov x3, #1
dsb sy
isb
ssbb
pssbb
sb
csdb
b Lloop

Ldone:

.exit_0:
.macro.measurement_end: NOP
B.al .test_case_exit
.section .data.main
.test_case_exit:
