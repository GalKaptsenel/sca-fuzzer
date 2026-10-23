mov x3, #512
.Lloop:
subs x3, x3, #1
csel x6, x5, x1, eq
csel x10, x2, x4, eq
cset x7, ne
ldr x12, [x1]
str x7, [x0]
dc civac, x0
dc civac, x2
dc civac, x4
dsb sy
isb
ldr x8, [x0]
cbz x8, .Lskip
ldr x9, [x6]
ldr x9, [x6]
ldr x9, [x6]
ldr x9, [x6]
ldr x9, [x6]
ldr x9, [x6]
ldr x9, [x6]
ldr x9, [x6]
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
orr x10, x10, x9
ldr x11, [x10]
.Lskip:
cbnz x3, .Lloop
