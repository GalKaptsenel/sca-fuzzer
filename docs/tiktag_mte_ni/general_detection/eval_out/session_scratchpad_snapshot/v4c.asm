ldr x13, [x0]
dc civac, x3
dc civac, x2
dsb sy
isb
ldr x14, [x3]
add x7, x0, x14
str xzr, [x7]
ldr x8, [x0]
subs xzr, x8, #0
b.eq .Ls
orr x6, xzr, xzr
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
orr x2, x2, x6
ldr x11, [x2]
.Ls:
