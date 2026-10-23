ldr x12, [x4]
dc civac, x2
udiv x5, x0, x0
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
udiv x5, x5, x5
madd x7, x0, x5, xzr
str xzr, [x7]
ldr x8, [x0]
subs xzr, x8, #0
b.eq .Lskip
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
ldr x9, [x1]
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
.Lskip:
