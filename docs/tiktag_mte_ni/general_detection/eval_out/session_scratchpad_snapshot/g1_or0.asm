ldr x10, [x3]
dc civac, x0
dc civac, x2
dsb sy
isb
ldr x0, [x0]
cbz x0, .Ls
ldr x0, [x1]
ldr x0, [x1]
ldr x9, [x2]
.Ls:
nop
