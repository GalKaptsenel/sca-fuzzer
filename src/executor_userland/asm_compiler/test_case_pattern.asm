// Minimal AArch64 test-case body (loads from the sandbox main region via x0).
ldr x1, [x0]
add x1, x1, #1
str x1, [x0, #64]
nop
