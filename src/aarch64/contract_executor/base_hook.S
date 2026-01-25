.section .data
.global base_hook, base_hook_end, base_hook_c_target
.type base_hook, %function

base_hook:
	/* Save context */
	stp x1, x0, [sp, #-16]!
	stp x3, x2, [sp, #-16]!
	stp x5, x4, [sp, #-16]!
	stp x7, x6, [sp, #-16]!
	stp x9, x8, [sp, #-16]!
	stp x11, x10, [sp, #-16]!
	stp x13, x12, [sp, #-16]!
	stp x15, x14, [sp, #-16]!
	stp x17, x16, [sp, #-16]!
	stp x19, x18, [sp, #-16]!
	stp x21, x20, [sp, #-16]!
	stp x23, x22, [sp, #-16]!
	stp x25, x24, [sp, #-16]!
	stp x27, x26, [sp, #-16]!
	stp x29, x28, [sp, #-16]!

	mov x0, lr
	sub x0, x0, 4 // store the PC of the instruction being hooked
	stp x0, lr, [sp, #-16]!

	mrs x0, nzcv
	add lr, sp, #(32 * 8) // store SP as it was before the above pushes into the stack: 8 bytes per registers * 32 registers saved above = 0x100 bytes = 256 bytes
	stp lr, x0, [sp, #-16]!


	// call the hook with x0 is the start of the above struct (aka its top)
	// Notice that LR is overriden by the bl instruction, which is ok because we already saved it above
	ldr lr, base_hook_c_target
	mov x0, sp
	blr lr               // call C helper


	ldp xzr, x0, [sp], #16 // pops out the SP and NZCV values, we don't need to restore SP as it will be restored by the following 'ldp' instructions
	msr nzcv, x0

	ldp xzr, lr, [sp], #16 // pops out PC value of the instruction being hooked, we don't need to restore it as it will be restored by 'ret'

	ldp x29, x28, [sp], #16
	ldp x27, x26, [sp], #16
	ldp x25, x24, [sp], #16
	ldp x23, x22, [sp], #16
	ldp x21, x20, [sp], #16
	ldp x19, x18, [sp], #16
	ldp x17, x16, [sp], #16
	ldp x15, x14, [sp], #16
	ldp x13, x12, [sp], #16
	ldp x11, x10, [sp], #16
	ldp x9, x8, [sp], #16
	ldp x7, x6, [sp], #16
	ldp x5, x4, [sp], #16
	ldp x3, x2, [sp], #16
	ldp x1, x0, [sp], #16
	ret

        .align 3              // 8-byte alignment
base_hook_c_target:
        .quad 0               // patched at runtime

base_hook_end:
