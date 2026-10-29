.section .data
.global base_hook, base_hook_end, base_hook_c_target
.type base_hook, %function

/* Saves/restores x0-x29, LR, PC, SP, NZCV, and the whole vector file v0-v31 around base_hook_c. The
 * vector registers hold the GST image the test case reads via UMOV; base_hook_c's C callees
 * (fprintf/malloc/Python) are free to clobber caller-saved vectors, so v0-v31 are saved across the
 * call below. */
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

	// Preserve the whole vector file v0-v31 across the C hook. x0 (the saved-register struct) was
	// captured above, so this push is below it and does not disturb the struct or the reported SP.
	stp q0, q1, [sp, #-32]!
	stp q2, q3, [sp, #-32]!
	stp q4, q5, [sp, #-32]!
	stp q6, q7, [sp, #-32]!
	stp q8, q9, [sp, #-32]!
	stp q10, q11, [sp, #-32]!
	stp q12, q13, [sp, #-32]!
	stp q14, q15, [sp, #-32]!
	stp q16, q17, [sp, #-32]!
	stp q18, q19, [sp, #-32]!
	stp q20, q21, [sp, #-32]!
	stp q22, q23, [sp, #-32]!
	stp q24, q25, [sp, #-32]!
	stp q26, q27, [sp, #-32]!
	stp q28, q29, [sp, #-32]!
	stp q30, q31, [sp, #-32]!

	blr lr               // call C helper

	ldp q30, q31, [sp], #32
	ldp q28, q29, [sp], #32
	ldp q26, q27, [sp], #32
	ldp q24, q25, [sp], #32
	ldp q22, q23, [sp], #32
	ldp q20, q21, [sp], #32
	ldp q18, q19, [sp], #32
	ldp q16, q17, [sp], #32
	ldp q14, q15, [sp], #32
	ldp q12, q13, [sp], #32
	ldp q10, q11, [sp], #32
	ldp q8, q9, [sp], #32
	ldp q6, q7, [sp], #32
	ldp q4, q5, [sp], #32
	ldp q2, q3, [sp], #32
	ldp q0, q1, [sp], #32

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
