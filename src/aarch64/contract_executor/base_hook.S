.section .data
.global base_hook, base_hook_end, base_hook_c_target, gvt_image_target
.type base_hook, %function

/* Saves/restores x0-x29, LR, PC, SP, NZCV around base_hook_c, and RELOADS the whole vector file
 * v0-v31 from the GVT image (sim_input.simd) after the call. The test case reads the GVT image via
 * UMOV but never writes the vector registers, so reloading them from the authoritative image each
 * step both (a) undoes the C callees' (fprintf/malloc/Python) clobbering of caller-saved vectors and
 * (b) lets an op-emulator inject a value by writing the image buffer during the hook — the UMOV then
 * reads it. gvt_image_target is patched at runtime to &sim_input.simd. */
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

	// Reload v0-v31 from the GVT image (sim_input.simd), which base_hook_c may have just updated (an
	// op-emulator injecting a value). x9 is restored from the stack below, so clobbering it is safe.
	ldr x9, gvt_image_target
	ldr q0,  [x9, #0]
	ldr q1,  [x9, #16]
	ldr q2,  [x9, #32]
	ldr q3,  [x9, #48]
	ldr q4,  [x9, #64]
	ldr q5,  [x9, #80]
	ldr q6,  [x9, #96]
	ldr q7,  [x9, #112]
	ldr q8,  [x9, #128]
	ldr q9,  [x9, #144]
	ldr q10, [x9, #160]
	ldr q11, [x9, #176]
	ldr q12, [x9, #192]
	ldr q13, [x9, #208]
	ldr q14, [x9, #224]
	ldr q15, [x9, #240]
	ldr q16, [x9, #256]
	ldr q17, [x9, #272]
	ldr q18, [x9, #288]
	ldr q19, [x9, #304]
	ldr q20, [x9, #320]
	ldr q21, [x9, #336]
	ldr q22, [x9, #352]
	ldr q23, [x9, #368]
	ldr q24, [x9, #384]
	ldr q25, [x9, #400]
	ldr q26, [x9, #416]
	ldr q27, [x9, #432]
	ldr q28, [x9, #448]
	ldr q29, [x9, #464]
	ldr q30, [x9, #480]
	ldr q31, [x9, #496]

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

        .align 3              // 8-byte alignment
gvt_image_target:
        .quad 0               // patched at runtime to &sim_input.simd

base_hook_end:
