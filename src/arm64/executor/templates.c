#include "main.h"

// Note on registers.
// Some of the registers are reserved for a specific purpose and should never be overwritten.
// These include:
//   * X15 - hardware trace
//   * X20 - performance counter 1
//   * X21 - performance counter 2
//   * X22 - performance counter 3
//   * X30 - address of sandbox_t

// clang-format off

// Helper functions:

#define RANDOM_PERMUTATION(TMP0, TMP1, TMP2, TMP3, TMP4, TMP7, TMP8, TMP9) asm volatile("" \
		"movi v0.16b, 0							\n"	\
		"movi v1.16b, 16						\n"	\
		"movi v2.16b, 32						\n"	\
		"movi v3.16b, 48						\n"	\
											\
		"mov "TMP0", #63						\n"	\
											\
		"random_permutation_loop:					\n"	\
		"mrs "TMP1", RNDR						\n"	\
		"udiv "TMP2", "TMP1", "TMP0"					\n"	\
		"msub "TMP1", "TMP2", "TMP0", "TMP1"				\n"	\
											\
		"lsr "TMP5", "TMP0", #0x4					\n"	\
		"lsr "TMP6", "TMP1", #0x4					\n"	\
		"and "TMP3", "TMP0", #0xF					\n"	\
		"and "TMP4", "TMP1", #0xF					\n"	\
											\
		"mov "TMP7", random_permutation_jump_table			\n"	\
		"lsl "TMP5", "TMP5", #0x5					\n"	\
		"lsl "TMP6", "TMP6", #0x3					\n"	\
		"add "TMP7", "TMP7", "TMP5"					\n"	\
		"add "TMP7", "TMP7", "TMP6"					\n"	\
		"mov "TMP7", random_permutation_jump_table			\n"	\
		"ldr "TMP7", ["TMP7"]						\n"	\
		"br "TMP7"							\n"	\
		"random_permutation_continue_loop:				\n"	\
											\
		"sub "TMP0", "TMP0", 1						\n"	\
		"cbnz "TMP0", random_permutation_loop				\n"	\
		"b random_permutation_skip_jump_table				\n"	\
		"random_permutation_jump_table:					\n"	\
		".quad swap_v0_v0						\n"	\
		".quad swap_v0_v1						\n"	\
		".quad swap_v0_v2						\n"	\
		".quad swap_v0_v3						\n"	\
		".quad swap_v1_v0						\n"	\
		".quad swap_v1_v1						\n"	\
		".quad swap_v1_v2						\n"	\
		".quad swap_v1_v3						\n"	\
		".quad swap_v2_v0						\n"	\
		".quad swap_v2_v1						\n"	\
		".quad swap_v2_v2						\n"	\
		".quad swap_v2_v3						\n"	\
		".quad swap_v3_v0						\n"	\
		".quad swap_v3_v1						\n"	\
		".quad swap_v3_v2						\n"	\
		".quad swap_v3_v3						\n"	\
											\
		"swap_v0_v0:							\n"	\
		"	eor v0.b[x3], v0.b[x4]					\n"	\
		"	eor v0.b[x4], v0.b[x3]					\n"	\
		"	eor v0.b[x3], v0.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v0_v1:							\n"	\
		"	eor v0.b[x3], v1.b[x4]					\n"	\
		"	eor v1.b[x4], v0.b[x3]					\n"	\
		"	eor v0.b[x3], v1.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v0_v2:							\n"	\
		"	eor v0.b[x3], v2.b[x4]					\n"	\
		"	eor v2.b[x4], v0.b[x3]					\n"	\
		"	eor v0.b[x3], v2.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v0_v3:							\n"	\
		"	eor v0.b[x3], v3.b[x4]					\n"	\
		"	eor v3.b[x4], v0.b[x3]					\n"	\
		"	eor v0.b[x3], v3.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v1_v0:							\n"	\
		"	eor v1.b[x3], v0.b[x4]					\n"	\
		"	eor v0.b[x4], v1.b[x3]					\n"	\
		"	eor v1.b[x3], v0.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v1_v1:							\n"	\
		"	eor v1.b[x3], v1.b[x4]					\n"	\
		"	eor v1.b[x4], v1.b[x3]					\n"	\
		"	eor v1.b[x3], v1.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v1_v2:							\n"	\
		"	eor v1.b[x3], v2.b[x4]					\n"	\
		"	eor v2.b[x4], v1.b[x3]					\n"	\
		"	eor v1.b[x3], v2.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v1_v3:							\n"	\
		"	eor v1.b[x3], v3.b[x4]					\n"	\
		"	eor v3.b[x4], v1.b[x3]					\n"	\
		"	eor v1.b[x3], v3.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v2_v0:							\n"	\
		"	eor v2.b[x3], v0.b[x4]					\n"	\
		"	eor v0.b[x4], v2.b[x3]					\n"	\
		"	eor v2.b[x3], v0.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v2_v1:							\n"	\
		"	eor v2.b[x3], v1.b[x4]					\n"	\
		"	eor v1.b[x4], v2.b[x3]					\n"	\
		"	eor v2.b[x3], v1.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v2_v2:							\n"	\
		"	eor v2.b[x3], v2.b[x4]					\n"	\
		"	eor v2.b[x4], v2.b[x3]					\n"	\
		"	eor v2.b[x3], v2.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v2_v3:							\n"	\
		"	eor v2.b[x3], v3.b[x4]					\n"	\
		"	eor v3.b[x4], v2.b[x3]					\n"	\
		"	eor v2.b[x3], v3.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v3_v0:							\n"	\
		"	eor v3.b[x3], v0.b[x4]					\n"	\
		"	eor v0.b[x4], v3.b[x3]					\n"	\
		"	eor v3.b[x3], v0.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v3_v1:							\n"	\
		"	eor v3.b[x3], v1.b[x4]					\n"	\
		"	eor v1.b[x4], v3.b[x3]					\n"	\
		"	eor v3.b[x3], v1.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v3_v2:							\n"	\
		"	eor v3.b[x3], v2.b[x4]					\n"	\
		"	eor v2.b[x4], v3.b[x3]					\n"	\
		"	eor v3.b[x3], v2.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"swap_v3_v3:							\n"	\
		"	eor v3.b[x3], v3.b[x4]					\n"	\
		"	eor v3.b[x4], v3.b[x3]					\n"	\
		"	eor v3.b[x3], v3.b[x4]					\n"	\
		"	b random_permutation_continue_loop			\n"	\
											\
		"								\n"	\
		"random_permutation_skip_jump_table:				\n"	\
		)


// =================================================================================================
// Template building blocks
// =================================================================================================

#define load_imm32(name, imm)	\
    asm volatile(""										\
		 "mov "name", %[low32]\n"							\
		 "movk "name", %[high32], lsl 16\n"						\
		 :										\
		 : [low32] "i"((uint16_t)(imm & 0xFFFF)),					\
		 [high32] "i"((uint16_t)(imm >> 16))				\
		 :										\
		)

#define ADJUST_REGISTER_TO(BASE, type, field) asm volatile(""	\
    "add "BASE", "BASE", #%[offset]\n"				            \
	:                                                           \
	: [offset] "i"(offsetof(type, field))			            \
	:							                                \
	)

#define ADJUST_REGISTER_FROM(BASE, type, field) asm volatile(""	\
    "sub "BASE", "BASE", #%[offset]\n"				            \
	:							                                \
	: [offset] "i"(offsetof(type, field))			            \
	:							                                \
	)

inline void prologue(void)
{
    // As we don't use a compiler to track clobbering,
    // we have to save the callee-saved regs
    asm volatile("" \
	"stp x16, x17, [sp, #-16]!\n"
	"stp x18, x19, [sp, #-16]!\n"
	"stp x20, x21, [sp, #-16]!\n"
	"stp x22, x23, [sp, #-16]!\n"
	"stp x24, x25, [sp, #-16]!\n"
	"stp x26, x27, [sp, #-16]!\n"
	"stp x28, x29, [sp, #-16]!\n"
	"str x30, [sp, #-16]!\n"

	// x30 <- input base address (stored in x0, the first argument of measurement_code)
	"mov x30, x0\n"

	// stored_rsp <- sp
	// "str sp, [x30, #"xstr(offsetof(sandbox_t, stored_rsp))"]\n"
	"mov x0, sp\n"
    );
	ADJUST_REGISTER_TO("x30", sandbox_t, stored_rsp);
    asm volatile("str x0, [x30]\n");
	ADJUST_REGISTER_FROM("x30", sandbox_t, stored_rsp);
}

inline void epilogue(void) {

	load_imm32("x16", offsetof(sandbox_t, latest_measurement));
	load_imm32("x0", offsetof(sandbox_t, stored_rsp));

    asm volatile(""
        // store the hardware trace (x15) and pfc readings (x20-x22)
        "add x16, x16, x30\n"
        "stp x15, x20, [x16]\n"
        "stp x21, x22, [x16, #16]\n"

        // rsp <- stored_rsp
        "ldr x0, [x30, x0]\n"
        "mov sp, x0\n"

        // restore registers
       "ldr x30, [sp], #16\n"
        "ldp x28, x29, [sp], #16\n"
        "ldp x26, x27, [sp], #16\n"
        "ldp x24, x25, [sp], #16\n"
        "ldp x22, x23, [sp], #16\n"
        "ldp x20, x21, [sp], #16\n"
        "ldp x18, x19, [sp], #16\n"
        "ldp x16, x17, [sp], #16\n"
    );
}



#define SET_REGISTER_FROM_INPUT(TMP) asm volatile(""			    \
    "add "TMP", x30, #%[upper_overflow]\n"					        \
    "ldp x0, x1, ["TMP"], #16\n"						            \
    "ldp x2, x3, ["TMP"], #16\n"						            \
    "ldp x4, x5, ["TMP"], #16\n"						            \
    "ldp x6, x7, ["TMP"], #16\n"						            \
    "msr nzcv, x6\n"							                \
    "mov sp, x7\n"							                    \
	:								                            \
	: [upper_overflow] "i"(offsetof(sandbox_t, upper_overflow))	\
	:								                            \
	)

// clobber: -
// dest: x20, x21, x22
#define READ_PFC_START() asm volatile(""                        \
    "eor x20, x20, x20\n"		                                    \
    "eor x21, x21, x21\n"		                                    \
    "eor x22, x22, x22\n"		                                    \
    "isb; dsb SY \n"		                                    \
    "mrs x20, pmevcntr1_el0 \n"                                 \
    "mrs x21, pmevcntr2_el0 \n"	                                \
    "mrs x22, pmevcntr3_el0 \n");

// clobber: x1
// dest: x20, x21, x22
#define READ_PFC_END() asm volatile(""                          \
    "isb; dsb SY \n"		                                    \
    "mrs x1, pmevcntr1_el0 \n"	                                \
    "sub x20, x1, x20 \n"	                                    \
    "mrs x1, pmevcntr2_el0 \n"	                                \
    "sub x21, x1, x21 \n"	                                    \
    "mrs x1, pmevcntr3_el0 \n"	                                \
    "sub x22, x1, x22 \n");


// =================================================================================================
// L1D Prime+Probe
// =================================================================================================

// clobber: -
#define PRIME(BASE, OFFSET, TMP, ASSOC_CTR, COUNTER, REPS) asm volatile(""	\
    "isb; dsb SY							                                \n"	\
    "mov "COUNTER", "REPS"						                            \n"	\
    										                                    \
    "_arm64_executor_prime_outer:					                        \n"	\
    "	mov "OFFSET", #"xstr(L1D_CONFLICT_DISTANCE)"			                \n"	\
    										                                    \
    "_arm64_executor_prime_inner:					                        \n"	\
    "	sub "OFFSET", "OFFSET", #64					                        \n"	\
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
    "	add "TMP", "TMP", "OFFSET"					                        \n"	\
    "	mov "ASSOC_CTR", "xstr(L1D_ASSOCIATIVITY)"			                \n"	\
    										                                    \
    "_arm64_executor_prime_inner_assoc:					                    \n"	\
    "	isb; dsb SY							                                \n"	\
    "	str xzr, ["TMP"]						                        \n"	\
    "	isb; dsb SY							                                \n"	\
    "	add "TMP", "TMP", #"xstr(L1D_CONFLICT_DISTANCE)"		            \n"	\
    "	sub "ASSOC_CTR", "ASSOC_CTR", #1				                    \n"	\
    "	cbnz "ASSOC_CTR",_arm64_executor_prime_inner_assoc				                \n"	\
    "	isb; dsb SY							                                \n"	\
    										                                    \
    "	cbnz "OFFSET", _arm64_executor_prime_inner		                            \n"	\
										                                        \
    "	sub "COUNTER", "COUNTER", #1					                    \n"	\
    "	cbnz "COUNTER", _arm64_executor_prime_outer				             \n"	\
										                                        \
    "isb; dsb SY							                                \n"	\
	:									                                        \
	: [eviction_region] "i"(offsetof(sandbox_t, eviction_region))		        \
	:									                                        \
)

// clobber: -
#define PROBE(BASE, OFFSET, TMP, ASSOC_CTR, ACC, DEST) asm volatile(""	\
    "eor "DEST", "DEST", "DEST"						                        \n"	\
    "mov "OFFSET", #"xstr(L1D_CONFLICT_DISTANCE)"			                \n"	\
										                                        \
    "_arm64_executor_probe_loop:					                        \n"	\
    "	isb; dsb SY							                                \n"	\
    "	sub "OFFSET", "OFFSET", #64					                        \n"	\
    "	mov "ASSOC_CTR", "xstr(L1D_ASSOCIATIVITY)"			                \n"	\
										                                        \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
    "	add "TMP", "TMP", "OFFSET"					                        \n"	\
    "	mrs "ACC", pmevcntr0_el0					                        \n"	\
    										                                    \
    "_arm64_executor_probe_inner_assoc:					                    \n"	\
    "	isb; dsb SY							                                \n"	\
    "	ldr xzr, ["TMP"]						                        \n"	\
    "	isb; dsb SY							                                \n"	\
    "	sub "ASSOC_CTR", "ASSOC_CTR", #1				                    \n"	\
    "	add "TMP", "TMP", #"xstr(L1D_CONFLICT_DISTANCE)"		            \n"	\
    "	cbnz "ASSOC_CTR", _arm64_executor_probe_inner_assoc				    \n"	\
    "	isb; dsb SY							                                \n"	\
										                                        \
    "	mrs "TMP", pmevcntr0_el0					                        \n"	\
    "	lsl "DEST", "DEST", #1						                        \n"	\
    "	cmp "ACC", "TMP"						                            \n"	\
    "	b.eq _arm64_executor_probe_failed				                    \n"	\
    										                                    \
    "	orr "DEST", "DEST", #1						                        \n"	\
    										                                    \
    "_arm64_executor_probe_failed:					                        \n"	\
    "	cbnz "OFFSET", _arm64_executor_probe_loop					        \n"	\
	:									                                        \
	: [eviction_region] "i"(offsetof(sandbox_t, eviction_region))		        \
	:									                                        \
)

#define PRIME_ONE_SET(BASE, OFFSET)	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\
	"add "OFFSET", "OFFSET", #4096			\n"	\
	"ldr xzr, ["BASE", "OFFSET"]			\n"	\



#define PROBE_UNROLL(BASE, OFFSET, TMP, ACC, DEST) asm volatile (""	\
    "eor "OFFSET", "OFFSET", "OFFSET"                                           \n"     \
    "eor "DEST", "DEST", "DEST"                                                     \n" \
        "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #128														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_0                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_0:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3456														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_1                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_1:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1728														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_2                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_2:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1408														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_3                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_3:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #320														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_4                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_4:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2240														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_5                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_5:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3072														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_6                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_6:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1600														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_7                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_7:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2688														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_8                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_8:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #512														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_9                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_9:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3712														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_10                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_10:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #832														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_11                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_11:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #640														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_12                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_12:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1984														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_13                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_13:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3968														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_14                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_14:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2304														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_15                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_15:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2944														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_16                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_16:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1280														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_17                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_17:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #0														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_18                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_18:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1920														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_19                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_19:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2752														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_20                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_20:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2816														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_21                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_21:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #896														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_22                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_22:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3328														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_23                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_23:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1024														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_24                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_24:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #448														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_25                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_25:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #384														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_26                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_26:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2624														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_27                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_27:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1152														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_28                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_28:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3136														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_29                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_29:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #768														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_30                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_30:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2880														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_31                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_31:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #256														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_32                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_32:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2432														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_33                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_33:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #64														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_34                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_34:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1664														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_35                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_35:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1792														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_36                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_36:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1856														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_37                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_37:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #4032														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_38                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_38:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1344														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_39                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_39:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3840														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_40                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_40:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1536														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_41                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_41:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1472														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_42                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_42:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3008														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_43                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_43:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3392														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_44                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_44:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1216														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_45                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_45:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3904														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_46                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_46:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3776														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_47                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_47:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2176														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_48                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_48:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3520														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_49                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_49:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2048														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_50                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_50:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #576														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_51                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_51:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3584														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_52                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_52:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #960														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_53                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_53:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #1088														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_54                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_54:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3264														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_55                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_55:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2496														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_56                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_56:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3200														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_57                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_57:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2560														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_58                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_58:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #192														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_59                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_59:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #704														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_60                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_60:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #3648														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_61                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_61:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2112														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_62                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_62:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                \n"	\
	"	mov "OFFSET", #2368														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
	PRIME_ONE_SET(TMP, OFFSET)						\
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.eq _arm64_executor_probe_failed_63                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_probe_failed_63:                                              \n" \
	:									                                        \
	: [eviction_region] "i"(offsetof(sandbox_t, eviction_region))		        \
	:										\
   )




// =================================================================================================
// Flush+Reload
// =================================================================================================

// clobber: -
#define FLUSH(BASE, OFFSET, TMP) asm volatile(""                    \
    "isb; dsb SY							                    \n"	\
    "eor "OFFSET", "OFFSET", "OFFSET"					        \n"	\
    										                        \
    "_arm64_executor_flush_loop:					            \n"	\
    "	add "TMP", "BASE", "OFFSET"					            \n"	\
    										                        \
    "	isb; dsb SY							                    \n"	\
    "	dc ivac, "TMP"							                \n"	\
    "	isb; dsb SY							                    \n"	\
    										                        \
    "	add "OFFSET", "OFFSET", #64					            \n"	\
    										                        \
    "	mov "TMP", #%[main_region_size]				    \n"	\
    "	cmp "TMP", "OFFSET"						                \n"	\
    "	b.gt _arm64_executor_flush_loop					        \n"	\
    										                        \
    "isb; dsb SY							                    \n"	\
	:									                            \
	: [main_region_size] "i"(sizeof(executor.sandbox.main_region))			\
	:									                            \
)

// clobber: -
#define RELOAD(BASE, OFFSET, TMP, ACC, DEST) asm volatile("" \
    "eor "OFFSET", "OFFSET", "OFFSET"					        \n"	\
    "eor "DEST", "DEST", "DEST"						            \n"	\
    										                        \
    "_arm64_executor_reload_loop:					            \n"	\
    "	lsl "DEST", "DEST", #1						            \n"	\
    "	mov "TMP", "BASE"						        \n"	\
    "	mrs "ACC", pmevcntr3_el0					            \n"	\
                                                           			\
    "	isb; dsb SY							                    \n"	\
    "	ldr xzr, ["TMP", "OFFSET"]   			                \n"	\
    "	isb; dsb SY							                    \n"	\
    										                        \
    "	mrs "TMP", pmevcntr3_el0					            \n"	\
    "	isb; dsb SY							                    \n"	\
    "	cmp "ACC", "TMP"						                \n"	\
    "	b.ne _arm64_executor_reload_failed				        \n"	\
										                            \
    "	orr "DEST", "DEST", #1						            \n"	\
										                            \
    "_arm64_executor_reload_failed:					            \n"	\
    "	add "OFFSET", "OFFSET", #64					            \n"	\
    "	mov "TMP", #%[main_region_size]				    \n"	\
    "	cmp "TMP", "OFFSET"						                \n"	\
    "	b.gt _arm64_executor_reload_loop				        \n"	\
	:									                            \
	: [main_region_size] "i"(sizeof(executor.sandbox.main_region))			\
	:									                            \
)

#define RELOAD_UNROLL(BASE, OFFSET, TMP, ACC, DEST) asm volatile (""	\
    "eor "OFFSET", "OFFSET", "OFFSET"                                           \n"     \
    "eor "DEST", "DEST", "DEST"                                                     \n" \
    										                        \
        "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #128														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_0                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_0:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3456														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_1                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_1:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1728														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_2                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_2:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1408														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_3                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_3:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #320														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_4                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_4:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2240														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_5                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_5:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3072														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_6                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_6:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1600														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_7                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_7:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2688														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_8                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_8:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #512														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_9                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_9:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3712														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_10                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_10:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #832														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_11                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_11:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #640														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_12                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_12:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1984														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_13                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_13:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3968														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_14                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_14:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2304														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_15                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_15:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2944														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_16                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_16:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1280														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_17                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_17:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #0														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_18                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_18:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1920														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_19                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_19:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2752														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_20                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_20:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2816														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_21                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_21:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #896														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_22                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_22:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3328														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_23                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_23:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1024														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_24                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_24:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #448														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_25                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_25:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #384														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_26                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_26:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2624														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_27                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_27:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1152														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_28                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_28:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3136														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_29                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_29:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #768														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_30                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_30:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2880														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_31                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_31:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #256														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_32                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_32:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2432														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_33                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_33:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #64														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_34                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_34:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1664														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_35                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_35:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1792														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_36                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_36:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1856														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_37                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_37:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #4032														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_38                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_38:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1344														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_39                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_39:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3840														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_40                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_40:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1536														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_41                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_41:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1472														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_42                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_42:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3008														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_43                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_43:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3392														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_44                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_44:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1216														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_45                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_45:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3904														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_46                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_46:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3776														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_47                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_47:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2176														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_48                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_48:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3520														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_49                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_49:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2048														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_50                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_50:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #576														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_51                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_51:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3584														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_52                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_52:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #960														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_53                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_53:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #1088														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_54                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_54:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3264														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_55                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_55:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2496														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_56                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_56:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3200														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_57                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_57:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2560														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_58                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_58:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #192														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_59                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_59:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #704														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_60                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_60:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #3648														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_61                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_61:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2112														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_62                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_62:                                              \n" \
    "   lsl "DEST", "DEST", #1                                                      \n" \
    "   mov "TMP", "BASE"                                                       \n"     \
	"	mov "OFFSET", #2368														\n"		\
    "   mrs "ACC", pmevcntr3_el0                                                    \n" \
                                                                                \
    "   isb; dsb SY                                                                         \n" \
    "   ldr xzr, ["TMP", "OFFSET"]                                      \n"     \
    "   isb; dsb SY                                                                         \n" \
                                                                                                        \
    "   mrs "TMP", pmevcntr3_el0                                                    \n" \
    "   isb; dsb SY                                                                         \n" \
    "   cmp "ACC", "TMP"                                                                \n"     \
    "   b.ne _arm64_executor_reload_failed_63                                   \n"     \
                                                                                                            \
    "   orr "DEST", "DEST", #1                                                      \n" \
                                                                                                            \
    "_arm64_executor_reload_failed_63:                                              \n" \
   )


/* MACRO: MEASUREMENT_METHOD:
 This MACRO prepends assembly boilerplate code that retrieves an address prior to TEMPLATE_ENTER within the function.
 The if statement inside the boilerplate code meant to guarantee that the compiler will continue to generate the rest of
 the code it gets from the preprocessor.
 It works because the compiler cannot guarantee that the statement will allways evaluate to True, and therefore
 he will continue to generate the rest of the function code.
 OUTPUT of the function is an address A s.t. exists c>=0 s.t. *(A+c)==TEMPLATE_ENTER */
#define MEASUREMENT_METHOD(func_name, ...)									                    \
	static void func_name(size_t* MEASUREMENT_METHOD_func_address, ##__VA_ARGS__)	{			\
		size_t MEASUREMENT_METHOD_value = *MEASUREMENT_METHOD_func_address;				\
		asm volatile ("adr %[output], .": [output]"=r"(*MEASUREMENT_METHOD_func_address));		\
		if(0 == MEASUREMENT_METHOD_value) return;

MEASUREMENT_METHOD(template_l1d_prime_probe)
	asm volatile(".long "xstr(TEMPLATE_ENTER));

	// ensure that we don't crash because of BTI
	asm volatile("bti c");

	prologue();

	// Initialize registers
	SET_REGISTER_FROM_INPUT("sp");

	PRIME("x30", "x16", "x17", "x18", "x19", "32");

	ADJUST_REGISTER_TO("x30", sandbox_t, main_region);

	READ_PFC_START();

	// Execute the test case
	asm(
		"\nisb; dsb SY\n"
		".long "xstr(TEMPLATE_INSERT_TC)" \n"
		"isb; dsb SY\n"
	);

	READ_PFC_END();

	ADJUST_REGISTER_FROM("x30", sandbox_t, main_region);

	// Probe and store the resulting eviction bitmap map into x15
	PROBE_UNROLL("x30", "x16", "x17", "x18", "x15");

	epilogue();
	asm volatile(".long "xstr(TEMPLATE_RETURN));
}

MEASUREMENT_METHOD(template_l1d_flush_reload)
	asm volatile(".long "xstr(TEMPLATE_ENTER));

	// ensure that we don't crash because of BTI
	asm volatile("bti c");

	prologue();

	// Initialize registers
	SET_REGISTER_FROM_INPUT("sp");

	ADJUST_REGISTER_TO("x30", sandbox_t, main_region);

	FLUSH("x30", "x16", "x17");

	READ_PFC_START();

	// Execute the test case
	asm(
		"\nisb; dsb SY\n"
		".long "xstr(TEMPLATE_INSERT_TC)" \n"
		"isb; dsb SY\n"
	);

	READ_PFC_END();

	// Reload and store the resulting eviction bitmap map into x15
	RELOAD_UNROLL("x30", "x16", "x17", "x18", "x15");

	ADJUST_REGISTER_FROM("x30", sandbox_t, main_region);

	epilogue();
	asm volatile(".long "xstr(TEMPLATE_RETURN));
}

// clang-format on

typedef void (*measurement_template)(size_t*);
struct measurement_method {
	enum Templates type;
	measurement_template wrapper;
};

static struct measurement_method methods[] = {
	{PRIME_AND_PROBE_TEMPLATE,	template_l1d_prime_probe},
	{FLUSH_AND_RELOAD_TEMPLATE,	template_l1d_flush_reload},
};

static measurement_template map_to_template(enum Templates required_template) {

	for(int i = 0; i < (sizeof(methods) / sizeof(methods[0])); ++i) {
		if(required_template == methods[i].type) {
			return methods[i].wrapper;
		}
	}

	return NULL;
}

int load_template(size_t tc_size) {
	unsigned template_pos = 0;
	unsigned code_pos = 0;
	size_t template_ptr = 0;
	const uint64_t max_size_of_template = MAX_MEASUREMENT_CODE_SIZE - 4; // guarantee that there is enough space for "ret" instruction
	size_t print_pos = 0;


	if(UNSET_TEMPLATE == executor.config.measurement_template) {
		module_err("Template is not set!");
		return -5;
	}

	map_to_template(executor.config.measurement_template)(&template_ptr);
	switch(executor.config.measurement_template) {
		case PRIME_AND_PROBE_TEMPLATE: module_err("loading prime and probe!\n");
					       break;
		case FLUSH_AND_RELOAD_TEMPLATE: module_err("loading flush and reload!\n");
					       break;
		default: module_err("loading tamplate unset!\n");
	}

	// skip until the beginning of the template
	for (;	TEMPLATE_ENTER != ((uint32_t*)(template_ptr))[template_pos];
			++template_pos) {

		size_t current_offset = sizeof(uint32_t) * template_pos;
		if (max_size_of_template <= current_offset) {
			return -1;
		}

	}

	template_pos += (4/sizeof(uint32_t)); // skip TEMPLATE_ENTER

	// copy the first part of the template
	for (;	TEMPLATE_INSERT_TC != ((uint32_t*)(template_ptr))[template_pos];
			++template_pos, ++code_pos) {

		size_t current_offset = sizeof(uint32_t) * template_pos;
	    if (max_size_of_template <= current_offset) {
	 	return -1;
	    }

	    ((uint32_t*)executor.measurement_code)[code_pos] = ((uint32_t*)template_ptr)[template_pos];
	}

	template_pos += (4/sizeof(uint32_t)); // skip TEMPLATE_INSERT_TC

	// copy the test case into the template
	memcpy((uint32_t*)executor.measurement_code + code_pos, executor.test_case, tc_size);
	code_pos += (tc_size/sizeof(uint32_t));

	// write the rest of the template
	for (;	TEMPLATE_RETURN != ((uint32_t*)(template_ptr))[template_pos];
			++template_pos, ++code_pos) {
		size_t current_offset = sizeof(uint32_t) * template_pos;

	    if (max_size_of_template <= current_offset) {
	        return -2;
	    }

	    if (TEMPLATE_INSERT_TC == ((uint32_t*)(template_ptr))[template_pos]) {
	        return -3;
	    }

	    ((uint32_t*)executor.measurement_code)[code_pos] = ((uint32_t*)template_ptr)[template_pos];
	}

    // RET encoding: 0xd65f03c0
    ((uint32_t*)executor.measurement_code)[code_pos] = 0xd65f03c0;
    code_pos += 1;

    for(; print_pos < code_pos; ++print_pos) {
	    module_err("%px -> %lx\n", ((uint32_t*)executor.measurement_code) + print_pos, ((uint32_t*)executor.measurement_code)[print_pos]);
    }
    return (sizeof(uint32_t) * code_pos);
}

