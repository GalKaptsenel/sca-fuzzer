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

// =================================================================================================
// Helpers
// =================================================================================================

#define GET_NEXT_SET(NEXT_SET, SOURCE) \
    "and "NEXT_SET", "SOURCE", #0xFFFF                                                      \n" \
    "ror "SOURCE", "SOURCE", #16                                                            \n" \

#define COMBINE_TO_LABEL(number, value) #number"_"#value

#define load_imm32(name, imm)                                                                   \
    asm volatile(""								                                                \
		 "mov "name", %[low32]              \n"	                                                \
		 "movk "name", %[high32], lsl 16    \n"	                                                \
		 :										                                                \
		 : [low32] "i"((uint16_t)(imm & 0xFFFF)),	                                            \
		 [high32] "i"((uint16_t)(imm >> 16))		                                            \
		 :										                                                \
		)

// =================================================================================================
// Template building blocks
// =================================================================================================



#define ADJUST_REGISTER_TO(BASE, type, field) asm volatile(""	                                \
    "add "BASE", "BASE", #%[offset]                                                         \n" \
	:                                                                                           \
	: [offset] "i"(offsetof(type, field))			                                            \
	:							                                                                \
	)

#define ADJUST_REGISTER_FROM(BASE, type, field) asm volatile(""	                                \
    "sub "BASE", "BASE", #%[offset]                                                         \n" \
	:							                                                                \
	: [offset] "i"(offsetof(type, field))			                                            \
	:							                                                                \
	)

inline void prologue(void)
{
    // As we don't use a compiler to track clobbering,
    // we have to save the callee-saved regs
    asm volatile(""                                                                             \
	"stp x16, x17, [sp, #-16]!                                                              \n" \
	"stp x18, x19, [sp, #-16]!                                                              \n" \
	"stp x20, x21, [sp, #-16]!                                                              \n" \
	"stp x22, x23, [sp, #-16]!                                                              \n" \
	"stp x24, x25, [sp, #-16]!                                                              \n" \
	"stp x26, x27, [sp, #-16]!                                                              \n" \
	"stp x28, x29, [sp, #-16]!                                                              \n" \
	"str x30, [sp, #-16]!                                                                   \n" \

	// x30 <- input base address (stored in x0, the first argument of measurement_code)
	"mov x30, x0                                                                            \n" \

	// stored_rsp <- sp
	// "str sp, [x30, #"xstr(offsetof(sandbox_t, stored_rsp))"]\n"
	"mov x0, sp                                                                             \n" \
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
        "add x16, x16, x30                                                                  \n" \
        "stp x15, x20, [x16]                                                                \n" \
        "stp x21, x22, [x16, #16]                                                           \n" \

        // rsp <- stored_rsp
        "ldr x0, [x30, x0]                                                                  \n" \
        "mov sp, x0                                                                         \n" \

        // restore registers
       "ldr x30, [sp], #16                                                                  \n" \
        "ldp x28, x29, [sp], #16                                                            \n" \
        "ldp x26, x27, [sp], #16                                                            \n" \
        "ldp x24, x25, [sp], #16                                                            \n" \
        "ldp x22, x23, [sp], #16                                                            \n" \
        "ldp x20, x21, [sp], #16                                                            \n" \
        "ldp x18, x19, [sp], #16                                                            \n" \
        "ldp x16, x17, [sp], #16                                                            \n" \
    );
}



#define SET_REGISTER_FROM_INPUT(TMP) asm volatile(""			                                \
    "add "TMP", x30, #%[upper_overflow]\n"					                                    \
    "ldp x0, x1, ["TMP"], #16\n"						                                        \
    "ldp x2, x3, ["TMP"], #16\n"						                                        \
    "ldp x4, x5, ["TMP"], #16\n"						                                        \
    "ldp x6, x7, ["TMP"], #16\n"						                                        \
    "msr nzcv, x6\n"							                                                \
    "mov sp, x7\n"							                                                    \
	:								                                                            \
	: [upper_overflow] "i"(offsetof(sandbox_t, upper_overflow))	                                \
	:								                                                            \
	)

// clobber: -
// dest: x20, x21, x22
#define READ_PFC_START() asm volatile(""                                                        \
    "eor x20, x20, x20\n"		                                                                \
    "eor x21, x21, x21\n"		                                                                \
    "eor x22, x22, x22\n"		                                                                \
    "isb; dsb SY \n"		                                                                    \
    "mrs x20, pmevcntr1_el0 \n"                                                                 \
    "mrs x21, pmevcntr2_el0 \n"	                                                                \
    "mrs x22, pmevcntr3_el0 \n");

// clobber: x1
// dest: x20, x21, x22
#define READ_PFC_END() asm volatile(""                                                          \
    "isb; dsb SY \n"		                                                                    \
    "mrs x1, pmevcntr1_el0 \n"	                                                                \
    "sub x20, x1, x20 \n"	                                                                    \
    "mrs x1, pmevcntr2_el0 \n"	                                                                \
    "sub x21, x1, x21 \n"	                                                                    \
    "mrs x1, pmevcntr3_el0 \n"	                                                                \
    "sub x22, x1, x22 \n");

// =================================================================================================
// L1D Prime+Probe
// =================================================================================================

// clobber: -
#define PRIME(BASE, OFFSET, TMP, ASSOC_CTR, COUNTER, REPS) asm volatile(""	                    \
    "isb; dsb SY							                                                \n"	\
    "mov "COUNTER", "REPS"						                                            \n"	\
    										                                                    \
    "_arm64_executor_prime_outer:					                                        \n"	\
    "	mov "OFFSET", #"xstr(L1D_CONFLICT_DISTANCE)"			                            \n"	\
    										                                                    \
    "_arm64_executor_prime_inner:					                                        \n"	\
    "	sub "OFFSET", "OFFSET", #64					                                        \n"	\
    "	add "TMP", "BASE", #%[eviction_region]				                                \n"	\
    "	add "TMP", "TMP", "OFFSET"					                                        \n"	\
    "	mov "ASSOC_CTR", "xstr(L1D_ASSOCIATIVITY)"			                                \n"	\
    										                                                    \
    "_arm64_executor_prime_inner_assoc:					                                    \n"	\
    "	isb; dsb SY							                                                \n"	\
    "	str xzr, ["TMP"]						                                            \n"	\
    "	isb; dsb SY							                                                \n"	\
    "	add "TMP", "TMP", #"xstr(L1D_CONFLICT_DISTANCE)"		                            \n"	\
    "	sub "ASSOC_CTR", "ASSOC_CTR", #1				                                    \n"	\
    "	cbnz "ASSOC_CTR",_arm64_executor_prime_inner_assoc				                    \n"	\
    "	isb; dsb SY							                                                \n"	\
    										                                                    \
    "	cbnz "OFFSET", _arm64_executor_prime_inner		                                    \n"	\
										                                                        \
    "	sub "COUNTER", "COUNTER", #1					                                    \n"	\
    "	cbnz "COUNTER", _arm64_executor_prime_outer				                            \n" \
										                                                        \
    "isb; dsb SY							                                                \n"	\
	:									                                                        \
	: [eviction_region] "i"(offsetof(sandbox_t, eviction_region))		                        \
	:									                                                        \
)

#if L1D_SIZE_K == 16

#define AGGREGATE_ONE_SET(TMPBASE, OFFSET)         \
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\

#elif L1D_SIZE_K == 32

#define AGGREGATE_ONE_SET(TMPBASE, OFFSET)         \
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\

#elif L1D_SIZE_K == 64

#define AGGREGATE_ONE_SET(TMPBASE, OFFSET)         \
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\

#elif L1D_SIZE_K == 128

#define AGGREGATE_ONE_SET(TMPBASE, OFFSET)         \
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\
	"add "TMPBASE", "TMPBASE", #4096			\n"	\
	"ldr xzr, ["TMPBASE", "OFFSET"]			\n"	\                                              \n" \

#else
#error "Unexpected associativity"
#endif

#define SETS_PROBE_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, LABEL_NUM)                \
    "   lsl "DEST", "DEST", #1                                                  \n" \
    "	add "TMP", "BASE", #%[eviction_region]				                    \n"	\
        GET_NEXT_SET(OFFSET, OFFSETS)                                               \
    "   lsl "OFFSET", "OFFSET", #6                                              \n" \
    "   mrs "ACC", pmevcntr3_el0                                                \n" \
                                                                                    \
    "   isb; dsb SY                                                             \n" \
	    AGGREGATE_ONE_SET(TMP, OFFSET)						                        \
    "   isb; dsb SY                                                             \n" \
                                                                                    \
    "   mrs "TMP", pmevcntr3_el0                                                \n" \
    "   isb; dsb SY                                                             \n" \
    "   cmp "ACC", "TMP"                                                        \n" \
    "   b.eq _arm64_executor_probe_failed_"LABEL_NUM"                                 \n" \
                                                                                    \
    "   mov "TMP", #1								\n" \
    "   rsl "OFFSET", "OFFSET", #6                                              \n" \
    "   lsl "OFFSET", "TMP", "OFFSET"                                              \n" \
    "   orr "DEST", "DEST", "OFFSET"                                                  \n" \
                                                                                    \
    "_arm64_executor_probe_failed_"LABEL_NUM":                                        \n" \

#define SETS_PROBE(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, INIT_LABEL_NUM)                        \
    SETS_PROBE_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 0))   \
    SETS_PROBE_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 1))   \
    SETS_PROBE_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 2))   \
    SETS_PROBE_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 3))   \
    SETS_PROBE_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 4))   \
    SETS_PROBE_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 5))   \
    SETS_PROBE_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 6))   \
    SETS_PROBE_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 7))   \

// clobber: -
#define PROBE_SCATTERED(BASE, OFFSET, OFFSETS, TMP, ACC, DEST) asm volatile (""	\
    "eor "DEST", "DEST", "DEST"								\n" \
    "movz "OFFSETS", #2									\n" \
    "movk "OFFSETS", #54,	lsl #08							\n" \
    "movk "OFFSETS", #27,	lsl #16							\n" \
    "movk "OFFSETS", #22,	lsl #24							\n" \
    "movk "OFFSETS", #5,	lsl #32							\n" \
    "movk "OFFSETS", #35,	lsl #40							\n" \
    "movk "OFFSETS", #48,	lsl #48							\n" \
    "movk "OFFSETS", #25,	lsl #56							\n" \
    SETS_PROBE(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 0)				\
    "movz "OFFSETS", #42								\n" \
    "movk "OFFSETS", #8,	lsl #08							\n" \
    "movk "OFFSETS", #58,	lsl #16							\n" \
    "movk "OFFSETS", #13,	lsl #24							\n" \
    "movk "OFFSETS", #10,	lsl #32							\n" \
    "movk "OFFSETS", #31,	lsl #40							\n" \
    "movk "OFFSETS", #62,	lsl #48							\n" \
    "movk "OFFSETS", #36,	lsl #56							\n" \
    SETS_PROBE(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 8)				\
    "movz "OFFSETS", #46								\n" \
    "movk "OFFSETS", #20,	lsl #08							\n" \
    "movk "OFFSETS", #0,	lsl #16							\n" \
    "movk "OFFSETS", #30,	lsl #24							\n" \
    "movk "OFFSETS", #43,	lsl #32							\n" \
    "movk "OFFSETS", #44,	lsl #40							\n" \
    "movk "OFFSETS", #14,	lsl #48							\n" \
    "movk "OFFSETS", #52,	lsl #56							\n" \
    SETS_PROBE(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 16)				\
    "movz "OFFSETS", #16								\n" \
    "movk "OFFSETS", #7,	lsl #08							\n" \
    "movk "OFFSETS", #6,	lsl #16							\n" \
    "movk "OFFSETS", #41,	lsl #24							\n" \
    "movk "OFFSETS", #18,	lsl #32							\n" \
    "movk "OFFSETS", #49,	lsl #40							\n" \
    "movk "OFFSETS", #12,	lsl #48							\n" \
    "movk "OFFSETS", #45,	lsl #56							\n" \
    SETS_PROBE(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 24)				\
    "movz "OFFSETS", #4									\n" \
    "movk "OFFSETS", #38,	lsl #08							\n" \
    "movk "OFFSETS", #1,	lsl #16							\n" \
    "movk "OFFSETS", #26,	lsl #24							\n" \
    "movk "OFFSETS", #28	lsl #32							\n" \
    "movk "OFFSETS", #29,	lsl #40							\n" \
    "movk "OFFSETS", #63,	lsl #48							\n" \
    "movk "OFFSETS", #21,	lsl #56							\n" \
    SETS_PROBE(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 32)				\
    "movz "OFFSETS", #60								\n" \
    "movk "OFFSETS", #24,	lsl #08							\n" \
    "movk "OFFSETS", #23,	lsl #16							\n" \
    "movk "OFFSETS", #47,	lsl #24							\n" \
    "movk "OFFSETS", #53,	lsl #32							\n" \
    "movk "OFFSETS", #19,	lsl #40							\n" \
    "movk "OFFSETS", #61,	lsl #48							\n" \
    "movk "OFFSETS", #59,	lsl #56							\n" \
    SETS_PROBE(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 40)				\
    "movz "OFFSETS", #34								\n" \
    "movk "OFFSETS", #55,	lsl #08							\n" \
    "movk "OFFSETS", #32,	lsl #16							\n" \
    "movk "OFFSETS", #9,	lsl #24							\n" \
    "movk "OFFSETS", #56,	lsl #32							\n" \
    "movk "OFFSETS", #15,	lsl #40							\n" \
    "movk "OFFSETS", #17,	lsl #48							\n" \
    "movk "OFFSETS", #51,	lsl #56							\n" \
    SETS_PROBE(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 48)				\
    "movz "OFFSETS", #39								\n" \
    "movk "OFFSETS", #50,	lsl #08							\n" \
    "movk "OFFSETS", #40,	lsl #16							\n" \
    "movk "OFFSETS", #3,	lsl #24							\n" \
    "movk "OFFSETS", #11,	lsl #32							\n" \
    "movk "OFFSETS", #57,	lsl #40 						\n" \
    "movk "OFFSETS", #33,	lsl #48 						\n" \
    "movk "OFFSETS", #37,	lsl #56 						\n" \
    SETS_PROBE(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 56)				\
   )




// =================================================================================================
// Flush+Reload
// =================================================================================================

// clobber: -
#define FLUSH(BASE, OFFSET, TMP) asm volatile(""                                                \
    "isb; dsb SY							                                                \n"	\
    "eor "OFFSET", "OFFSET", "OFFSET"					                                    \n"	\
    										                                                    \
    "_arm64_executor_flush_loop:					                                        \n"	\
    "	add "TMP", "BASE", "OFFSET"					                                        \n"	\
    										                                                    \
    "	isb; dsb SY							                                                \n"	\
    "	dc ivac, "TMP"							                                            \n"	\
    "	isb; dsb SY							                                                \n"	\
    										                                                    \
    "	add "OFFSET", "OFFSET", #64					                                        \n"	\
    										                                                    \
    "	mov "TMP", #%[main_region_size]				                                        \n"	\
    "	cmp "TMP", "OFFSET"						                                            \n"	\
    "	b.gt _arm64_executor_flush_loop					                                    \n"	\
    										                                                    \
    "isb; dsb SY							                                                \n"	\
	:									                                                        \
	: [main_region_size] "i"(sizeof(executor.sandbox.main_region))			                    \
	:									                                                        \
)

#define SETS_RELOAD_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, LABEL_NUM)         \
    "   lsl "DEST", "DEST", #1                                                  \n" \
        GET_NEXT_SET(OFFSET, OFFSETS)                                               \
    "   lsl "OFFSET", "OFFSET", #6                                              \n" \
    "   mrs "ACC", pmevcntr3_el0                                                \n" \
                                                                                    \
    "   isb; dsb SY                                                             \n" \
    "   ldr xzr, ["BASE", "OFFSET"]                                             \n" \
    "   isb; dsb SY                                                             \n" \
                                                                                    \
    "   mrs "TMP", pmevcntr3_el0                                                \n" \
    "   isb; dsb SY                                                             \n" \
    "   cmp "ACC", "TMP"                                                        \n" \
    "   b.ne _arm64_executor_reload_failed_"LABEL_NUM"                                \n" \
                                                                                    \
    "   mov "TMP", #1								\n" \
    "   rsl "OFFSET", "OFFSET", #6                                              \n" \
    "   lsl "OFFSET", "TMP", "OFFSET"                                              \n" \
    "   orr "DEST", "DEST", "OFFSET"                                                  \n" \
                                                                                    \
    "_arm64_executor_reload_failed_"LABEL_NUM":                                 \n" \

#define SETS_RELOAD(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, INIT_LABEL_NUM)                        \
    SETS_RELOAD_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 0))   \
    SETS_RELOAD_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 1))   \
    SETS_RELOAD_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 2))   \
    SETS_RELOAD_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 3))   \
    SETS_RELOAD_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 4))   \
    SETS_RELOAD_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 5))   \
    SETS_RELOAD_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 6))   \
    SETS_RELOAD_INNER(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, COMBINE_TO_LABEL(INIT_LABEL_NUM, 7))   \

// clobber: -
#define RELOAD_SCATTERED(BASE, OFFSET, OFFSETS, TMP, ACC, DEST) asm volatile (""	\
    "eor "DEST", "DEST", "DEST"								\n" \
    "movz "OFFSETS", #2									\n" \
    "movk "OFFSETS", #54,	lsl #08							\n" \
    "movk "OFFSETS", #27,	lsl #16							\n" \
    "movk "OFFSETS", #22,	lsl #24							\n" \
    "movk "OFFSETS", #5,	lsl #32							\n" \
    "movk "OFFSETS", #35,	lsl #40							\n" \
    "movk "OFFSETS", #48,	lsl #48							\n" \
    "movk "OFFSETS", #25,	lsl #56							\n" \
    SETS_RELOAD(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 0)				\
    "movz "OFFSETS", #42								\n" \
    "movk "OFFSETS", #8,	lsl #08							\n" \
    "movk "OFFSETS", #58,	lsl #16							\n" \
    "movk "OFFSETS", #13,	lsl #24							\n" \
    "movk "OFFSETS", #10,	lsl #32							\n" \
    "movk "OFFSETS", #31,	lsl #40							\n" \
    "movk "OFFSETS", #62,	lsl #48							\n" \
    "movk "OFFSETS", #36,	lsl #56							\n" \
    SETS_RELOAD(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 8)				\
    "movz "OFFSETS", #46								\n" \
    "movk "OFFSETS", #20,	lsl #08							\n" \
    "movk "OFFSETS", #0,	lsl #16							\n" \
    "movk "OFFSETS", #30,	lsl #24							\n" \
    "movk "OFFSETS", #43,	lsl #32							\n" \
    "movk "OFFSETS", #44,	lsl #40							\n" \
    "movk "OFFSETS", #14,	lsl #48							\n" \
    "movk "OFFSETS", #52,	lsl #56							\n" \
    SETS_RELOAD(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 16)				\
    "movz "OFFSETS", #16								\n" \
    "movk "OFFSETS", #7,	lsl #08							\n" \
    "movk "OFFSETS", #6,	lsl #16							\n" \
    "movk "OFFSETS", #41,	lsl #24							\n" \
    "movk "OFFSETS", #18,	lsl #32							\n" \
    "movk "OFFSETS", #49,	lsl #40							\n" \
    "movk "OFFSETS", #12,	lsl #48							\n" \
    "movk "OFFSETS", #45,	lsl #56							\n" \
    SETS_RELOAD(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 24)				\
    "movz "OFFSETS", #4									\n" \
    "movk "OFFSETS", #38,	lsl #08							\n" \
    "movk "OFFSETS", #1,	lsl #16							\n" \
    "movk "OFFSETS", #26,	lsl #24							\n" \
    "movk "OFFSETS", #28	lsl #32							\n" \
    "movk "OFFSETS", #29,	lsl #40							\n" \
    "movk "OFFSETS", #63,	lsl #48							\n" \
    "movk "OFFSETS", #21,	lsl #56							\n" \
    SETS_RELOAD(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 32)				\
    "movz "OFFSETS", #60								\n" \
    "movk "OFFSETS", #24,	lsl #08							\n" \
    "movk "OFFSETS", #23,	lsl #16							\n" \
    "movk "OFFSETS", #47,	lsl #24							\n" \
    "movk "OFFSETS", #53,	lsl #32							\n" \
    "movk "OFFSETS", #19,	lsl #40							\n" \
    "movk "OFFSETS", #61,	lsl #48							\n" \
    "movk "OFFSETS", #59,	lsl #56							\n" \
    SETS_RELOAD(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 40)				\
    "movz "OFFSETS", #34								\n" \
    "movk "OFFSETS", #55,	lsl #08							\n" \
    "movk "OFFSETS", #32,	lsl #16							\n" \
    "movk "OFFSETS", #9,	lsl #24							\n" \
    "movk "OFFSETS", #56,	lsl #32							\n" \
    "movk "OFFSETS", #15,	lsl #40							\n" \
    "movk "OFFSETS", #17,	lsl #48							\n" \
    "movk "OFFSETS", #51,	lsl #56							\n" \
    SETS_RELOAD(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 48)				\
    "movz "OFFSETS", #39								\n" \
    "movk "OFFSETS", #50,	lsl #08							\n" \
    "movk "OFFSETS", #40,	lsl #16							\n" \
    "movk "OFFSETS", #3,	lsl #24							\n" \
    "movk "OFFSETS", #11,	lsl #32							\n" \
    "movk "OFFSETS", #57,	lsl #40 						\n" \
    "movk "OFFSETS", #33,	lsl #48 						\n" \
    "movk "OFFSETS", #37,	lsl #56 						\n" \
    SETS_RELOAD(BASE, OFFSET, OFFSETS, TMP, ACC, DEST, 56)				\
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
		size_t MEASUREMENT_METHOD_value = *MEASUREMENT_METHOD_func_address;				        \
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
	PROBE_SCATTERED("x30", "x16", "x17", "x18", "x19", "x15");

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
	RELOAD_SCATTERED("x30", "x16", "x17", "x18", "x19", "x15");

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
	const uint64_t max_size_of_template = MAX_MEASUREMENT_CODE_SIZE - 4; // guarantee enough memory for "ret" instruction
	size_t print_pos = 0;


	if(UNSET_TEMPLATE == executor.config.measurement_template) {
		module_err("Template is not set!");
		return -5;
	}

	map_to_template(executor.config.measurement_template)(&template_ptr);
	switch(executor.config.measurement_template) {
		case PRIME_AND_PROBE_TEMPLATE: module_debug("loading prime and probe!\n");
					       break;
		case FLUSH_AND_RELOAD_TEMPLATE: module_debug("loading flush and reload!\n");
					       break;
		default: module_debug("loading tamplate unset!\n");
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
	    module_debug("%px -> %lx\n", ((uint32_t*)executor.measurement_code) + print_pos,
	     ((uint32_t*)executor.measurement_code)[print_pos]);
    }
    return (sizeof(uint32_t) * code_pos);
}
