/// File: Kernel module interface
///
// Copyright (C) Microsoft Corporation
// SPDX-License-Identifier: MIT

// clang-format off

// INCLUDES
#include "main.h"

kallsyms_lookup_name_t kallsyms_lookup_name_fn	    =	NULL;

//#define RANDOM_PERMUTATION(TMP0, TMP1, TMP2, TMP3, TMP4, TMP5, TMP6, TMP7, TMP1W, TMP2W) asm volatile("" \
//		"movi v0.16b, 0							\n"	\
//		"movi v1.16b, 16						\n"	\
//		"movi v2.16b, 32						\n"	\
//		"movi v3.16b, 48						\n"	\
//											\
//		"mov "TMP0", #63						\n"	\
//											\
//		"random_permutation_loop:					\n"	\
//		"mrs "TMP1", RNDR						\n"	\
//		"udiv "TMP2", "TMP1", "TMP0"					\n"	\
//		"msub "TMP1", "TMP2", "TMP0", "TMP1"				\n"	\
//											\
//		"lsr "TMP5", "TMP0", #0x4					\n"	\
//		"lsr "TMP6", "TMP1", #0x4					\n"	\
//		"and "TMP3", "TMP0", #0xF					\n"	\
//		"and "TMP4", "TMP1", #0xF					\n"	\
//											\
//		"mov "TMP7", random_permutation_jump_table			\n"	\
//		"lsl "TMP5", "TMP5", #0x5					\n"	\
//		"lsl "TMP6", "TMP6", #0x3					\n"	\
//		"add "TMP7", "TMP7", "TMP5"					\n"	\
//		"add "TMP7", "TMP7", "TMP6"					\n"	\
//		"mov "TMP7", random_permutation_jump_table			\n"	\
//		"ldr "TMP7", ["TMP7"]						\n"	\
//		"br "TMP7"							\n"	\
//		"random_permutation_continue_loop:				\n"	\
//											\
//		"sub "TMP0", "TMP0", 1						\n"	\
//		"cbnz "TMP0", random_permutation_loop				\n"	\
//		"b random_permutation_skip_jump_table				\n"	\
//		"random_permutation_jump_table:					\n"	\
//		".quad swap_v0_v0						\n"	\
//		".quad swap_v0_v1						\n"	\
//		".quad swap_v0_v2						\n"	\
//		".quad swap_v0_v3						\n"	\
//		".quad swap_v1_v0						\n"	\
//		".quad swap_v1_v1						\n"	\
//		".quad swap_v1_v2						\n"	\
//		".quad swap_v1_v3						\n"	\
//		".quad swap_v2_v0						\n"	\
//		".quad swap_v2_v1						\n"	\
//		".quad swap_v2_v2						\n"	\
//		".quad swap_v2_v3						\n"	\
//		".quad swap_v3_v0						\n"	\
//		".quad swap_v3_v1						\n"	\
//		".quad swap_v3_v2						\n"	\
//		".quad swap_v3_v3						\n"	\
//											\
//		"swap_v0_v0:							\n"	\
//		"	ldrb "TMP1W", [v0, x3, lsl #2]					\n"	\
//		"	ldrb "TMP2W", [v0, x4]					\n"	\
//		"	strb "TMP1W", [v0, x4]					\n"	\
//		"	sdrb "TMP2W", [v0, x3]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v0_v1:							\n"	\
//		"	eor v0.b[x3], v1.b[x4]					\n"	\
//		"	eor v1.b[x4], v0.b[x3]					\n"	\
//		"	eor v0.b[x3], v1.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v0_v2:							\n"	\
//		"	eor v0.b[x3], v2.b[x4]					\n"	\
//		"	eor v2.b[x4], v0.b[x3]					\n"	\
//		"	eor v0.b[x3], v2.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v0_v3:							\n"	\
//		"	eor v0.b[x3], v3.b[x4]					\n"	\
//		"	eor v3.b[x4], v0.b[x3]					\n"	\
//		"	eor v0.b[x3], v3.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v1_v0:							\n"	\
//		"	eor v1.b[x3], v0.b[x4]					\n"	\
//		"	eor v0.b[x4], v1.b[x3]					\n"	\
//		"	eor v1.b[x3], v0.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v1_v1:							\n"	\
//		"	eor v1.b[x3], v1.b[x4]					\n"	\
//		"	eor v1.b[x4], v1.b[x3]					\n"	\
//		"	eor v1.b[x3], v1.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v1_v2:							\n"	\
//		"	eor v1.b[x3], v2.b[x4]					\n"	\
//		"	eor v2.b[x4], v1.b[x3]					\n"	\
//		"	eor v1.b[x3], v2.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v1_v3:							\n"	\
//		"	eor v1.b[x3], v3.b[x4]					\n"	\
//		"	eor v3.b[x4], v1.b[x3]					\n"	\
//		"	eor v1.b[x3], v3.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v2_v0:							\n"	\
//		"	eor v2.b[x3], v0.b[x4]					\n"	\
//		"	eor v0.b[x4], v2.b[x3]					\n"	\
//		"	eor v2.b[x3], v0.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v2_v1:							\n"	\
//		"	eor v2.b[x3], v1.b[x4]					\n"	\
//		"	eor v1.b[x4], v2.b[x3]					\n"	\
//		"	eor v2.b[x3], v1.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v2_v2:							\n"	\
//		"	eor v2.b[x3], v2.b[x4]					\n"	\
//		"	eor v2.b[x4], v2.b[x3]					\n"	\
//		"	eor v2.b[x3], v2.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v2_v3:							\n"	\
//		"	eor v2.b[x3], v3.b[x4]					\n"	\
//		"	eor v3.b[x4], v2.b[x3]					\n"	\
//		"	eor v2.b[x3], v3.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v3_v0:							\n"	\
//		"	eor v3.b[x3], v0.b[x4]					\n"	\
//		"	eor v0.b[x4], v3.b[x3]					\n"	\
//		"	eor v3.b[x3], v0.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v3_v1:							\n"	\
//		"	eor v3.b[x3], v1.b[x4]					\n"	\
//		"	eor v1.b[x4], v3.b[x3]					\n"	\
//		"	eor v3.b[x3], v1.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v3_v2:							\n"	\
//		"	eor v3.b[x3], v2.b[x4]					\n"	\
//		"	eor v2.b[x4], v3.b[x3]					\n"	\
//		"	eor v3.b[x3], v2.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"swap_v3_v3:							\n"	\
//		"	eor v3.b[x3], v3.b[x4]					\n"	\
//		"	eor v3.b[x4], v3.b[x3]					\n"	\
//		"	eor v3.b[x3], v3.b[x4]					\n"	\
//		"	b random_permutation_continue_loop			\n"	\
//											\
//		"								\n"	\
//		"random_permutation_skip_jump_table:				\n"	\
//		)
//
//static void random_permutation(void) {
//	uint64_t i = 0;
//
//	RANDOM_PERMUTATION("x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "w1", "w2");
//
//	for(i = 0; i < 16; ++i) {
//		uint64_t local = 0;
//		asm volatile("mov %[local], v0.b[%[index]]": [local] "=r"(local) : [index] "r"(i) :);
//		module_err("%d", local);
//	}
//}

// STATICS
static int (*set_memory_x)(unsigned long, int)		=	NULL;
static int (*set_memory_nx)(unsigned long, int)		=	NULL;

static size_t __nocfi kprobe_trick(char* function_symbol) {
	size_t address = 0;
	struct kprobe kp = {.symbol_name = function_symbol};
	register_kprobe(&kp);
	address = (size_t)kp.addr;
	unregister_kprobe(&kp);
	return address;
}

static bool __nocfi load_set_memory_permissions_handling(void) {
	load_global_symbol(kallsyms_lookup_name_fn, set_memory_t, set_memory_x, set_memory_x);
	load_global_symbol(kallsyms_lookup_name_fn, set_memory_t, set_memory_nx, set_memory_nx);
	return set_memory_x && set_memory_nx;
}

static bool load_kallsyms_lookup_name(void) {
	load_global_symbol(kprobe_trick, kallsyms_lookup_name_t, kallsyms_lookup_name_fn, kallsyms_lookup_name);
	return kallsyms_lookup_name_fn;
}

static bool __nocfi load_globals(void) {
	bool success_kallsyms = load_kallsyms_lookup_name(); // must be the first initialization, as all other loads depend on this
	bool success_permisstion_handling = load_set_memory_permissions_handling();
	return success_kallsyms && success_permisstion_handling;
}


static int  __init executor_init(void) {
	int err = 0;

	if(!load_globals()) {
        module_err("Unable to load global kernel exports\n");
		err = -ENOENT;
		goto init_failed_execution;
	}

	err = initialize_executor(set_memory_x);
	if(err) {
	    module_err("Unable to initialize executor\n");
	    goto init_failed_execution;
	}

	err = initialize_sysfs();
	if(err) {
        module_err("Unable to initialize sysfs\n");
		goto init_cleanup_executor;
	}

	err = initialize_device_interface();
	if(err) {
        module_err("Unable to initialize character device interface (error code: %d)\n", err);
		goto init_cleanup_sysfs;
	}

	return 0;

init_cleanup_sysfs:
    free_sysfs();
init_cleanup_executor:
    free_executor(set_memory_nx);
init_failed_execution:
	return err;
}

static void __nocfi __exit executor_exit(void) {
	module_err("executor is unloaded.\n");
	if(executor.tracing_error) {
		module_err("Failed to unload the module due to corrupted state\n");
		return;
	}

	free_device_interface();

	free_sysfs();

	free_executor(set_memory_nx);
}

module_init(executor_init);
module_exit(executor_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Gal Kaptsenel");
MODULE_DESCRIPTION("AArch64 implementation of Revisor's executor");

