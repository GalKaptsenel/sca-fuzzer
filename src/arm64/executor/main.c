/// File: Kernel module interface
///
// Copyright (C) Microsoft Corporation
// SPDX-License-Identifier: MIT

// clang-format off

// INCLUDES
#include "main.h"

kallsyms_lookup_name_t kallsyms_lookup_name_fn	    =	NULL;

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

	err = initialize_executor(set_memory_x, set_memory_nx);
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
        module_err("Unable to initialize character device interface\n");
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

