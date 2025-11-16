/// File: Kernel module interface
///
// Copyright (C) Microsoft Corporation
// SPDX-License-Identifier: MIT

// clang-format off

// INCLUDES
//
#include "main.h"

// STATICS
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 4, 0)
static int (*set_memory_x)(unsigned long, int)		=	NULL;
static int (*set_memory_nx)(unsigned long, int)		=	NULL;
#endif

static size_t __nocfi kprobe_trick(char* function_symbol) {
	size_t address = 0;
	struct kprobe kp = {.symbol_name = function_symbol};
	register_kprobe(&kp);
	address = (size_t)kp.addr;
	unregister_kprobe(&kp);
	return address;
}

static bool __nocfi load_set_memory_permissions_handling(void) {
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 4, 0)
	load_global_symbol(kallsyms_lookup_name_fn, set_memory_t, set_memory_x, set_memory_x);
	load_global_symbol(kallsyms_lookup_name_fn, set_memory_t, set_memory_nx, set_memory_nx);
#endif
	return set_memory_x && set_memory_nx;
}

static bool load_kallsyms_lookup_name(void) {
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 7, 0)
	load_global_symbol(kprobe_trick, kallsyms_lookup_name_t, kallsyms_lookup_name_fn, kallsyms_lookup_name);
#else
	kallsyms_lookup_name_fn = kallsyms_lookup_name;
#endif
	return kallsyms_lookup_name_fn;
}

static bool __nocfi load_globals(void) {
	bool success_kallsyms = load_kallsyms_lookup_name(); // must be the first initialization, as all other loads depend on this
	bool success_permisstion_handling = load_set_memory_permissions_handling();
	return success_kallsyms && success_permisstion_handling;
}

static bool cpu_has_ssbs(void)
{
    return cpus_have_final_cap(ARM64_SSBS);
}

static bool is_ssbs_enabled(void)
{
    unsigned long pstate = 0;

    asm volatile("mrs %0, spsr_el1" : "=r"(pstate));
    /* Bit 12 = SSBS */
    return (pstate & BIT(12)) != 0;
}

static void check_ssb_state(void)
{
    bool has_ssbs = cpu_has_ssbs();
    bool ssbs_enabled = is_ssbs_enabled();

    module_info("[SSB] CPU has SSBS feature: %s\n",
            has_ssbs ? "YES" : "NO");

    module_info("[SSB] Current PSTATE.SSBS bit: %s\n",
            ssbs_enabled ? "1 (Speculative Store Bypass ENABLED – VULNERABLE)"
                        : "0 (Speculative Store Bypass DISABLED – MITIGATED)");
}

static int  __init executor_init(void) {
	int err = 0;

	module_info("Starting running executor kernel module init\n");
	check_ssb_state();
	on_each_cpu((void(*)(void *))arm64_module_enable_ssbs, NULL, 1);

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
	
	check_ssb_state();
	module_err("Loaded Successfully\n");
	return 0;

init_cleanup_sysfs:
    free_sysfs();
init_cleanup_executor:
    free_executor(set_memory_nx);
init_failed_execution:
	return err;
}

static void __nocfi __exit executor_exit(void) {
	module_info("executor is being unloaded.\n");

	check_ssb_state();
	on_each_cpu((void(*)(void *))arm64_module_restore_ssbs, NULL, 1);

	if(executor.tracing_error) {
		module_err("Failed to unload the module due to corrupted state\n");
		return;
	}

	free_device_interface();

	free_sysfs();

	free_executor(set_memory_nx);

	check_ssb_state();
}

module_init(executor_init);
module_exit(executor_exit);

