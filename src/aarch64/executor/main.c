/// File: Kernel module interface
///
// Copyright (C) Microsoft Corporation
// SPDX-License-Identifier: MIT

// clang-format off

// INCLUDES
//
#include "main.h"

// STATICS
int (*set_memory_x_fn)(unsigned long, int)		=	NULL;
int (*set_memory_nx_fn)(unsigned long, int)		=	NULL;
int (*set_memory_rw_fn)(unsigned long, int)		=	NULL;
int (*set_memory_ro_fn)(unsigned long, int)		=	NULL;

typedef void (*set_mte_ctrl_t)(struct task_struct *task, unsigned long ctrl);
typedef void (*mte_sync_tags_t)(pte_t addr, int size);
void (*set_mte_ctrl_fn)(struct task_struct *task, unsigned long ctrl) = NULL;
void (*mte_sync_tags_fn)(pte_t addr, int size) = NULL;

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
	load_global_symbol(kallsyms_lookup_name_fn, set_memory_t, set_memory_x_fn, set_memory_x);
	load_global_symbol(kallsyms_lookup_name_fn, set_memory_t, set_memory_nx_fn, set_memory_nx);
	load_global_symbol(kallsyms_lookup_name_fn, set_memory_t, set_memory_rw_fn, set_memory_rw);
	load_global_symbol(kallsyms_lookup_name_fn, set_memory_t, set_memory_ro_fn, set_memory_ro);
#else
	set_memory_x_fn = set_memory_x;
	set_memory_nx_fn = set_memory_nx;
	set_memory_rw_fn = set_memory_rw;
	set_memory_ro_fn = set_memory_ro;
#endif
	return set_memory_x_fn && set_memory_nx_fn && set_memory_rw_fn && set_memory_ro_fn;
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

static int  __init executor_init(void) {
	int err = 0;

	module_info("Starting running executor kernel module init\n");

	if(!load_globals()) {
		module_err("Unable to load global kernel exports\n");
		err = -ENOENT;
		goto init_failed_execution;
	}

	err = initialize_executor(set_memory_x_fn);
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
	
	module_err("Loaded Successfully\n");
//	module_err("mte_ext() => %d", mte_ext());
//
//	pr_info("[MTE POC] init\n");
//
//	set_mte_ctrl_fn(current, 1 /* ENABLE */ );
//
//	struct page* page = alloc_page(GFP_KERNEL);
//	if (!page) {
//		return -ENOMEM;
//	}
//	void* addr = page_address(page);
//
//	pr_info("[MTE POC] page=%px addr=%px\n", page, addr);
//
//	pte_t pte = mk_pte(page, PAGE_KERNEL);
//	mte_sync_tags_fn(pte, 1);
//
//	// tag_region(buf, PAGE_SIZE, 0xA);
//
//	void* good = tag_ptr_(addr, 0xA);
//	void* bad  = tag_ptr_(addr, 0xB);
//
//	*(volatile int *)good = 42;
//	pr_info("[MTE POC] good write OK\n");
//
//	pr_info("[MTE POC] about to trigger MTE fault...\n");
//
//	*(volatile int *)bad = 123;
//
//	pr_info("[MTE POC] ERROR: no fault triggered\n");

//	trigger_pauth_fault();

	return 0;

init_cleanup_sysfs:
    free_sysfs();
init_cleanup_executor:
    free_executor(set_memory_nx_fn);
init_failed_execution:
	return err;
}

static void __nocfi __exit executor_exit(void) {
	module_info("executor is being unloaded.\n");

	if(executor.tracing_error) {
		module_err("Failed to unload the module due to corrupted state\n");
		return;
	}

	free_device_interface();

	free_sysfs();

	free_executor(set_memory_nx_fn);
}


MODULE_LICENSE("GPL");

module_init(executor_init);
module_exit(executor_exit);

