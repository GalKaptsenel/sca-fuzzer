/// File: Kernel module interface
///

// clang-format off

// INCLUDES
//
#include "main.h"

// STATICS
int (*set_memory_x_fn)(unsigned long, int)		=	NULL;
int (*set_memory_nx_fn)(unsigned long, int)		=	NULL;
int (*set_memory_rw_fn)(unsigned long, int)		=	NULL;
int (*set_memory_ro_fn)(unsigned long, int)		=	NULL;

static size_t __nocfi kprobe_trick(char* function_symbol) {
	struct kprobe kp = {.symbol_name = function_symbol};
	if (0 != register_kprobe(&kp)) {
		return 0;
	}
	size_t address = (size_t)kp.addr;
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
	// every other load resolves symbols via kallsyms_lookup_name; don't run them with a NULL resolver
	if (!load_kallsyms_lookup_name()) {
		return false;
	}
	return load_set_memory_permissions_handling();
}

#if CONFIG_ARM64_MTE_HW

static DEFINE_PER_CPU(struct mte_control_state, mte_saved_control);
static DEFINE_PER_CPU(bool, mte_control_saved);

static void mte_configure_this_cpu(void* arg) {
	(void)arg;
	mte_save_control(this_cpu_ptr(&mte_saved_control));
	*this_cpu_ptr(&mte_control_saved) = true;

	disable_TCO_bit();
	mte_set_sync();
	enable_TCMA1_bit();
}

static void mte_restore_this_cpu(void* arg) {
	(void)arg;
	if (*this_cpu_ptr(&mte_control_saved)) {
		mte_restore_control(this_cpu_ptr(&mte_saved_control));
		*this_cpu_ptr(&mte_control_saved) = false;
	}
}

static void mte_configure_all_cpus(void) {
	int cpu;
	for_each_online_cpu(cpu) {
		int err = execute_on_pinned_cpu(cpu, mte_configure_this_cpu, NULL);
		if (0 != err) {
			module_err("MTE: failed to configure CPU %d (err=%d)\n", cpu, err);
		}
	}
}

static void mte_restore_all_cpus(void) {
	int cpu;
	for_each_online_cpu(cpu) {
		execute_on_pinned_cpu(cpu, mte_restore_this_cpu, NULL);
	}
}

#endif // CONFIG_ARM64_MTE_HW

static int  __init executor_init(void) {
	int err = 0;

	module_info("Starting running executor kernel module init\n");

	if(!load_globals()) {
		module_err("Unable to load global kernel exports\n");
		err = -ENOENT;
		goto init_failed_execution;
	}

#if CONFIG_ARM64_MTE_HW
	mte_configure_all_cpus();
#endif

	err = initialize_executor(set_memory_x_fn);
	if (0 != err) {
	    module_err("Unable to initialize executor\n");
	    goto init_cleanup_mte;
	}

	err = initialize_sysfs();
	if (0 != err) {
		module_err("Unable to initialize sysfs\n");
		goto init_cleanup_executor;
	}

	err = initialize_device_interface();
	if (0 != err) {
		module_err("Unable to initialize character device interface (error code: %d)\n", err);
		goto init_cleanup_sysfs;
	}
	
	module_info("Loaded Successfully\n");

	return 0;

init_cleanup_sysfs:
    free_sysfs();
init_cleanup_executor:
    free_executor(set_memory_nx_fn);
init_cleanup_mte:
#if CONFIG_ARM64_MTE_HW
	mte_restore_all_cpus();
#endif
init_failed_execution:
	return err;
}

static void __nocfi __exit executor_exit(void) {
	module_info("executor is being unloaded.\n");

	if (0 != executor.tracing_error) {
		module_err("Failed to unload the module due to corrupted state (tracing_error=%d)\n",
			   executor.tracing_error);
		/* __exit cannot abort unload; fall through and free or we leak the
		 * cdev/class/sysfs/test_case and the RWX vmap view region. */
	}

	free_device_interface();

	free_sysfs();

	free_executor(set_memory_nx_fn);

#if CONFIG_ARM64_MTE_HW
	mte_restore_all_cpus();
#endif
}


MODULE_LICENSE("GPL");
MODULE_AUTHOR("ACSL - Gal Kaptsenel");
MODULE_DESCRIPTION("AArch64 implementation of Revisor's executor");

module_init(executor_init);
module_exit(executor_exit);

