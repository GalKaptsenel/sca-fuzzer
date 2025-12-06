#include "main.h"

static void init_executor_defaults(void) {
	executor.config.uarch_reset_rounds = UARCH_RESET_ROUNDS_DEFAULT;
	executor.config.enable_faulty_page = ENABLE_FAULTY_DEFAULT;
	executor.config.pre_run_flush = PRE_RUN_FLUSH_DEFAULT;
	executor.config.measurement_template = MEASUREMENT_TEMPLATE_DEFAULT;
	executor.config.pinned_cpu_id = CPU_ID_DEFAULT;
}

int __nocfi initialize_executor(set_memory_t set_memory_x) {
	int err = 0;

	init_executor_defaults();

	executor.test_case = kmalloc(MAX_TEST_CASE_SIZE, GFP_KERNEL);
	if (NULL == executor.test_case) {
        	module_err("Could not allocate memory for test case\n");
        	err = -ENOMEM;
		goto executor_init_failed_execution;
	}

	executor.test_case_length = 0;

	err = set_memory_x(
			(unsigned long)executor.measurement_code,
			sizeof(executor.measurement_code) / PAGESIZE
		);

	if(err) {
		module_err("Failed to make executor.measurement_code executable (errcode: %d)\n", err);
		goto executor_init_cleanup_free_test_case;
	}

	executor.sandbox_scratchpad_size = SANDBOX_SCRATCHPAD_MEMORY_SIZE;
	executor.sandbox_scratchpad_memory = kmalloc(executor.sandbox_scratchpad_size, GFP_KERNEL);
	if(NULL == executor.sandbox_scratchpad_memory) {
        	module_err("Could not allocate memory for scratchpad\n");
		err = -ENOMEM;
		goto executor_init_cleanup_free_test_case;
	}

	initialize_inputs_db();

	executor.tracing_error = 0;
	executor.state = CONFIGURATION_STATE;
	executor.checkout_region = TEST_REGION;

	initialize_sandbox(&executor.sandbox);

	return 0;

executor_init_cleanup_free_test_case:
	executor.sandbox_scratchpad_size = 0;
	kfree(executor.test_case);
	executor.test_case = NULL;

executor_init_failed_execution:
	return err;
}
EXPORT_SYMBOL(initialize_executor);

void __nocfi free_executor(set_memory_t set_memory_nx) {

	destroy_inputs_db();

	set_memory_nx((unsigned long)executor.measurement_code,
			sizeof(executor.measurement_code) / PAGESIZE
		);

	if (executor.test_case) {
		kfree(executor.test_case);
		executor.test_case = NULL;
	}


	if (executor.sandbox_scratchpad_memory) {
		kfree(executor.sandbox_scratchpad_memory);
		executor.sandbox_scratchpad_size = 0;
		executor.sandbox_scratchpad_memory = NULL;
	}
}
EXPORT_SYMBOL(free_executor);
