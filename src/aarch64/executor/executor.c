#include "main.h"

static struct page** pages = NULL;
static struct page* phys_pages[MAX_MEASUREMENT_CODE_SIZE / PAGE_SIZE] = { 0 };
static void* base_va = NULL;
static int create_aliasing_mappings(void** buffer, size_t aliasing_count, set_memory_t set_memory_x) {
	int i = 0, ret = 0;

	size_t num_phys_pages = sizeof(phys_pages) / sizeof(phys_pages[0]);
	for(i = 0; i < num_phys_pages; ++i) {
		phys_pages[i] = alloc_page(GFP_KERNEL);
		if(NULL == phys_pages[i]) {
			module_err("alloc_page[%d] failes", i);
			ret = -ENOMEM;
			goto create_aliasing_mappings_free_phys;
		}
		clear_highpage(phys_pages[i]);
	}

	pages = kmalloc_array(num_phys_pages * aliasing_count, sizeof(*pages), GFP_KERNEL);
	if(NULL == pages) {
		module_err("kamlloc_array failed");
		ret = -ENOMEM;
		goto create_aliasing_mappings_free_phys;
	}

	for(int j = 0; j < num_phys_pages * aliasing_count; ++j) {
		pages[j] = phys_pages[j % num_phys_pages];
	}

	base_va = vmap(pages, num_phys_pages * aliasing_count,VM_MAP| VM_ALLOC, PAGE_KERNEL_EXEC);
	if(NULL == base_va) {
		module_err("vmap failed");
		ret = -ENOMEM;
		goto create_aliasing_mappings_free_pages_array;
	}
	
	for(int j = 0; j < aliasing_count; ++j) {
		buffer[j] = (char*)base_va + MAX_MEASUREMENT_CODE_SIZE * j;
	}

	return ret;

create_aliasing_mappings_free_pages_array:
	kfree(pages);
	pages = NULL;

create_aliasing_mappings_free_phys:
	while(--i >= 0) {
		__free_page(phys_pages[i]);
		phys_pages[i] = NULL;
	}

	return ret;
}

static void destroy_aliasing_mappings(void** buffer, size_t aliasing_count) {
	size_t num_phys_pages = sizeof(phys_pages) / sizeof(phys_pages[0]);
	if(NULL != base_va) {
		vfree(base_va);
		base_va = NULL;
	}

	kfree(pages);
	pages = NULL;

	for(int i = 0; i < num_phys_pages; ++i) {
		if(NULL != phys_pages[i]) {
			__free_page(phys_pages[i]);
			phys_pages[i] = NULL;
		}
	}

	for(int i = 0; i < aliasing_count; ++i) {
		buffer[i] = NULL;
	}
}

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

	memset(executor.measurement_code_views, 0, sizeof(executor.measurement_code_views));
	err = create_aliasing_mappings(executor.measurement_code_views, MAX_MEASUREMENT_VIEWS, set_memory_x);
	if(err) {
		module_err("Failed to create_aliasing_mappings(errcode: %d)\n", err);
		goto executor_init_cleanup_free_test_case;
	}

	size_t num_phys_pages = sizeof(phys_pages) / sizeof(phys_pages[0]);
	err = set_memory_x((unsigned long)base_va, num_phys_pages * MAX_MEASUREMENT_VIEWS);
	
	if(err) {
		module_err("Failed to make executor measurement views executable (errcode: %d)\n", err);
		goto executor_init_unmap_views;
	}

	initialize_inputs_db();

	executor.tracing_error = 0;
	executor.state = CONFIGURATION_STATE;
	executor.checkout_region = TEST_REGION;

	initialize_sandbox(&executor.sandbox);

	return 0;

executor_init_unmap_views:
	destroy_aliasing_mappings(executor.measurement_code_views, MAX_MEASUREMENT_VIEWS);

executor_init_cleanup_free_test_case:
	kfree(executor.test_case);
	executor.test_case = NULL;

executor_init_failed_execution:
	return err;
}
EXPORT_SYMBOL(initialize_executor);

void __nocfi free_executor(set_memory_t set_memory_nx) {

	destroy_inputs_db();

	set_memory_nx((unsigned long)base_va, (sizeof(phys_pages) / sizeof(phys_pages[0])) * MAX_MEASUREMENT_VIEWS);

	if (executor.test_case) {
		kfree(executor.test_case);
		executor.test_case = NULL;
	}

	destroy_aliasing_mappings(executor.measurement_code_views, MAX_MEASUREMENT_VIEWS);
}
EXPORT_SYMBOL(free_executor);
