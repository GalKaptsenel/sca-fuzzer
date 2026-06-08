#include "main.h"

/* Page-aligned stride (no sub-page offset): views differ only in the upper VA
 * bits, so all views share the same base-predictor index. */
#define VIEW_STRIDE       (MAX_MEASUREMENT_CODE_SIZE)
#define VIEW_REGION_PAGES DIV_ROUND_UP((size_t)MAX_MEASUREMENT_VIEWS * VIEW_STRIDE, PAGE_SIZE)
#define CODE_PAGES        (MAX_MEASUREMENT_CODE_SIZE / PAGE_SIZE)  /* distinct phys pages */

static struct page* view_phys_pages[CODE_PAGES];   /* the single SHARED set of code pages */
static struct page** view_page_map = NULL;
static void* base_va = NULL;

/* All views alias the SAME physical pages: allocate one set of CODE_PAGES pages
 * and repeat them across the vmap region, so writing the test case once is
 * visible at every view VA. */
static int create_view_mappings(void** buffer, size_t view_count, set_memory_t set_memory_x) {
	int i = 0, ret = 0;

	for (i = 0; i < CODE_PAGES; ++i) {
		view_phys_pages[i] = alloc_page(GFP_KERNEL);
		if (NULL == view_phys_pages[i]) {
			module_err("alloc_page[%d] failed", i);
			ret = -ENOMEM;
			goto free_phys;
		}
		clear_highpage(view_phys_pages[i]);
	}

	view_page_map = kmalloc_array(VIEW_REGION_PAGES, sizeof(*view_page_map), GFP_KERNEL);
	if (NULL == view_page_map) {
		module_err("kmalloc_array for view_page_map failed");
		ret = -ENOMEM;
		goto free_phys;
	}
	for (int j = 0; j < VIEW_REGION_PAGES; ++j) {
		view_page_map[j] = view_phys_pages[j % CODE_PAGES];   /* repeat -> aliasing */
	}

	base_va = vmap(view_page_map, VIEW_REGION_PAGES, VM_MAP | VM_ALLOC, PAGE_KERNEL_EXEC);
	if (NULL == base_va) {
		module_err("vmap failed");
		ret = -ENOMEM;
		goto free_page_map;
	}

	ret = set_memory_x((unsigned long)base_va, VIEW_REGION_PAGES);
	if (0 != ret) {
		module_err("set_memory_x failed (errcode: %d)", ret);
		goto unmap;
	}

	for (int j = 0; j < (int)view_count; ++j) {
		buffer[j] = (char*)base_va + (size_t)j * VIEW_STRIDE;
	}

	return 0;

unmap:
	vunmap(base_va);
	base_va = NULL;
free_page_map:
	kfree(view_page_map);
	view_page_map = NULL;
free_phys:
	while (0 <= --i) {
		__free_page(view_phys_pages[i]);
		view_phys_pages[i] = NULL;
	}
	return ret;
}

static void destroy_view_mappings(void** buffer, size_t view_count, set_memory_t set_memory_nx) {
	if (NULL != base_va) {
		set_memory_nx((unsigned long)base_va, VIEW_REGION_PAGES);
		vunmap(base_va);
		base_va = NULL;
	}
	kfree(view_page_map);
	view_page_map = NULL;
	for (int i = 0; i < CODE_PAGES; ++i) {
		if (NULL != view_phys_pages[i]) {
			__free_page(view_phys_pages[i]);
			view_phys_pages[i] = NULL;
		}
	}
	for (int j = 0; j < (int)view_count; ++j) {
		buffer[j] = NULL;
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
	enable_mte_tag_checking();

	init_executor_defaults();

	executor.test_case = kmalloc(MAX_TEST_CASE_SIZE, GFP_KERNEL);
	if (NULL == executor.test_case) {
		module_err("Could not allocate memory for test case\n");
		err = -ENOMEM;
		goto executor_init_failed_execution;
	}
	executor.test_case_length = 0;

	memset(executor.measurement_code_views, 0, sizeof(executor.measurement_code_views));
	err = create_view_mappings(executor.measurement_code_views, MAX_MEASUREMENT_VIEWS, set_memory_x);
	if (0 != err) {
		module_err("Failed to create view mappings (errcode: %d)\n", err);
		goto executor_init_cleanup_free_test_case;
	}

	refresh_tc_insert_offsets();

	initialize_inputs_db();

	executor.tracing_error = 0;
	executor.state = CONFIGURATION_STATE;
	executor.checkout_region = REGION_DEFAULT;

	initialize_sandbox(&executor.sandbox);

	return 0;

executor_init_cleanup_free_test_case:
	kfree(executor.test_case);
	executor.test_case = NULL;

executor_init_failed_execution:
	return err;
}

void __nocfi free_executor(set_memory_t set_memory_nx) {

	destroy_inputs_db();

	if (NULL != executor.test_case) {
		kfree(executor.test_case);
		executor.test_case = NULL;
	}

	destroy_view_mappings(executor.measurement_code_views, MAX_MEASUREMENT_VIEWS, set_memory_nx);
}
