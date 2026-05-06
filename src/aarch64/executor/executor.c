#include "main.h"

/* Each view needs MAX_MEASUREMENT_CODE_SIZE bytes of usable space.
 * Spacing views by (MAX_MEASUREMENT_CODE_SIZE + 4) ensures that
 * bits [13:2] of the start PC differ by 2 per view:
 *   (VIEW_STRIDE >> 2) & 0xFFF = (0x4008 >> 2) & 0xFFF = 0x002
 * so views 0..15 get base-predictor index deltas 0,2,4,...,30 — all
 * distinct — preventing the aliasing that drove misprediction counts
 * to zero after the first measurement. */
#define VIEW_STRIDE       (MAX_MEASUREMENT_CODE_SIZE + 8)
#define VIEW_REGION_PAGES DIV_ROUND_UP((size_t)MAX_MEASUREMENT_VIEWS * VIEW_STRIDE, PAGE_SIZE)

static struct page* view_phys_pages[VIEW_REGION_PAGES];
static struct page** view_page_map = NULL;
static void* base_va = NULL;

static int create_independent_view_mappings(void** buffer, size_t view_count, set_memory_t set_memory_x) {
	int i = 0, ret = 0;

	for (i = 0; i < VIEW_REGION_PAGES; ++i) {
		view_phys_pages[i] = alloc_page(GFP_KERNEL);
		if (!view_phys_pages[i]) {
			module_err("alloc_page[%d] failed", i);
			ret = -ENOMEM;
			goto free_phys;
		}
		clear_highpage(view_phys_pages[i]);
	}

	view_page_map = kmalloc_array(VIEW_REGION_PAGES, sizeof(*view_page_map), GFP_KERNEL);
	if (!view_page_map) {
		module_err("kmalloc_array for view_page_map failed");
		ret = -ENOMEM;
		goto free_phys;
	}
	for (int j = 0; j < VIEW_REGION_PAGES; ++j)
		view_page_map[j] = view_phys_pages[j];

	base_va = vmap(view_page_map, VIEW_REGION_PAGES, VM_MAP | VM_ALLOC, PAGE_KERNEL_EXEC);
	if (!base_va) {
		module_err("vmap failed");
		ret = -ENOMEM;
		goto free_page_map;
	}

	ret = set_memory_x((unsigned long)base_va, VIEW_REGION_PAGES);
	if (ret) {
		module_err("set_memory_x failed (errcode: %d)", ret);
		goto unmap;
	}

	for (int j = 0; j < (int)view_count; ++j)
		buffer[j] = (char*)base_va + (size_t)j * VIEW_STRIDE;

	return 0;

unmap:
	vunmap(base_va);
	base_va = NULL;
free_page_map:
	kfree(view_page_map);
	view_page_map = NULL;
free_phys:
	while (--i >= 0) {
		__free_page(view_phys_pages[i]);
		view_phys_pages[i] = NULL;
	}
	return ret;
}

static void destroy_independent_view_mappings(void** buffer, size_t view_count, set_memory_t set_memory_nx) {
	if (base_va) {
		set_memory_nx((unsigned long)base_va, VIEW_REGION_PAGES);
		vunmap(base_va);
		base_va = NULL;
	}
	kfree(view_page_map);
	view_page_map = NULL;
	for (int i = 0; i < VIEW_REGION_PAGES; ++i) {
		if (view_phys_pages[i]) {
			__free_page(view_phys_pages[i]);
			view_phys_pages[i] = NULL;
		}
	}
	for (int j = 0; j < (int)view_count; ++j)
		buffer[j] = NULL;
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

	executor.mistraining_code = kmalloc(MAX_TEST_CASE_SIZE, GFP_KERNEL);
	if (NULL == executor.mistraining_code) {
		module_err("Could not allocate memory for mistraining code\n");
		err = -ENOMEM;
		goto executor_init_cleanup_free_test_case;
	}
	executor.mistraining_code_length = 0;

	memset(executor.measurement_code_views, 0, sizeof(executor.measurement_code_views));
	err = create_independent_view_mappings(executor.measurement_code_views, MAX_MEASUREMENT_VIEWS, set_memory_x);
	if (err) {
		module_err("Failed to create independent view mappings (errcode: %d)\n", err);
		goto executor_init_cleanup_free_test_case;
	}

	initialize_inputs_db();

	executor.tracing_error = 0;
	executor.state = CONFIGURATION_STATE;
	executor.checkout_region = TEST_REGION;

	initialize_sandbox(&executor.sandbox);

	return 0;

executor_init_cleanup_free_test_case:
	kfree(executor.test_case);
	executor.test_case = NULL;

executor_init_failed_execution:
	return err;
}
EXPORT_SYMBOL(initialize_executor);

void __nocfi free_executor(set_memory_t set_memory_nx) {

	destroy_inputs_db();

	if (executor.test_case) {
		kfree(executor.test_case);
		executor.test_case = NULL;
	}

	if (executor.mistraining_code) {
		kfree(executor.mistraining_code);
		executor.mistraining_code = NULL;
	}

	destroy_independent_view_mappings(executor.measurement_code_views, MAX_MEASUREMENT_VIEWS, set_memory_nx);
}
EXPORT_SYMBOL(free_executor);
