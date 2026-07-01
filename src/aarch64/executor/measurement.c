#include "main.h"
#include "pmu.h"

static inline int setup_environment(void) {
    int err = config_pfc();
    if (0 != err) {
        return err;
    }

    // TBD: configure faulty page
    return 0;
}

static void load_memory_from_input(input_t* input) {

	// - sandbox: main and faulty regions
	for (int j = 0; j < (sizeof(executor.sandbox->main_region) / sizeof(uint64_t)); ++j) {
	        ((uint64_t*)executor.sandbox->main_region)[j] = ((uint64_t*)(input->main_region))[j];
	}

	for (int j = 0; j < (sizeof(executor.sandbox->faulty_region) / sizeof(uint64_t)); ++j) {
	        ((uint64_t*)executor.sandbox->faulty_region)[j] = ((uint64_t*)(input->faulty_region))[j];
	}
}

// RSP must be aligned to 16 bytes boundary, according to documentation of AARCH64
static size_t get_stack_base_address(void) {
	size_t address = ((size_t)executor.sandbox->main_region + sizeof(executor.sandbox->main_region));
	return PTR_ALIGN(address, 16); // Technically, kernel stack should be aligned to THREAD_SIZE, for example it allows access the thread_indo structure. But it is fine to just align to 16 bytes, due to hardware only checks this constraint.
}

static void load_registers_from_input(input_t* input) {

	// Initial register values
	*((registers_t*)executor.sandbox->lower_overflow) = input->regs;

	// flags is already in ARM PSTATE format (N=bit31 Z=bit30 C=bit29 V=bit28)
	// from _reconstruct_pstate() in Python; msr nzcv reads bits [31:28] directly.

	// - RSP and RBP
	((registers_t*)executor.sandbox->lower_overflow)->sp = get_stack_base_address();

//	module_debug("Input regs: x0:%llx, x1:%llx, x2:%llx x3:%llx, x4:%llx, x5:%llx, flags:%llx, sp:%llx\n",
//			*(uint64_t*)executor.sandbox->lower_overflow,
//			*((uint64_t*)executor.sandbox->lower_overflow+1),
//			*((uint64_t*)executor.sandbox->lower_overflow+2),
//			*((uint64_t*)executor.sandbox->lower_overflow+3),
//			*((uint64_t*)executor.sandbox->lower_overflow+4),
//			*((uint64_t*)executor.sandbox->lower_overflow+5),
//			*((uint64_t*)executor.sandbox->lower_overflow+6),
//			*((uint64_t*)executor.sandbox->lower_overflow+7));
}

static void load_input_to_sandbox(input_t* input) {
	load_memory_from_input(input);
	load_registers_from_input(input);

	// Per-input MTE tags override the setup tagging over the contiguous main|faulty span. When an
	// input carries no tags, reset the span to the default so it never inherits a prior input's tags.
	if (input->mte_tags_present) {
		mte_apply_sandbox_tags(executor.sandbox->main_region, input->mte_tags,
		                       INPUT_MTE_TAG_COUNT);
	} else {
		mte_init_sandbox_tags(executor.sandbox->main_region, MEMORY_INPUT_SIZE, MTE_INITIAL_TAG);
	}
}

static void initialize_overflow_pages(void) {

	// Initialize memory:
	// NOTE: memset is not used intentionally! somehow, it messes up with P+P measurements
	// - overflows are initialized with zeroes
	memset(executor.sandbox->lower_overflow, 0, sizeof(executor.sandbox->lower_overflow));
	memset(executor.sandbox->upper_overflow, 0, sizeof(executor.sandbox->upper_overflow));
//	for (int j = 0; j < (sizeof(executor.sandbox->upper_overflow) / sizeof(uint64_t)); ++j) {
//	    ((uint64_t *)executor.sandbox->upper_overflow)[j] = 0;
//	}
}

int64_t initialize_measurement(measurement_t* measurement) {
	if (NULL == measurement) {
		return -EINVAL;
	}
	memset(measurement, 0, sizeof(measurement_t));
	return 0;
}


void free_measurement(measurement_t* measurement) {
	if (NULL == measurement) {
		return;
	}
}

static void measure(measurement_t* measurement) {
	if (NULL == measurement) {
		return;
	}

	for(size_t i = 0; i < HTRACE_WIDTH; ++i) {
		measurement->htrace[i] = executor.sandbox->latest_measurement.htrace[i];
	}
	
	for(size_t i = 0; i < NUM_PFC; ++i) {
		measurement->pfc[i] = executor.sandbox->latest_measurement.pfc[i];
	}
}

static int __nocfi run_experiments(void) {
	int64_t rounds = (int64_t)executor.number_of_inputs;
	unsigned long flags = 0;
	struct rb_node* current_input_node = NULL;

	if(0 >= executor.number_of_inputs){
		BUG_ON(0 > executor.number_of_inputs);
		module_err("No inputs were set!\n");
		return -EINVAL;
	}

	current_input_node = rb_first(&executor.inputs_root);
	BUG_ON(NULL == current_input_node);

	// Zero-initialize the region of memory used by Prime+Probe
	memset(executor.sandbox->eviction_region, 0, sizeof(executor.sandbox->eviction_region));

	// S3_3_C4_C2_6 is the SSBS register (mrs/msr ssbs); SSBS is PSTATE bit 12.
	uint64_t saved_ssbs = 0;
	bool ssbs_changed = false;
	if (executor.config.enable_ssbs) {
		asm volatile("mrs %0, s3_3_c4_c2_6" : "=r"(saved_ssbs));
		if (0 == (saved_ssbs & (1ULL << 12))) {
			asm volatile("msr s3_3_c4_c2_6, %0\n isb\n" :: "r"(saved_ssbs | (1ULL << 12)) : "memory");
			ssbs_changed = true;
		}
	}

	for (int64_t i = -executor.config.uarch_reset_rounds; i < rounds; ++i) {
		struct input_node* current_input = NULL;

		// ignore "warm-up" runs (i<0)
		if (0 < i) {
			current_input_node = rb_next(current_input_node);
			BUG_ON(NULL == current_input_node);
		}

		current_input = rb_entry(current_input_node, struct input_node, node);
		initialize_overflow_pages();
		load_input_to_sandbox(&current_input->input);

		raw_local_irq_save(flags);

		void* measurement_code = executor.measurement_code_views[0];

		/* Three independent knobs (decoupled from the legacy pre_run_flush):
		 *   view_rotation : serve the next view (invalidate_bpu_entries) -> tagged-table miss
		 *   branch_training: re-apply the mistraining config
		 *   phr_flush     : overwrite the branch-history register before the run */
		if (executor.config.view_rotation) {
			measurement_code = invalidate_bpu_entries();
		}
		if (executor.config.enable_branch_training) {
			reapply_branch_training(measurement_code);
		}
		if (executor.config.phr_flush) {
			flush_bpu_phr();
		}
		config_pfc();

		struct pac_keys saved_hw_keys;
		uint64_t saved_sctlr = 0;
		// PAC key precedence: per-input keys > ioctl-swapped keys > live hardware keys.
		bool use_exec_keys = current_input->input.pac_keys_present ||
		                     executor.config.pac_keys_set;
		const struct pac_keys* exec_keys = current_input->input.pac_keys_present
			? &current_input->input.pac_keys : &executor.config.pac_keys;
		if (use_exec_keys) {
			pac_save_keys(&saved_hw_keys);
			pac_load_keys(exec_keys);
			saved_sctlr = pac_enable_all_keys();
		}

		// execute
		((void(*)(void*))measurement_code)(executor.sandbox);

		if (use_exec_keys) {
			pac_restore_sctlr(saved_sctlr);
			pac_load_keys(&saved_hw_keys);
		}

		raw_local_irq_restore(flags);

		measure(&current_input->measurement);
	}

	if (ssbs_changed) {
		asm volatile("msr s3_3_c4_c2_6, %0\n isb\n" :: "r"(saved_ssbs) : "memory");
	}

	return 0;
}

int execute(void) {

    int err = setup_environment();
    if (0 != err) {
        return err;
    }

    return run_experiments();
}

