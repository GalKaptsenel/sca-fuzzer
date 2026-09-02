#include "main.h"
#include "pmu.h"
#include <linux/random.h>

/* Fisher-Yates shuffle of executor.reload_order[0..63] -- a fresh random set-visit order for reload(). */
static void fill_reload_order(void) {
	int i = 0;
	for (i = 0; i < 64; ++i) {
		executor.reload_order[i] = i;
	}
	for (i = 63; i > 0; --i) {
		uint32_t j = get_random_u32() % (uint32_t)(i + 1);
		uint64_t t = executor.reload_order[i];
		executor.reload_order[i] = executor.reload_order[j];
		executor.reload_order[j] = t;
	}
}

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
		// A <=16B access from the sandbox's last byte spills into upper_overflow's first granule; give
		// that granule the last input tag so the boundary spill matches the in-bounds tag (else the
		// overflow's default tag mismatches -> an architectural MTE tag fault on an accidental spill).
		mte_apply_sandbox_tags(executor.sandbox->upper_overflow,
		                       &input->mte_tags[INPUT_MTE_TAG_COUNT - 1], 1);
	} else {
		mte_init_sandbox_tags(executor.sandbox->main_region, MEMORY_INPUT_SIZE, MTE_INITIAL_DEFAULT_TAG);
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

/* Code-relocation offsets index the test-case body; the bound needs the loaded test case, so it
 * is verified here at trace time (inputs and the test case load in any order). */
static int validate_code_relocations(void) {
	for (struct rb_node* node = rb_first(&executor.inputs_root); NULL != node; node = rb_next(node)) {
		const input_t* input = &rb_entry(node, struct input_node, node)->input;
		for (const struct revisor_code_reloc_entry* r = input->code_reloc;
		     REVISOR_CODE_RELOC_TERMINATOR != r->offset; ++r) {
			if (executor.test_case_length < (r->offset + sizeof(uint32_t))) {
				module_err("code relocation offset %u exceeds test-case length %zu\n",
				           r->offset, executor.test_case_length);
				return -EINVAL;
			}
		}
	}
	return 0;
}

/* Splice this input's relocations into the test-case body, or restore the body to the pristine test
 * case when `revert` is set. Bytes are written through view[0] (all views alias one set of physical
 * pages); the icache is VA-indexed, so it is invalidated only at `exec_view`, the VA that will run.
 * The pinned CPU both patches and runs the code, so a local cache maintenance (no SMP broadcast)
 * suffices. */
static void splice_code_relocations(void* exec_view, const input_t* input, bool revert) {
	const struct revisor_code_reloc_entry* relocs = input->code_reloc;
	if (REVISOR_CODE_RELOC_TERMINATOR == relocs->offset) {
		return;
	}

	size_t body = current_tc_insert_offset_bytes();
	char* write_body = (char*)executor.measurement_code_views[0] + body;
	char* exec_body  = (char*)exec_view + body;
	const char* pristine = executor.test_case;

	const struct revisor_code_reloc_entry* r;
	for (r = relocs; REVISOR_CODE_RELOC_TERMINATOR != r->offset; ++r) {
		uint32_t word = revert ? *(const uint32_t*)(pristine + r->offset) : r->value;
		*(uint32_t*)(write_body + r->offset) = word;
	}
	for (r = relocs; REVISOR_CODE_RELOC_TERMINATOR != r->offset; ++r) {
		asm volatile("dc cvau, %0" :: "r"(write_body + r->offset) : "memory");
	}
	asm volatile("dsb ish" ::: "memory");
	for (r = relocs; REVISOR_CODE_RELOC_TERMINATOR != r->offset; ++r) {
		asm volatile("ic ivau, %0" :: "r"(exec_body + r->offset) : "memory");
	}
	asm volatile("dsb ish\n isb" ::: "memory");
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

	int reloc_err = validate_code_relocations();
	if (0 != reloc_err) {
		return reloc_err;
	}

	current_input_node = rb_first(&executor.inputs_root);
	BUG_ON(NULL == current_input_node);

	// Zero-initialize the region of memory used by Prime+Probe
	memset(executor.sandbox->eviction_region, 0, sizeof(executor.sandbox->eviction_region));

	// S3_3_C4_C2_6 is the SSBS register (mrs/msr ssbs); SSBS is PSTATE bit 12. enable_ssbs=1 => SSBS=1
	// (speculative store bypass ALLOWED, mitigation off); enable_ssbs=0 => SSBS=0 (bypass disabled,
	// SSB mitigation ON). Set explicitly both ways and read back into debug_ssbs for verification.
	uint64_t saved_ssbs = 0;
	bool ssbs_changed = false;
	asm volatile("mrs %0, s3_3_c4_c2_6" : "=r"(saved_ssbs));
	uint64_t want_ssbs = executor.config.enable_ssbs ? (saved_ssbs | (1ULL << 12))
	                                                 : (saved_ssbs & ~(1ULL << 12));
	if (want_ssbs != saved_ssbs) {
		asm volatile("msr s3_3_c4_c2_6, %0\n isb\n" :: "r"(want_ssbs) : "memory");
		ssbs_changed = true;
	}
	asm volatile("mrs %0, s3_3_c4_c2_6" : "=r"(executor.debug_ssbs));

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

		if (executor.config.reload_random_order) {
			fill_reload_order();
		}

		raw_local_irq_save(flags);

		void* measurement_code = executor.measurement_code_views[0];

		/* Three independent knobs (decoupled from the legacy pre_run_flush):
		 *   view_rotation : serve the next view (invalidate_bpu_entries) -> tagged-table miss
		 *   branch_training: re-apply the mistraining config
		 *   phr_flush     : overwrite the branch-history register before the run */
		if (executor.config.view_rotation) {
			measurement_code = invalidate_bpu_entries();
		}
		splice_code_relocations(measurement_code, &current_input->input, false);
		apply_input_branch_training(measurement_code, &current_input->input);
		if (executor.config.phr_flush) {
			flush_bpu_phr();
		}
		config_pfc();

		struct pac_keys saved_hw_keys;
		uint64_t saved_sctlr = 0;
		bool use_exec_keys = current_input->input.pac_keys_present;
		if (use_exec_keys) {
			pac_save_keys(&saved_hw_keys);
			pac_load_keys(&current_input->input.pac_keys);
			saved_sctlr = pac_enable_all_keys();
		}

		// execute. Clear GCR_EL1.Exclude first so ADDG/IRG in the test case (and the seal's ADDG
		// retags) use plain modular tag arithmetic, matching the tag-blind contract model — the kernel
		// leaves reserved tags in Exclude, which would make a sealed retag land on a different tag.
		uint64_t saved_gcr = mte_gcr_clear_exclude();
		if (executor.config.reload_isolate) {
			/* Per-set isolation: re-run the (deterministic) test once per set, each probing only that
			 * set, and OR the single-set htraces so no reload sweep can prefetch the page into itself. */
			uint64_t acc = 0;
			for (executor.reload_target_set = 0; executor.reload_target_set < 64; ++executor.reload_target_set) {
				((void(*)(void*))measurement_code)(executor.sandbox);
				acc |= executor.sandbox->latest_measurement.htrace[0];
			}
			executor.sandbox->latest_measurement.htrace[0] = acc;
		} else {
			((void(*)(void*))measurement_code)(executor.sandbox);
		}
		mte_gcr_restore(saved_gcr);

		if (use_exec_keys) {
			pac_restore_sctlr(saved_sctlr);
			pac_load_keys(&saved_hw_keys);
		}

		raw_local_irq_restore(flags);

		measure(&current_input->measurement);
		splice_code_relocations(measurement_code, &current_input->input, true);
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

