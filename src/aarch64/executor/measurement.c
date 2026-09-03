#include "main.h"
#include "pmu.h"
#include <linux/random.h>
#include <linux/moduleparam.h>

// TEMP bug-hunt: when set, dump the pre-execution tag/pointer state (mte_tag_verbose) but SKIP the
// test-case execution entirely. Isolates whether an "unexpected reset" comes from the divergent
// access during execution or from the tag setup/dump itself. Revert once the divergence is found.
int mte_skip_exec = 0;
module_param(mte_skip_exec, int, 0644);
MODULE_PARM_DESC(mte_skip_exec, "Dump pre-exec state then skip test-case execution (MTE divergence bug-hunt)");

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

/* Dump the actual per-granule tag of one region, read back via LDG, as chunked MTETAG lines. off_base
 * is the region's byte offset (may be negative) relative to main_region, so every HW/CE line aligns by
 * offset. */
static void mte_log_region_actual(const char* when, const char* region, int64_t off_base, const char* base, uint64_t nbytes) {
	static const char hexd[] = "0123456789abcdef";
	uint64_t ngran = nbytes / MTE_GRANULE_SIZE;
	char buf[129];
	for (uint64_t off = 0; off < ngran; off += 128) {
		uint64_t m = (ngran - off < 128) ? (ngran - off) : 128;
		for (uint64_t k = 0; k < m; ++k) {
			buf[k] = hexd[mte_read_tag(base + (off + k) * MTE_GRANULE_SIZE) & 0xF];
		}
		buf[m] = 0;
		mte_dump_add("MTETAG side=HW when=%s region=%s off=%lld n=%llu tags=%s",
		             when, region, (long long)(off_base + (int64_t)(off * MTE_GRANULE_SIZE)), m, buf);
	}
}

/* Dump the actual sandbox tag memory (LDG over lower_overflow | main | faulty | upper_overflow) for
 * CE<->HW persistency checking. `when` is "pretrace" (after the per-input tag reset, before the run) or
 * "posttrace" (after the run, reflecting the test case's STG* effects). Pointers only on pretrace. */
static void mte_dump_tags(const char* when, int64_t seq, int64_t iid, const input_t* input, int dump_ptrs) {
	if (!mte_tag_verbose) {
		return;
	}
	const char* mainr = executor.sandbox->main_region;
	mte_dump_add("MTETAG side=HW when=%s seq=%lld iid=%lld base=0x%lx mte_tags_present=%d",
	             when, (long long)seq, (long long)iid, (unsigned long)mainr, (int)input->mte_tags_present);
	mte_log_region_actual(when, "lower_overflow", -(int64_t)OVERFLOW_REGION_SIZE,
	                      executor.sandbox->lower_overflow, OVERFLOW_REGION_SIZE);
	mte_log_region_actual(when, "main",   0,                mainr, MAIN_REGION_SIZE);
	mte_log_region_actual(when, "faulty", MAIN_REGION_SIZE, executor.sandbox->faulty_region, FAULTY_REGION_SIZE);
	mte_log_region_actual(when, "upper_overflow", MEMORY_INPUT_SIZE,
	                      executor.sandbox->upper_overflow, OVERFLOW_REGION_SIZE);
	if (!dump_ptrs) {
		return;
	}
	const registers_t* r = &input->regs;
	const uint64_t regv[6] = { r->x0, r->x1, r->x2, r->x3, r->x4, r->x5 };
	for (int k = 0; k < 6; ++k) {
		mte_dump_add("MTEPTR side=HW seq=%lld x%d=0x%llx t=%x",
		             (long long)seq, k, (unsigned long long)regv[k], (unsigned)((regv[k] >> 56) & 0xF));
	}
	mte_dump_add("MTEPTR side=HW seq=%lld flags=0x%llx sp=0x%llx x29_base=0x%lx t=%x",
	             (long long)seq, (unsigned long long)r->flags, (unsigned long long)r->sp,
	             (unsigned long)mainr, (unsigned)(((uintptr_t)mainr >> 56) & 0xF));
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

	mte_dump_reset();   // buffer this run's tag log; flushed by the caller after the pinned call

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

	// The kernel's GCR_EL1 (KASAN reserves tags in Exclude) at test-case entry; each input clears
	// Exclude for its own execution and restores this value, and it is restored again at the TC
	// boundary so the kernel's GCR is intact between test cases.
	uint64_t pre_tc_gcr = mte_gcr_read();

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

		// Dump the actual tags/pointers BEFORE disabling IRQs: LDG reads allocation-tag memory and is
		// never tag-checked, so it is valid under the kernel's MTE control, and doing the (chatty) dump
		// here keeps its printk storm out of the IRQ-off measurement window (a long printk run on the
		// pinned CPU with IRQs disabled trips the hard-lockup watchdog -> panic).
		static int64_t mte_dump_seq = 0;
		int64_t this_dump_seq = mte_dump_seq;
		if (mte_tag_verbose) {
			++mte_dump_seq;
			mte_dump_tags("pretrace", this_dump_seq, current_input->id, &current_input->input, 1);
		}

		if (executor.config.reload_random_order) {
			fill_reload_order();
		}

		raw_local_irq_save(flags);

		// Re-assert the executor's full MTE control for THIS measurement, not just once at load: the
		// kernel restores SCTLR_EL1/TCR_EL1 on context switches, so TCF (sync vs the kernel's async),
		// TCMA1 (the harness reads the sandbox through the canonical tag-0b1111 base, which is only
		// Unchecked when TCMA1 is set) and TCO can otherwise revert to the kernel's values mid-campaign
		// and make the harness's own accesses tag-fault. Restore the kernel's values afterwards.
		struct mte_control_state pre_mte;
		mte_save_control(&pre_mte);
		mte_set_sync();            // TCF=SYNC (+ATA): precise, executor-attributed faults
		enable_TCMA1_bit();        // tag 0b0000/0b1111 Unchecked (canonical harness base)
		disable_TCO_bit();         // tag checks enabled

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
		mte_gcr_clear_exclude();
		mte_read_clear_tfsr();   // clear so a fault below is attributable to THIS input
		if (mte_skip_exec) {
			// BUGHUNT: state already dumped above; do NOT run the test case. If the phone still resets
			// with execution skipped, the reset is in the tag setup/dump, not the divergent access.
			executor.sandbox->latest_measurement.htrace[0] = 0;
		} else if (executor.config.reload_isolate) {
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
		mte_gcr_restore(pre_tc_gcr);

		// Post-execution tagmem (reflects the test case's STG* effects) for CE<->HW persistency checking;
		// the NEXT input's pretrace dump must then show a clean reset back to that input's tags.
		if (mte_tag_verbose) {
			mte_dump_tags("posttrace", this_dump_seq, current_input->id, &current_input->input, 0);
		}

		{
			uint64_t tf = mte_read_clear_tfsr();
			if (tf) {
				static int64_t tfsr_faults = 0;
				if (tfsr_faults++ < 20) {
					module_err("TFSR tag fault: input loop-idx=%lld iid=%lld TFSR_EL1=0x%llx "
					           "mte_tags_present=%d\n",
					           (long long)i, (long long)current_input->id, tf,
					           (int)current_input->input.mte_tags_present);
				}
			}
		}

		if (use_exec_keys) {
			pac_restore_sctlr(saved_sctlr);
			pac_load_keys(&saved_hw_keys);
		}
		mte_restore_control(&pre_mte);

		raw_local_irq_restore(flags);

		measure(&current_input->measurement);
		splice_code_relocations(measurement_code, &current_input->input, true);
	}

	if (ssbs_changed) {
		asm volatile("msr s3_3_c4_c2_6, %0\n isb\n" :: "r"(saved_ssbs) : "memory");
	}

	// Reset GCR_EL1 to the kernel's pre-test-case value at the TC boundary (each input already
	// restored it, but make the between-TC state explicit).
	mte_gcr_restore(pre_tc_gcr);

	return 0;
}

int execute(void) {

    int err = setup_environment();
    if (0 != err) {
        return err;
    }

    return run_experiments();
}

