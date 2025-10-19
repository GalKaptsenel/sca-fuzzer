#include "main.h"

// =================================================================================================
// Helper Functions
// =================================================================================================

/// Clears the programmable performance counters and writes the
/// configurations to the corresponding MSRs.

//static void prepare_perf_event_attribute(struct perf_event_attr* attr, uint32_t type, uint64_t config) {
//	if(!attr) return;
//
//	memset(attr, 0, sizeof(*attr));
//	attr->type		= type;
//	attr->config		= config;
//	attr->size		= sizeof(*attr);
//	attr->disabled		= 0;      /* start counting immediately */
//	attr->pinned		= 1;      /* try hard not to multiplex */
//	attr->exclude_hv	= 1;
//	attr->exclude_user	= 1;
//	attr->exclude_kernel	= 0;
////	attr->exclude_idle	= 1;
//}
//
//static uint64_t next_perf_event_index = 0;
//static struct perf_event* perf_events[16] = { 0 };
//static int perf_event_to_pmevcntr_index(struct perf_event* ev) { return ev ? ev->hw.idx : -1; }
//static void overflow_cb(struct perf_event *event, struct perf_sample_data *data, struct pt_regs *regs) { }
//static int create_perf_event(uint32_t type, uint64_t config) {
//	if(next_perf_event_index >= ARRAY_SIZE(perf_events)) return -2;
//
//	uint64_t current_index = next_perf_event_index;
//
//	struct perf_event_attr attr = { 0 };
//	prepare_perf_event_attribute(&attr, type, config);
//	perf_events[current_index] = perf_event_create_kernel_counter(&attr, smp_processor_id(), NULL, overflow_cb, NULL);
//	if (IS_ERR(perf_events[current_index])) {
//		int err = PTR_ERR(perf_events[current_index]);
//		module_err("failed to create pmu event: %d (type: %u, config: %llu)", err, type, config);
//		perf_events[current_index] = NULL;
//		return -3;
//	}
//
//	perf_event_enable(perf_events[current_index]);
//
//	++next_perf_event_index;
//
//	return perf_event_to_pmevcntr_index(perf_events[current_index]);
//}
//
static int config_pfc(void) {

    // disable PMU user-mode access (not necessary?)
    uint64_t val = 0;
    uint64_t filter_events = (1 << 30) | (1 << 27);

    asm volatile("msr pmuserenr_el0, %0" :: "r" (0x1));
    asm volatile("isb\n");

    // disable PMU counters before selecting the event we want
    val = 0;
    asm volatile("mrs %0, pmcr_el0" : "=r" (val));
    asm volatile("msr pmcr_el0, %0" :: "r" ((uint64_t)0x0));
    asm volatile("isb\n");
    asm volatile("msr pmcntenclr_el0, %0" :: "r" ((uint64_t)0b1111));
    asm volatile("isb\n");


    // select events:
    // 1. L1D cache refills (0x3)
    asm volatile("msr pmevtyper0_el0, %0" :: "r" ((uint64_t)(filter_events | 0x03)));
    asm volatile("isb\n");

    // 2. Instructions retired (0x08)
    asm volatile("msr pmevtyper1_el0, %0" :: "r" ((uint64_t)(filter_events | 0x08)));
    asm volatile("isb\n");

    // 3. Instruction speculatively executed (0x1b)
    asm volatile("msr pmevtyper2_el0, %0" :: "r" ((uint64_t)(filter_events | 0x1b)));
    asm volatile("isb\n");

    // 4. L1D cache refills (0x3)
    asm volatile("msr pmevtyper3_el0, %0" :: "r" ((uint64_t)(filter_events | 0x3)));
    asm volatile("isb\n");

    // enable counting
    val = 0;
    asm volatile("msr pmcntenset_el0, %0" :: "r" (((uint64_t)0b1111) | (1ULL << 31)));
    asm volatile("isb\n");

    // enable PMU counters and reset the counters (using 3 bits)
    val = 0;
    asm volatile("mrs %0, pmcr_el0" : "=r" (val));
    asm volatile("msr pmcr_el0, %0" :: "r" (val | 0b111));
    asm volatile("isb\n");
    
    // debug prints (view via 'sudo dmesg')
    
     val = 0;
     asm volatile("mrs %0, pmuserenr_el0" : "=r" (val));
     module_debug(KERN_ERR "%-24s 0x%0llx\n", "PMUSERENR_EL0:", val);
     asm volatile("mrs %0, pmcr_el0" : "=r" (val));
     module_debug(KERN_ERR "%-24s 0x%0llx\n", "PMCR_EL0:", val);
     asm volatile("mrs %0, pmselr_el0" : "=r" (val));
     module_debug(KERN_ERR "%-24s 0x%0llx\n", "PMSELR_EL0:", val);
     asm volatile("mrs %0, pmevtyper0_el0" : "=r" (val));
     module_debug(KERN_ERR "%-24s 0x%0llx\n", "PMEVTYPER0_EL0:", val);
     asm volatile("mrs %0, pmcntenset_el0" : "=r" (val));
     module_debug(KERN_ERR "%-24s 0x%0llx\n", "PMCNTENSET_EL0:", val);
     asm volatile("mrs %0, pmevcntr1_el0" : "=r" (val));
     module_debug(KERN_ERR "%-24s 0x%0llx\n", "PMECNTR1_EL0:", val);
     asm volatile("mrs %0, pmevcntr2_el0" : "=r" (val));
     module_debug(KERN_ERR "%-24s 0x%0llx\n", "PMECNTR2_EL0:", val);
     asm volatile("mrs %0, pmevcntr3_el0" : "=r" (val));
     module_debug(KERN_ERR "%-24s 0x%0llx\n", "PMECNTR3_EL0:", val);
     asm volatile("mrs %0, pmccntr_el0" : "=r"(val));
     module_debug(KERN_ERR "%-24s 0x%0llx\n", "PNCCNTR_EL0:", val);

    return 0;
}

static void pmu_discover(void *arg)
{
    unsigned long flags;
    u64 pmcr, pmceid0, pmceid1, n, i;

    preempt_disable();
    local_irq_save(flags);

    /* Read PMCR_EL0 */
    asm volatile("mrs %0, pmcr_el0" : "=r"(pmcr));
    pr_info("PMCR_EL0: 0x%016llx\n", pmcr);

    /* Number of general-purpose counters: N = bits[15:11] + 1 */
    n = ((pmcr >> 11) & 0x1f) + 1;
    pr_info("Number of general-purpose counters: %llu\n", n);

    /* Check cycle counter */
    pr_info("Cycle counter exists: %s\n", (pmcr & (1<<31)) ? "YES" : "NO");

    /* Read supported events (PMCEID0_EL0 and PMCEID1_EL0) */
    asm volatile("mrs %0, pmceid0_el0" : "=r"(pmceid0));
    asm volatile("mrs %0, pmceid1_el0" : "=r"(pmceid1));

    pr_info("Supported event IDs 0-31 : 0x%016llx\n", pmceid0);
    pr_info("Supported event IDs 32-63: 0x%016llx\n", pmceid1);

    /* Print which counters are enabled (PMCNTENSET_EL0) */
    u64 cntenset;
    asm volatile("mrs %0, pmcntenset_el0" : "=r"(cntenset));
    pr_info("PMCNTENSET_EL0: 0x%016llx\n", cntenset);

    /* Print initial values of general-purpose counters and cycle counter */
    for (i = 0; i < n; i++) {
        u64 val;
        switch(i) {
            case 0: asm volatile("mrs %0, pmevcntr0_el0" : "=r"(val)); break;
            case 1: asm volatile("mrs %0, pmevcntr1_el0" : "=r"(val)); break;
            case 2: asm volatile("mrs %0, pmevcntr2_el0" : "=r"(val)); break;
            case 3: asm volatile("mrs %0, pmevcntr3_el0" : "=r"(val)); break;
            default: val = 0; break;
        }
        pr_info("PMEVCNTR%llu_EL0 initial value: %llu\n", i, val);
    }

    /* Cycle counter */
    u64 cc;
    asm volatile("mrs %0, pmccntr_el0" : "=r"(cc));
    pr_info("PMCCNTR_EL0 initial value: %llu\n", cc);

    local_irq_restore(flags);
    preempt_enable();
}

static inline u64 read_inst_count(void) {
	u64 val;
	asm volatile("mrs %0, PMEVCNTR4_EL0" : "=r"(val));
	return val;
}

static inline int setup_environment(void) {
    int err = 0;

    // TBD: configure PFC
    err = config_pfc();
    if (err)
        return err;

    int cpu = smp_processor_id();
    module_err("Starting PMU discovery on CPU %d\n", cpu);
    smp_call_function_single(cpu, pmu_discover, NULL, 1);
    // TBD: configure faulty page
    return 0;
}

static void load_memory_from_input(input_t* input) {

	// - sandbox: main and faulty regions
	for (int j = 0; j < (sizeof(executor.sandbox.main_region) / sizeof(uint64_t)); ++j) {
	        ((uint64_t*)executor.sandbox.main_region)[j] = ((uint64_t*)(input->main_region))[j];
	}

	for (int j = 0; j < (sizeof(executor.sandbox.faulty_region) / sizeof(uint64_t)); ++j) {
	        ((uint64_t*)executor.sandbox.faulty_region)[j] = ((uint64_t*)(input->faulty_region))[j];
	}
}

// RSP must be aligned to 16 bytes boundary, according to documentation of AARCH64
static size_t get_stack_base_address(void) {
	size_t address = ((size_t)executor.sandbox.main_region + sizeof(executor.sandbox.main_region));
	return PTR_ALIGN(address, 16); // Technically, kernel stack should be aligned to THREAD_SIZE, for example it allows access the thread_indo structure. But it is fine to just align to 16 bytes, due to hardware only checks this constraint.
}

static void load_registers_from_input(input_t* input, void* aux_buffer) {

	// Initial register values
	*((registers_t*)executor.sandbox.lower_overflow) = input->regs;

	// - flags
	((registers_t*)executor.sandbox.lower_overflow)->flags <<= 28;

	// - RSP and RBP
	((registers_t*)executor.sandbox.lower_overflow)->sp = get_stack_base_address();

	if(NULL != aux_buffer) {
		((registers_t*)executor.sandbox.lower_overflow)->x7 = (size_t)aux_buffer;
	}

	module_err("Input regs: x0:%llx, x1:%llx, x2:%llx x3:%llx, x4:%llx, x5:%llx, x6:%llx, x7 (Debug Page):%llx, flags:%llx, sp:%llx\n",
			*(uint64_t*)executor.sandbox.lower_overflow,
			*((uint64_t*)executor.sandbox.lower_overflow+1),
			*((uint64_t*)executor.sandbox.lower_overflow+2),
			*((uint64_t*)executor.sandbox.lower_overflow+3),
			*((uint64_t*)executor.sandbox.lower_overflow+4),
			*((uint64_t*)executor.sandbox.lower_overflow+5),
			*((uint64_t*)executor.sandbox.lower_overflow+6),
			*((uint64_t*)executor.sandbox.lower_overflow+7),
			*((uint64_t*)executor.sandbox.lower_overflow+8),
			*((uint64_t*)executor.sandbox.lower_overflow+9));
}

static void load_input_to_sandbox(input_t* input, void* aux_buffer) {
	load_memory_from_input(input);
	load_registers_from_input(input, aux_buffer);
}

static void initialize_overflow_pages(void) {

	// Initialize memory:
	// NOTE: memset is not used intentionally! somehow, it messes up with P+P measurements
	// - overflows are initialized with zeroes
	memset(executor.sandbox.lower_overflow, 0, sizeof(executor.sandbox.lower_overflow));
	for (int j = 0; j < (sizeof(executor.sandbox.upper_overflow) / sizeof(uint64_t)); ++j) {
	    ((uint64_t *)executor.sandbox.upper_overflow)[j] = 0;
	}
}

void initialize_measurement(measurement_t* measurement) {
	if(NULL == measurement) return;
	memset(measurement, 0, sizeof(measurement_t));
	measurement->aux_buffer = aux_buffer_alloc(19 * PAGE_SIZE); // For Full trace support, we allocate 19 pages of 4096 bytes. This allows us to log 255 instructions in total.
	if (NULL == measurement->aux_buffer) {
		module_err("initialize_measurement: aux_buffer_alloc returned NULL\n");
	}
}
EXPORT_SYMBOL(initialize_measurement);


void free_measurement(measurement_t* measurement) {
	if(NULL == measurement) return;
	aux_buffer_free(measurement->aux_buffer);
}
EXPORT_SYMBOL(free_measurement);

static void measure(measurement_t* measurement) {
	if(NULL == measurement) return;

	for(size_t i = 0; i < HTRACE_WIDTH; ++i) {
		measurement->htrace[i] = executor.sandbox.latest_measurement.htrace[i];
	}
	
	for(size_t i = 0; i < NUM_PFC; ++i) {
		measurement->pfc[i] = executor.sandbox.latest_measurement.pfc[i];
	}

	for(size_t i = 0; i < WIDTH_MEMORY_IDS; ++i) {
		measurement->memory_ids_bitmap[i] = executor.sandbox.latest_measurement.memory_ids_bitmap[i];
	}
}

static void flush_l1d_cache(void) {
	uint64_t clidr = 0, ccsidr = 0;
	uint32_t line_size = 0, assoc = 0, num_sets = 0;

	asm volatile("mrs %0, CLIDR_EL1" : "=r"(clidr) :: "memory");

	for(int level = 0; level < 7; ++level) {
		int ctype = (clidr >> (level * 3)) & 0b111;
		if(ctype < 2) {
			// no data/unified cache at this level
			continue;
		}

		write_sysreg(level << 1, csselr_el1);
		isb();

		ccsidr = read_sysreg(ccsidr_el1);
		line_size = (ccsidr & 0b111) + 4; // log2(words per line) + 2 for bytes: 2^line_size is the size of the line in bytes
		assoc = ((ccsidr >> 3) & 0x3FF) + 1; // ways
		num_sets = ((ccsidr >> 13) & 0x7FFF) + 1; // sets

		for(int way = 0; way < assoc; ++way) {
			for(int set = 0; set < num_sets; ++set) {
				uint64_t sw = (way << (32 - __builtin_clz(assoc - 1))) | (set << line_size);
				asm volatile("dc cisw, %0" :: "r"(sw) : "memory");
			}
		}
	}

	dsb(ish);
	isb();
}


static void __nocfi run_experiments(void) {
	int64_t rounds = (int64_t)executor.number_of_inputs;
	unsigned long flags = 0;
	struct rb_node* current_input_node = NULL;

	if(0 >= executor.number_of_inputs){
		BUG_ON(0 > executor.number_of_inputs);
		module_err("No inputs were set!\n");
		return;
	}

	module_err("inputs_root.rb_node=%zx, number_of_inputs=%llu\n", (size_t)executor.inputs_root.rb_node, executor.number_of_inputs);
	module_err("measurement area is at =%zx\n", (size_t)executor.measurement_code);

	current_input_node = rb_first(&executor.inputs_root);
	BUG_ON(NULL == current_input_node);

	// Zero-initialize the region of memory used by Prime+Probe
	memset(executor.sandbox.eviction_region, 0, sizeof(executor.sandbox.eviction_region));

	current->policy = SCHED_FIFO;
	current->rt_priority = MAX_RT_PRIO - 1;
	cpumask_t mask;
	cpumask_clear(&mask);
	cpumask_set_cpu(smp_processor_id(), &mask);
	set_cpus_allowed_ptr(current, &mask);

	for (int64_t i = -executor.config.uarch_reset_rounds; i < rounds; ++i) {

		struct input_node* current_input = NULL;

		// ignore "warm-up" runs (i<0)uarch_reset_rounds
		if(0 < i) {
			current_input_node = rb_next(current_input_node);
			BUG_ON(NULL == current_input_node);
		}

		current_input = rb_entry(current_input_node, struct input_node, node);

		initialize_overflow_pages();
		aux_buffer_init(current_input->measurement.aux_buffer);
		load_input_to_sandbox(&current_input->input, current_input->measurement.aux_buffer->addr);

		// flush some of the uarch state
		if (1 == executor.config.pre_run_flush) {
			// TBD
		}


//		config_pfc();

		raw_local_irq_save(flags); // disable local interrupts and save current state

		flush_l1d_cache();

		// execute
//		module_err("DEBUG 1: Before trace");
//		aux_buffer_dump_range(current_input->measurement.aux_buffer, 0, 0x400);
		((void(*)(void*))executor.measurement_code)(&executor.sandbox);
//		module_err("DEBUG 2: After trace");
//		aux_buffer_dump_range(current_input->measurement.aux_buffer, 0, 0x400);



		//enable_mte_tag_checking();

		raw_local_irq_restore(flags); // enable local interrupts with previously saved state

		measure(&current_input->measurement);
		module_err("htrace: %llu", current_input->measurement.htrace[0]);
		{
			char buff[65] = { 0 };
			for(int i = 0; i < 64; ++i) {
				buff[63-i] = (current_input->measurement.htrace[0] & ((uint64_t)1 << i)) ? '1' : '0';
			}
			buff[64] = 0;
	
			module_err("htrace: %s", buff);
			module_err("pfc[0]: %llu", current_input->measurement.pfc[0]);
			module_err("pfc[1]: %llu", current_input->measurement.pfc[1]);
			module_err("pfc[2]: %llu", current_input->measurement.pfc[2]);
//			module_err("DEBUG 3: Printer:");
			aux_buffer_dump_range(current_input->measurement.aux_buffer, 0, 0x400);

//			aux_buffer_dump(current_input->measurement.aux_buffer);
		}

	}
}

int execute(void) {

    if (setup_environment()) {
        return -1;
    }

    run_experiments();
    return 0;
}
EXPORT_SYMBOL(execute);

