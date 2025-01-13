#include "main.h"

// =================================================================================================
// Helper Functions
// =================================================================================================

/// Clears the programmable performance counters and writes the
/// configurations to the corresponding MSRs.
static int config_pfc(void) {

    // disable PMU user-mode access (not necessary?)
    uint64_t val = 0;

    // asm volatile("msr pmuserenr_el0, %0" :: "r" (0x1));
    // asm volatile("isb\n");

    // disable PMU counters before selecting the event we want
    val = 0;
    asm volatile("mrs %0, pmcr_el0" : "=r" (val));
    asm volatile("msr pmcr_el0, %0" :: "r" ((uint64_t)0x0));
    asm volatile("isb\n");

    // select events:
    // 1. L1D cache refills (0x3)
    asm volatile("msr pmevtyper0_el0, %0" :: "r" ((uint64_t)0x3));
    asm volatile("isb\n");

    // 2. Instructions retired (0x08)
    asm volatile("msr pmevtyper1_el0, %0" :: "r" ((uint64_t)0x08));
    asm volatile("isb\n");

    // 3. Instruction speculatively executed (0x1b)
    asm volatile("msr pmevtyper2_el0, %0" :: "r" ((uint64_t)0x1b));
    asm volatile("isb\n");
    // enable counting
    val = 0;
    asm volatile("msr pmcntenset_el0, %0" :: "r" ((uint64_t)0b111));
    asm volatile("isb\n");

    // enable PMU counters and reset the counters (using two bits)
    val = 0;
    asm volatile("mrs %0, pmcr_el0" : "=r" (val));
    asm volatile("msr pmcr_el0, %0" :: "r" (val | 0x3));
    asm volatile("isb\n");
    // debug prints (view via 'sudo dmesg')
    // val = 0;
    // asm volatile("mrs %0, pmuserenr_el0" : "=r" (val));
    // printk(KERN_ERR "%-24s 0x%0llx\n", "PMUSERENR_EL0:", val);
    // asm volatile("mrs %0, pmcr_el0" : "=r" (val));
    // printk(KERN_ERR "%-24s 0x%0llx\n", "PMCR_EL0:", val);
    // asm volatile("mrs %0, pmselr_el0" : "=r" (val));
    // printk(KERN_ERR "%-24s 0x%0llx\n", "PMSELR_EL0:", val);
    // asm volatile("mrs %0, pmevtyper0_el0" : "=r" (val));
    // printk(KERN_ERR "%-24s 0x%0llx\n", "PMEVTYPER0_EL0:", val);
    // asm volatile("mrs %0, pmcntenset_el0" : "=r" (val));
    // printk(KERN_ERR "%-24s 0x%0llx\n", "PMCNTENSET_EL0:", val);

    return 0;
}

static inline int setup_environment(void) {
    int err = 0;

    // TBD: configure PFC
    err = config_pfc();
    if (err)
        return err;

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

static void load_registers_from_input(input_t* input) {

	// Initial register values
	*((registers_t*)executor.sandbox.upper_overflow) = input->regs;

	// - flags
	((registers_t*)executor.sandbox.upper_overflow)->flags <<= 28;

	// - RSP and RBP
	((registers_t*)executor.sandbox.upper_overflow)->sp = (size_t)executor.sandbox.main_region + sizeof(executor.sandbox.main_region) - 8;
}

static void load_input_to_sandbox(input_t* input) {

	load_memory_from_input(input);
	load_registers_from_input(input);
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

static void measure(measurement_t* measurement) {

	// store the measurement results
	measurement->htrace[0] = executor.sandbox.latest_measurement.htrace[0];
	measurement->pfc[0] = executor.sandbox.latest_measurement.pfc[0];
	measurement->pfc[1] = executor.sandbox.latest_measurement.pfc[1];
	measurement->pfc[2] = executor.sandbox.latest_measurement.pfc[2];
}

static void __nocfi run_experiments(void) {
	int64_t rounds = (int64_t)executor.number_of_inputs;
	unsigned long flags = 0;
	struct rb_node* current_input_node = NULL;

	module_err("in run experiment 1!\n");
	if(0 >= executor.number_of_inputs){
		BUG_ON(0 > executor.number_of_inputs);
		module_err("No inputs were set!\n");
		return;
	}

	module_err("in run experiment 2!\n");
	get_cpu(); // pin the current task to the current cpu
	raw_local_irq_save(flags); // disable local interrupts and save current state

	module_err("in run experiment 3!\n");
	current_input_node = rb_first(&executor.inputs_root);
	BUG_ON(NULL == current_input_node);

	module_err("in run experiment 4!\n");
	// Zero-initialize the region of memory used by Prime+Probe
	memset(executor.sandbox.eviction_region, 0, sizeof(executor.sandbox.eviction_region));

	for (int64_t i = -executor.config.uarch_reset_rounds; i < rounds; ++i) {

		module_err("in run experiment 5!\n");
		struct input_node* current_input = NULL;

		// ignore "warm-up" runs (i<0)uarch_reset_rounds
		if(0 < i) {
			current_input_node = rb_next(current_input_node);
			BUG_ON(NULL == current_input_node);
		}

		module_err("in run experiment 6!\n");

		current_input = rb_entry(current_input_node, struct input_node, node);

		initialize_overflow_pages();

		load_input_to_sandbox(&current_input->input);
		module_err("in run experiment 7!\n");

		// flush some of the uarch state
		if (1 == executor.config.pre_run_flush) {
			// TBD
		}

		// execute
		module_err("Before execution!\n");
		((void(*)(void*))executor.measurement_code)(&executor.sandbox);
		module_err("After execution!\n");

		measure(&current_input->measurement);
		module_err("in run experiment 8!\n");
	}

	module_err("in run experiment 9!\n");

	raw_local_irq_restore(flags); // enable local interrupts with previously saved state
	put_cpu(); // free the current task from the current cpu
	module_err("in run experiment 10!\n");
}

int execute(void) {

    if (setup_environment()) {
        return -1;
    }

    run_experiments();

    return 0;
}

void initialize_measurement(measurement_t* measurement) {
	if(NULL == measurement) return;
	memset(measurement, 0, sizeof(measurement_t));
}
