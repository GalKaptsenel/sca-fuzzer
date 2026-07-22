#ifndef ARM64_EXECUTOR_EXECUTOR_H
#define ARM64_EXECUTOR_EXECUTOR_H

#include "utils.h"
#include "templates_jit.h"
#include "pac.h"

enum State {
	CONFIGURATION_STATE,
	LOADED_TEST_STATE,
	LOADED_INPUTS_STATE,
	READY_STATE,
	TRACED_STATE,

};

// Executor Configuration Interface
#define UARCH_RESET_ROUNDS_DEFAULT	    	5
#define ENABLE_FAULTY_DEFAULT		    	0
#define PRE_RUN_FLUSH_DEFAULT		    	1
#define MEASUREMENT_TEMPLATE_DEFAULT		(FLUSH_AND_RELOAD_TEMPLATE)
#define CPU_ID_DEFAULT				(-1)
#define REGION_DEFAULT				(TEST_REGION)

#define MAX_TEST_CASE_SIZE              (1 * PAGESIZE)
/* Fits a full test case plus the largest harness; F+R's unrolled flush+reload is a ~14KB fixed cost. */
#define MAX_MEASUREMENT_CODE_SIZE       (MAX_TEST_CASE_SIZE * 8)

#define MAX_MEASUREMENT_VIEWS		(16)

#define TEST_REGION				        (-1)

typedef struct executor_config {
	long uarch_reset_rounds;
	char enable_faulty_page;
	char pre_run_flush;
	char phr_flush;        /* independent: run flush_bpu_phr before the measured run */
	char view_rotation;    /* independent: rotate measurement view (invalidate_bpu_entries) */
	enum Templates measurement_template;
	int pinned_cpu_id;
	char enable_ssbs;
	int reload_timer_shift;  /* F+R reload hit test: <0 => L1D_CACHE_REFILL delta==0 (default); >=0 => PMCCNTR cycle delta < (1<<shift) */
	int reload_random_order; /* F+R reload: 0 => fixed bit-reversed set order; 1 => fresh random permutation per run */
	int reload_pmu_event;    /* PMU event id counted by reload()'s hit test (delta==0). Default 0x03 = L1D_CACHE_REFILL; e.g. 0x17 = L2D_CACHE_REFILL */
	int reload_isolate;      /* 1 => reload probes only executor.reload_target_set (per-set isolation; C re-runs the test per set) */
} executor_config_t;

typedef struct device_managment {
	struct cdev character_device;
	dev_t device_number;
	struct class *device_class;
} device_management_t;

typedef struct executor {
	sandbox_t *sandbox;
	executor_config_t config;
	volatile uint64_t number_of_inputs;
	char* test_case; // The member test_case is NOT embedded inside the struct because we require that it wll be continuous within physical memory, and therefore it should be acquired by kmalloc
	size_t test_case_length;
	struct rb_root inputs_root;
	int64_t checkout_region;
	int tracing_error;
	enum State state;
	device_management_t device_mgmt;
	void* measurement_code_views[MAX_MEASUREMENT_VIEWS];
	uint64_t debug_reload_refills[128];  /* DEBUG: per-set counter0 delta from reload(); [0,63]=main, [64,127]=faulty */
	uint64_t debug_reload_refills2[128]; /* DEBUG: per-set counter1 (L2D_CACHE_REFILL) delta from reload() */
	uint64_t reload_order[64];           /* per-run set-visit order for reload() when reload_random_order is set */
	uint64_t reload_target_set;          /* the single set reload() probes when reload_isolate is set */
	uint64_t debug_ssbs;                 /* DEBUG: SSBS register value read back during the last measurement (bit 12 = PSTATE.SSBS) */
	uint64_t debug_reload_total_l2d;     /* DEBUG: L2D_CACHE_REFILL count over the whole reload (demand + prefetch) */
	uint64_t debug_reload_total_mem;     /* DEBUG: MEM_ACCESS count over the whole reload (demand only) */
} executor_t;

int __nocfi initialize_executor(set_memory_t);
void __nocfi free_executor(set_memory_t);

#endif // ARM64_EXECUTOR_EXECUTOR_H
