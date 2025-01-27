#ifndef ARM64_EXECUTOR_EXECUTOR_H
#define ARM64_EXECUTOR_EXECUTOR_H

#include "utils.h"
#include "templates.h"

enum Templates;  // Forward declaration
enum State {
	CONFIGURATION_STATE,
	LOADED_TEST_STATE,
	LOADED_INPUTS_STATE,
	READY_STATE,
	TRACED_STATE,

};

// Executor Configuration Interface
#define UARCH_RESET_ROUNDS_DEFAULT	    1
#define ENABLE_FAULTY_DEFAULT		    0
#define PRE_RUN_FLUSH_DEFAULT		    1
#define NUMBER_OF_INPUTS_DEFAULT	    1
#define MEASUREMENT_TEMPLATE_DEFAULT	(FLUSH_AND_RELOAD_TEMPLATE)
#define REGION_DEFFAULT			        (TEST_REGION)

#define MAX_TEST_CASE_SIZE              PAGESIZE 
#define MAX_MEASUREMENT_CODE_SIZE       (PAGESIZE * 4)

#define TEST_REGION				        (-1)

typedef struct executor_config {
	long uarch_reset_rounds;
	char enable_faulty_page;
	char pre_run_flush;
	enum Templates measurement_template;
} executor_config_t;

typedef struct device_managment {
	struct cdev character_device;
	dev_t device_number;
	struct class *device_class;
} device_management_t;

typedef struct executor {
	executor_config_t config;
	char measurement_code[MAX_MEASUREMENT_CODE_SIZE];
	sandbox_t sandbox;
	volatile uint64_t number_of_inputs;
	char* test_case; // It is NOT embedded inside the struct because we require that it wll be continuous within physical memory, and therefore it should be acquired by kmalloc
	size_t test_case_length;
	struct rb_root inputs_root;
	long checkout_region;
	int tracing_error;
	enum State state;
	device_management_t device_mgmt;
} executor_t;

extern executor_t executor;

int __nocfi initialize_executor(set_memory_t);
void __nocfi free_executor(set_memory_t);

#endif // ARM64_EXECUTOR_EXECUTOR_H
