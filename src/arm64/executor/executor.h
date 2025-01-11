#ifndef ARM64_EXECUTOR_EXECUTOR_H
#define ARM64_EXECUTOR_EXECUTOR_H

#include "utils.h"
#include "measurement.h"
#include "templates.h"

enum Templates;  // Forward declaration
enum Region;  // Forward declaration
enum State;  // Forward declaration

// Executor Configuration Interface
#define UARCH_RESET_ROUNDS_DEFAULT	(1)
#define ENABLE_FAULTY_DEFAULT		(0)
#define PRE_RUN_FLUSH_DEFAULT		(1)
#define NUMBER_OF_INPUTS_DEFAULT	(1)
#define MEASUREMENT_TEMPLATE_DEFAULT	(FLUSH_AND_RELOAD)
#define REGION_DEFFAULT			(TEST_REGION)

#define MAX_TEST_CASE_SIZE              PAGESIZE        // must be exactly 1 page to detect sysfs buffering
#define MAX_MEASUREMENT_CODE_SIZE       (PAGESIZE * 2)

#define TEST_REGION				(-1)

typedef struct executor_config {
	long uarch_reset_rounds;
	char enable_faulty_page;
	char pre_run_flush;
	enum Templates measurement_template;
} executor_config_t;

typedef struct executor {
	executor_config_t config;
	sandbox_t sandbox;
	volatile u64 number_of_inputs;
	char* test_case; // It is NOT embeded inside the struct because we require that it wll be continuous within physival memory, and therefore it should be acquired by kmalloc
	u64 test_case_length;
	char measurement_code[MAX_MEASUREMENT_CODE_SIZE];
	struct rb_node inputs_root;
	long checkedout_region;
	int tracing_error;
	enum State state;
} executor_t;

extern executor_t executor;

int initialize_executor(set_memory_t, set_memory_t);
void free_executor(set_memory_t);
int load_test(char* test, size_t length);
void unload_test();
int load_input(input_t* in);
void unload_input(int id);
int allocate_input();
input_t get_input(int id);
u64 get_number_of_inputs();
char* get_test(); 

#endif // ARM64_EXECUTOR_EXECUTOR_H
