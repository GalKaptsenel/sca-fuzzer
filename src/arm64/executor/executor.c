#include "main.h"

enum State {
	CONFIGURATION_STATE,
	LOADED_TEST_STATE,
	LOADED_INPUTS_STATE,
	READY_STATE,
	TRACED_STATE,
};

// Globals
executor_t executor	=	{ 0 }; 

static void init_executor_defaults(void) {
	executor.config.uarch_reset_rounds = UARCH_RESET_ROUNDS_DEFAULT;
	executor.config.enable_faulty_page = ENABLE_FAULTY_DEFAULT;
	executor.config.pre_run_flush = PRE_RUN_FLUSH_DEFAULT;
	executor.config.measurement_template = MEASUREMENT_TEMPLATE_DEFAULT;
	executor.checkedout_region = TEST_REGION;
}

int initialize_executor(set_memory_t set_memory_x, set_memory_t set_memory_nx) {
	int err = 0;

	init_executor_defaults();

	executor.test_case = kmalloc(MAX_TEST_CASE_SIZE, GFP_KERNEL);
	if (NULL == executor.test_case) {
        	module_err("Could not allocate memory for test case\n");
        	err = -ENOMEM;
		goto executor_init_failed_execution;
	}

	executor.test_case_length = 0;
	
	err = set_memory_x((unsigned long)executor.measurement_code, sizeof(executor.measurement_code) / PAGESIZE);
	if(err) {
		module_err("Failed to make executor.measurement_code executable\n");
		goto executor_init_failed_free_test_case;
	}

	initialize_inputs_db();

	executor.tracing_error = 0;
	executor.state = CONFIGURATION_STATE;

	return 0;

executor_init_failed_free_inputs:
	destroy_inputs_db();

executor_init_failed_make_ro:
	set_memory_nx((unsigned long)executor.measurement_code, sizeof(executor.measurement_code) / PAGESIZE);

executor_init_failed_free_test_case:
	kfree(executor.test_case);
	executor.test_case = NULL;

executor_init_failed_execution:
	return err;
}

void free_executor(set_memory_t set_memory_nx) {

	set_memory_nx((unsigned long)executor.measurement_code, sizeof(executor.measurement_code) / PAGESIZE); // TODO: can be skipped?

	destroy_inputs_db();

	if (executor.test_case) {
		kfree(executor.test_case);
		executor.test_case = NULL;
	}
}


int load_test(char* test, size_t length) {
	u64 test_case_size = 0;

	if (MAX_TEST_CASE_SIZE < length) {
		return -ENOMEM;
	}

	executor.test_case_length = length;
	strncpy(executor.test_case, test, length);
	test_case_size = load_template(length);
	if(test_case_size < 0) {
		module_err("Failed to load test case (code: %d)\n", test_case_size);
	}

	switch(executor.state) {
		case CONFIGURATION_STATE:
			executor.state = LOADED_TEST_STATE;
			break;
		case LOADED_INPUTS_STATE:
			executor.state = READY_STATE;
			break;
	}

	return test_case_size;
}

static void measurements_became_unavailable(void) {
	module_info("Measurements became unavailable!\n");
}

void unload_test() {
	memset(executor.test_case, 0, MAX_TEST_CASE_SIZE);
	memset(executor.measurement_code, 0, MAX_MEASUREMENT_CODE_SIZE); 
	executor.test_case_length = 0;

	switch(executor.state) {
		case: TRACED_STATE:
			measurements_became_unavailable();
		case READY_STATE:
		      executor.state = LOADED_INPUTS_STATE;
		      break;
		case LOADED_TEST_STATE:
		      executor.state = CONFIGURATION_STATE;
		      break;
	}
}

int load_input(input_t *in) {
	return insert_input(in);
}

static void reset_state_and_region_after_unloading_all_inputs(void) {

	module_info("all inputs were unloaded, checking out into TEST_REGION\n");
	executor.checkedout_region = TEST_REGION;

	switch(executor.state) {
		case TRACED_STATE:
			measurements_became_unavailable();
		case READY_STATE:
			executor.state = LOADED_TEST_STATE;
			break;
		case LOADED_INPUTS_STATE:
		       executor.state = CONFIGURATION_STATE;	
		       break;
	}
}

void unload_input(int id) {
	remove_input(id);

	if(0 == get_number_of_inputs()) {

		reset_state_after_unloading_all_inputs();
	}
}

void clear_all_inputs(void) {
	destroy_inputs_db();
	initialize_inputs_db();
	reset_state_and_region_after_unloading_all_inputs();
}

input_t* get_input(int id) {
	return find_input(id);
}

u64 get_number_of_inputs() {
	return executor.number_of_inputs;
}

char* get_test(size_t *length) {
	if(length) *length = executor.test_case_length;
	return executor.test_case;
}

