#include "main.h"

static long load_input_id(long* p_input_id, void __user* arg) {

	if(NULL == p_input_id) {

		module_err("load_input_id - got NULL as output argument!\n");
		return -1;
	}

	memcpy(p_input_id, arg, sizeof(long)); 

	if(0 > *p_input_id) {
		module_err("Invalid input id! (got input id: %d)\n", *input_id);
	}

	return *p_input_id;
}

static void inner_checkout_into_input_id(void __user* arg) {

	long input_id = -1;

	if(0 <= load_input_id(&input_id, arg)) {

		input_t* current = get_input(input_id);

		if(NULL != current) {
			executor.checkedout_region = input_id;

		} else {
			module_err("Checkedout an input id that does not exist! (requested input id: %d)\n", input_id);
		
		}
	}
}

static void inner_free_input_id(void __user* arg) {

	long input_id = -1;

	if(0 <= load_input_id(&input_id, (void __user*)arg)) { 

		if(input_id == executor.checkedout_region) {
			module_info("Freeing the currently checkedout input, checking out into TEST_REGION.\n");
			executor.checkedout_region = TEST_REGION;
		}
		
		unload_input(input_id);
	}
}

static void inner_measure_input_id(void __user* arg) {

	if(TRACED_STATE != executor.state) {
		module_err("Measurements are available only after performing a trace!\n");

	} else if(TEST_REGION == executor.checkedout_region) {
		module_err("Checkout into the desired input!\n"); 

	} else {

		input_t* current = get_input(executor.checkedout_region);
		if(NULL != current) {
			memcpy(arg, &(current->measurement), sizeof(input->measurement));
		}
		else {
			module_alert("Unexpected error at REVISOR_MEASUREMENT_CONSTANT\n");
		}
	}
}

static int trace(void) {

	if(READY_STATE != executor.state) {
		module_err("In order to trace, please load inputs and test case.\n");
		return -EINVAL;
	}

	executor.state = TRACED_STATE;

	execute();
}

static void inner_get_test_length(void __user* arg) {
	u64 size = -1;

	if(LOADED_TEST_STATE == executor.state || READY_TEST_STATE == executor.state || TRACED_STATE == executor.state) {
		size = executor.test_case_length;
		memcpy(arg, &size, sizeof(size));
	} else {
		module_err("Test has not been loaded!\n");
	}
}

static long revisor_ioctl(struct file* file, unsigned int cmd, unsigned long arg) {

	u64 result = 0;

	if(REVISOR_IOC_MAGIC != _IOC_TYPE(cmd)) {
		return -ENOTTY;
	}
	
	switch(_IOC_NR(cmd)) {

		case REVISOR_CHECKOUT_TEST:
			executor.checkedout_region = TEST_REGION;
			break;

		case REVISOR_UNLOAD_TEST_CONSTANT:
			unload_test();
			break;

		case REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT:
			result = get_number_of_inputs();
			memcpy((void __user*)arg, &result, sizeof(result)); 
			break;

		case REVISOR_CHECKOUT_INPUT:
			inner_checkout_into_input_id((void __user*)arg);
			break;

		case REVISOR_ALLOCATE_INPUT_CONSTANT:
			result = allocate_input();
			memcpy((void __user*)arg, &result, sizeof(result)); 
			break;

		case REVISOR_FREE_INPUT_CONSTANT:
			inner_free_input_id((void __user*)arg);
			break;

		case REVISOR_MEASUREMENT_CONSTANT:
			inner_measure_input_id((void __user*)arg);
			break;

		case REVISOR_TRACE_CONSTANT:
			trace();
			break;

		case REVISOR_CLEAR_ALL_INPUTS_CONSTANT:
			clear_all_inputs();
			break;	
		case REVISOR_GET_TEST_LENGTH_CONSTANT:
			inner_get_test_length((void __user* arg)arg);
			break;
		default:
			module_err("Invalid IOCTL!\n");
			return -ENOTTY;
	}
}

static ssize_t revisor_read(struct file* File, char __user* user_buffer, size_t count, loff_t* off) {
	int number_of_byted_to_copy  = 0;
	void* from_buffer = NULL;

	if(NULL == user_buffer) {
		module_err("read callback - got NULL inside user_buffer!\n");
		return -EINVAL;
	}

	if(TEST_REGION == executor.checkedout_region) {

		number_of_byted_to_copy = min(count, executor.test_case_length);
		from_buffer = executor.test_case;

	} else {

		number_of_byted_to_copy = min(count, sizeof(input_t));
		from_buffer = get_input(executor.checkedout_region);

	}

	if(NULL == from_buffer) {
		module_alert("read callback - failed to locate the source buffer inside the kernel! This is an unexpected behaviour!\n");
		return -EIO;
	}

	memcpy(user_buffer, from_buffer, number_of_bytes_to_copy);

	return 0; // return EOF
}

static void update_state_after_writing_test(void) {
	switch(executor.state) {
		case CONFIGURATION_STATE:
			executor.state = LOADED_TEST_STATE;
			break;
		case LOAD_INPUTS_STATE:
			executor.state = READY_STATE;
			break;
	}
}

static void update_state_after_writing_input(void) {
	switch(executor.state) {
		case CONFIGURATION_STATE:
			executor.state = LOADED_INPUTS_STATE;
			break;
		case LOADED_TEST_STATE:
			executor.state = READY_STATE;
			break;
	}
}

static ssize_t revisor_write(struct file* File, char __user* user_buffer, size_t count, loff_t* off) {

	int number_of_bytes_to_copy = 0;
	void* to_buffer = NULL;

	if(NULL == user_buffer) {
		module_err("write callback - got NULL inside user_buffer!\n");
		return -EINVAL;
	}

	if(TEST_REGION == executor.checkedout_region) {
		number_of_bytes_to_copy = min(count, MAX_TEST_CASE_SIZE);
		to_buffer = executor.test_case;

	} else {

		if(sizeof(input_t) != count) {
			module_err("write callback - input must be of sizeof(input_t)(=%d)!\n", sizeof(input_t));
			return -EINVAL;
		}

		number_of_bytes_to_copy = count; 
		to_buffer = get_input(executor.checkedout_region);
	}

	if(NULL == to_buffer) {
		module_alert("write callback - failed to locate the deestination buffer inside the kernel! This is an unexpected behaviour!\n");
		return -EIO;
	}

	memcpy(to_buffer, user_buffer, number_of_bytes_to_copy);

	if(TEST_REGION == executor.checkedout_region) {
		update_state_after_writing_test();
		executor.test_case_length = number_of_bytes_to_copy;

	} else {
		update_state_after_writing_input();
	}

	return number_of_bytes_to_copy;
}

