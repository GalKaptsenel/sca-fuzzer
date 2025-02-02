#include "main.h"

static unsigned long copy_to_user_with_access_check(void __user* user_buffer,
 const void* from_buffer, size_t size) {

	if(!access_ok(user_buffer, size)) {
		module_err("Unable to access user buffer for writing!\n");
		return -EFAULT;
	}

	return copy_to_user(user_buffer, from_buffer, size);
}

static unsigned long copy_from_user_with_access_check(void* to_buffer,
 const void __user* user_buffer, size_t size) {

	if(!access_ok(user_buffer, size)) {
		module_err("Unable to access user buffer for reading!\n");
		return -EFAULT;
	}

	return copy_from_user(to_buffer, user_buffer, size);
}

static int load_input_id(int* p_input_id, void __user* arg) {
	unsigned long not_read = 0;

	if(NULL == p_input_id) {

		module_err("load_input_id - got NULL as output argument!\n");
		return -1;
	}

	not_read = copy_from_user_with_access_check(p_input_id, arg, sizeof(int));

	if(0 > *p_input_id) {
		module_err("Invalid input id! (got input id: %d)\n", *p_input_id);
	}

	return sizeof(int) - not_read;
}

static void checkout_into_input_id(void __user* arg) {
	int input_id = -1;

	load_input_id(&input_id, arg);

	if(0 <= input_id) {

		input_t* current_input = get_input(input_id);

		if(NULL != current_input) {
			executor.checkout_region = input_id;

		} else {
			module_err("Checkedout an input id that does not exist! (requested input id: %d)\n",
			 input_id);

		}
	}
}

static void measurements_became_unavailable(void) {
	module_info("Measurements became unavailable!\n");
}

static void reset_state_and_region_after_unloading_all_inputs(void) {

	module_info("all inputs were unloaded, checking out into TEST_REGION\n");
	executor.checkout_region = TEST_REGION;

	switch(executor.state) {
		case TRACED_STATE:
			measurements_became_unavailable();
		case READY_STATE:
			executor.state = LOADED_TEST_STATE;
			break;
		case LOADED_INPUTS_STATE:
		       executor.state = CONFIGURATION_STATE;
		       break;
		default:
		       break;
	}
}

static void free_input_id(void __user* arg) {

	int input_id = -1;

	load_input_id(&input_id, (void __user*)arg);

	if(0 <= input_id) {

		if(input_id == executor.checkout_region) {
			module_info("Freeing the currently checked out input, checking out into TEST_REGION.\n");
			executor.checkout_region = TEST_REGION;
		}

		remove_input(input_id);

		if(0 == get_number_of_inputs()) {
			reset_state_and_region_after_unloading_all_inputs();
		}
	}
}

static void clear_all_inputs(void) {
	destroy_inputs_db();
	initialize_inputs_db();
	reset_state_and_region_after_unloading_all_inputs();
}

static void measure_input_id(void __user* arg) {

	if(TRACED_STATE != executor.state) {
		module_err("Measurements are available only after performing a trace!\n");

	} else if(TEST_REGION == executor.checkout_region) {
		module_err("Checkout into the desired input!\n");

	} else {

		measurement_t* current_measurement = get_measurement(executor.checkout_region);
		BUG_ON(NULL == current_measurement);
		copy_to_user_with_access_check(arg, current_measurement, sizeof(measurement_t));
	}
}

static void unload_test_and_update_state(void) {
	memset(executor.test_case, 0, MAX_TEST_CASE_SIZE);
	memset(executor.measurement_code, 0, MAX_MEASUREMENT_CODE_SIZE);
	executor.test_case_length = 0;

	switch(executor.state) {
		case TRACED_STATE:
			measurements_became_unavailable();
		case READY_STATE:
		      executor.state = LOADED_INPUTS_STATE;
		      break;
		case LOADED_TEST_STATE:
		      executor.state = CONFIGURATION_STATE;
		      break;
		default:
		      break;
	}
}

static void update_state_after_writing_test(void) {
	switch(executor.state) {
		case CONFIGURATION_STATE:
			executor.state = LOADED_TEST_STATE;
			break;
		case LOADED_INPUTS_STATE:
			executor.state = READY_STATE;
			break;
		default:
			break;
	}
}

static int load_test_and_update_state(const char __user* test, size_t length) {
	int full_test_case_size = 0;

	if (MAX_TEST_CASE_SIZE < length) {
		return -ENOMEM;
	}

	copy_from_user_with_access_check(executor.test_case, test, length);

	executor.test_case_length = length;

	full_test_case_size = load_template(length);
	if(full_test_case_size < 0) {
		module_err("Failed to load test case (error code: %d)\n", full_test_case_size);
		return full_test_case_size;
	}

	update_state_after_writing_test();

	module_debug("%u bytes were written into test memory!\n", full_test_case_size);

	return full_test_case_size;
}


static int trace(void) {

	if(READY_STATE != executor.state && TRACED_STATE != executor.state) {
		module_err("In order to trace, please load inputs and test case.\n");
		return -EINVAL;
	}

	executor.state = TRACED_STATE;

	return execute();
}



static void get_test_length(void __user* arg) {
	uint64_t size = (uint64_t)-1;

	if(LOADED_TEST_STATE == executor.state ||
	 READY_STATE == executor.state ||
	  TRACED_STATE == executor.state) {
		size = executor.test_case_length;
		copy_to_user_with_access_check(arg, &size, sizeof(size));
	} else {
		module_err("Test has not been loaded!\n");
	}
}

static long revisor_ioctl(struct file* file, unsigned int cmd, unsigned long arg) {
	uint64_t result = 0;

	if(REVISOR_IOC_MAGIC != _IOC_TYPE(cmd)) {
		return -ENOTTY;
	}

	switch(_IOC_NR(cmd)) {

		case REVISOR_CHECKOUT_TEST_CONSTANT:
			module_debug("Checking out test memory (cmd: %d)..\n", REVISOR_CHECKOUT_TEST_CONSTANT);
			executor.checkout_region = TEST_REGION;
			break;

		case REVISOR_UNLOAD_TEST_CONSTANT:
			module_debug("Unloading test case (cmd: %d)..\n", REVISOR_UNLOAD_TEST_CONSTANT);
			unload_test_and_update_state();
			break;

		case REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT:
			module_debug("Querying number of inputs configured (cmd: %d)..\n",
			 REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT);
			result = get_number_of_inputs();
			result = copy_to_user_with_access_check((void __user*)arg, &result, sizeof(result));
			break;

		case REVISOR_CHECKOUT_INPUT_CONSTANT:
			module_debug("Checking out an input (cmd: %d)..\n", REVISOR_CHECKOUT_INPUT_CONSTANT);
			checkout_into_input_id((void __user*)arg);
			break;

		case REVISOR_ALLOCATE_INPUT_CONSTANT:
			module_debug("Allocating new input (cmd: %d)..\n", REVISOR_CHECKOUT_INPUT_CONSTANT);
			result = allocate_input();
			result = copy_to_user_with_access_check((void __user*)arg, &result, sizeof(result));
			break;

		case REVISOR_FREE_INPUT_CONSTANT:
			module_debug("Freeing input (cmd: %d)..\n", REVISOR_FREE_INPUT_CONSTANT);
			free_input_id((void __user*)arg);
			break;

		case REVISOR_MEASUREMENT_CONSTANT:
			module_debug("Querying measurement (cmd: %d)..\n", REVISOR_MEASUREMENT_CONSTANT);
			measure_input_id((void __user*)arg);
			break;

		case REVISOR_TRACE_CONSTANT:
			module_debug("Beginning trace (cmd: %d)..\n", REVISOR_TRACE_CONSTANT);
			trace();
			break;

		case REVISOR_CLEAR_ALL_INPUTS_CONSTANT:
			module_debug("Clearing all inputs (cmd: %d)..\n", REVISOR_CLEAR_ALL_INPUTS_CONSTANT);
			clear_all_inputs();
			break;

		case REVISOR_GET_TEST_LENGTH_CONSTANT:
			module_debug("Querying test case length (cmd: %d)..\n", REVISOR_GET_TEST_LENGTH_CONSTANT);
			get_test_length((void __user*)arg);
			break;
		default:
			module_err("Invalid IOCTL! Entered default case..\n");
			return -ENOTTY;
	}

	return 0;
}

static ssize_t revisor_read(struct file* File, char __user* user_buffer,
 size_t count, loff_t* off) {
	int number_of_bytes_to_copy  = 0;
	int not_copied = 0;
	uint64_t total_size = 0;
	void* from_buffer = NULL;

	if(NULL == user_buffer) {
		module_err("read callback - got NULL inside user_buffer!\n");
		return -EINVAL;
	}

	if(TEST_REGION == executor.checkout_region) {

		number_of_bytes_to_copy = min(count, (size_t)(executor.test_case_length - *off));
		from_buffer = executor.test_case;
		total_size = executor.test_case_length;

	} else {

		number_of_bytes_to_copy = min(count, (size_t)(sizeof(input_t) - *off));
		from_buffer = get_input(executor.checkout_region);
		total_size = sizeof(input_t);
	}

	if(total_size <= *off) {
		return 0; // return EOF
	}

	BUG_ON(NULL == from_buffer);

	not_copied = copy_to_user_with_access_check(user_buffer + *off, from_buffer + *off,
	 number_of_bytes_to_copy);

	*off += (number_of_bytes_to_copy - not_copied);

	return number_of_bytes_to_copy - not_copied;
}

static void update_state_after_writing_input(void) {
	switch(executor.state) {
		case CONFIGURATION_STATE:
			executor.state = LOADED_INPUTS_STATE;
			break;
		case LOADED_TEST_STATE:
			executor.state = READY_STATE;
			break;
		default:
			break;
	}
}

static void copy_input_from_user_and_update_state(const char __user* user_buffer, size_t count) {

	if(USER_CONTROLLED_INPUT_LENGTH == count) {
		void* to_buffer = get_input(executor.checkout_region);
		BUG_ON(NULL == to_buffer);

		copy_from_user_with_access_check(to_buffer, user_buffer, USER_CONTROLLED_INPUT_LENGTH);

		update_state_after_writing_input();

	} else {
		module_err("write callback - input must be exactly of length USER_CONTROLLED_INPUT_LENGTH(=%d)!\n",
		 USER_CONTROLLED_INPUT_LENGTH);

	}
}

static ssize_t revisor_write(struct file* File, const char __user* user_buffer,
 size_t count, loff_t* off) {

	if(NULL == user_buffer) {
		module_err("write callback - got NULL inside user_buffer!\n");
		return -1;
	}

	if(TEST_REGION == executor.checkout_region) {

		if(0 > load_test_and_update_state(user_buffer, count)) {
			return -1;
		}

	} else {

		BUG_ON(0 > executor.checkout_region);
		copy_input_from_user_and_update_state(user_buffer, count);
	}

	return count;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = revisor_ioctl,
    .read = revisor_read,
    .write = revisor_write,
};


int initialize_device_interface(void) {
	int status = 0;

	status = alloc_chrdev_region(&executor.device_mgmt.device_number, 0, 1, REVISOR_DEVICE_NAME);
	if(status < 0) {
		module_err("Unable to allocate character device number! Error code: %d\n", status);
		goto initialize_cleanup_exit_with_error;
	}

	cdev_init(&executor.device_mgmt.character_device, &fops);
	executor.device_mgmt.character_device.owner = THIS_MODULE;

	status = cdev_add(&executor.device_mgmt.character_device, executor.device_mgmt.device_number, 1);
	if(status < 0) {
		module_err("Unable to add character device to the system! Error code: %d\n", status);
		goto initialize_cleanup_allocated_device_numbers;
	}

	executor.device_mgmt.device_class = class_create(THIS_MODULE, REVISOR_DEVICE_CLASS_NAME);
	if (IS_ERR(executor.device_mgmt.device_class)) {
		module_err("Unable to create device class\n");
		status = PTR_ERR(executor.device_mgmt.device_class);
		goto initialize_cleanup_cdev_del;
	}

	if (!device_create(executor.device_mgmt.device_class, NULL, executor.device_mgmt.device_number,
	 NULL, REVISOR_DEVICE_NODE_NAME)) {
		module_err("Unable to create device node\n");
		status = -EINVAL;
		goto initialize_cleanup_class_destroy;
	}

	module_info("Registered device number MAJOR: %d, MINOR: %d and created cdev\n",
	 MAJOR(executor.device_mgmt.device_number), MINOR(executor.device_mgmt.device_number));

	return 0;

initialize_cleanup_class_destroy:
	class_destroy(executor.device_mgmt.device_class);
initialize_cleanup_cdev_del:
	cdev_del(&executor.device_mgmt.character_device);
initialize_cleanup_allocated_device_numbers:
	unregister_chrdev_region(executor.device_mgmt.device_number, 1);
initialize_cleanup_exit_with_error:
	return status;
}

void free_device_interface(void) {

    if (executor.device_mgmt.device_class) {
        device_destroy(executor.device_mgmt.device_class, executor.device_mgmt.device_number);
        class_destroy(executor.device_mgmt.device_class);
    }

    if (executor.device_mgmt.character_device.dev) {
        cdev_del(&executor.device_mgmt.character_device);
    }

    if (executor.device_mgmt.device_number) {
        unregister_chrdev_region(executor.device_mgmt.device_number, 1);
    }

    module_info("Unregistered device number MAJOR: %d, MINOR: %d and deleted cdev\n",
     MAJOR(executor.device_mgmt.device_number), MINOR(executor.device_mgmt.device_number));
}
