#include "executor_ioctl.h"
#include "executor_utils.h"

unsigned long int_to_cmd[] = {
	0,
	REVISOR_CHECKOUT_TEST,
	REVISOR_UNLOAD_TEST,
	REVISOR_GET_NUMBER_OF_INPUTS,
	REVISOR_CHECKOUT_INPUT,
	REVISOR_ALLOCATE_INPUT,
	REVISOR_FREE_INPUT,
	REVISOR_MEASUREMENT,
	REVISOR_TRACE,
	REVISOR_CLEAR_ALL_INPUTS,
	REVISOR_GET_TEST_LENGTH,
};

/*
static inline struct timespec timer_start(){
    struct timespec start_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    return start_time;
}

static inline long timer_end(struct timespec start_time){
    struct timespec end_time = { 0 };
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time.tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    return diffInNanos;
}

static long inline function_under_test_single(int fd) {
	long time_elapsed_nanos = 0;
	struct timespec stime = {0};

	unsigned char buffer[sizeof(input_t)] = { 0 };

	int fd_randomness = open("/dev/urandom", O_RDONLY);

    	if (fd_randomness < 0) {
        	perror("open");
	       	return -1;
	}

	ssize_t bytes_read = read(fd_randomness, buffer, USER_CONTROLLED_INPUT_LENGTH);
	if (bytes_read != USER_CONTROLLED_INPUT_LENGTH) {
        	perror("read");
		close(fd_randomness);
        	return -2;
    	}

	stime = timer_start();

	uint64_t allocated_iid = ioctl(fd, int_to_cmd[REVISOR_ALLOCATE_INPUT_CONSTANT]);
	ioctl(fd, int_to_cmd[REVISOR_CHECKOUT_INPUT_CONSTANT], allocated_iid);
	write(fd, buffer, USER_CONTROLLED_INPUT_LENGTH);

	time_elapsed_nanos = timer_end(stime);

	close(fd_randomness);

	return time_elapsed_nanos;
}


static long inline function_under_test_batch_indirect(int fd, size_t batch_size) {
	int err = 0;
	long time_elapsed_nanos = 0;
	struct timespec stime = {0};

	struct input_batch* input_batch = (struct input_batch*)malloc(sizeof(struct input_batch) + batch_size * sizeof(struct input_and_id_pair));
	if(NULL == input_batch) {
		perror("malloc batched input_batch");
		err = -1;
		goto function_under_test_batch_indirect_error;
	}

	input_batch->size = batch_size;

	int fd_randomness = open("/dev/urandom", O_RDONLY);

    	if (fd_randomness < 0) {
        	perror("open");
		err = -2;
		goto function_under_test_batch_indirect_input_batch;
	}



	for(int i = 0; i < batch_size; ++i) {
		input_batch->array[i].id = -1;
		input_batch->array[i].input = (input_t*)malloc(sizeof(input_t));

		if(NULL == input_batch->array[i].input) {
			err = -3;
			goto function_under_test_batch_indirect_free_inputs;
		}

		ssize_t bytes_read = read(fd_randomness, input_batch->array[i].input, sizeof(input_t));
		if (bytes_read != sizeof(input_t)) {
			perror("read urandom");
			err = -4;
			goto function_under_test_batch_indirect_free_inputs;
		}
	}

	stime = timer_start();

	ioctl(fd, REVISOR_BATCHED_INPUTS, input_batch);

	time_elapsed_nanos = timer_end(stime);

	err = time_elapsed_nanos;

	for(int i = 0; i < batch_size; ++i) {
		printf("input at index %d got id %ld\n", i, input_batch->array[i].id);
	}


	printf("---------- finished %zu inputs! -------------\n", input_batch->size);



function_under_test_batch_indirect_free_inputs:

	for(int i = 0; i < batch_size; ++i) {
		if(input_batch->array[i].input) {
			free(input_batch->array[i].input);
		}
	}

function_under_test_batch_indirect_randomness_close:
	close(fd_randomness);

function_under_test_batch_indirect_input_batch:
	free(input_batch);

function_under_test_batch_indirect_error:
	return err;
}


static long inline function_under_test_batch_direct(int fd, size_t batch_size) {
	int err = 0;
	long time_elapsed_nanos = 0;
	struct timespec stime = {0};

	struct input_batch* input_batch = (struct input_batch*)malloc(sizeof(struct input_batch) + batch_size * sizeof(struct input_and_id_pair));
	if(NULL == input_batch) {
		perror("malloc batched input_batch");
		err = -1;
		goto function_under_test_batch_direct_error;
	}

	input_batch->size = batch_size;

	int fd_randomness = open("/dev/urandom", O_RDONLY);

    	if (fd_randomness < 0) {
        	perror("open");
		err = -2;
		goto function_under_test_batch_direct_input_batch;
	}



	for(int i = 0; i < batch_size; ++i) {
		input_batch->array[i].id = -1;

		ssize_t bytes_read = read(fd_randomness, &(input_batch->array[i].input), sizeof(input_t));
		if (bytes_read != sizeof(input_t)) {
			perror("read urandom");
			err = -4;
			goto  function_under_test_batch_direct_randomness_close;
		}
	}

	stime = timer_start();

	ioctl(fd, REVISOR_BATCHED_INPUTS, input_batch);

	time_elapsed_nanos = timer_end(stime);

	err = time_elapsed_nanos;

	for(int i = 0; i < batch_size; ++i) {
		printf("input at index %d got id %ld\n", i, input_batch->array[i].id);
	}


	printf("---------- finished %zu inputs! -------------\n", input_batch->size);


function_under_test_batch_direct_randomness_close:
	close(fd_randomness);

function_under_test_batch_direct_input_batch:
	free(input_batch);

function_under_test_batch_direct_error:
	return err;
}


static inline long post_test(int fd) {
	long time_elapsed_nanos = 0;
	struct timespec stime = {0};

	stime = timer_start();

	ioctl(fd, int_to_cmd[REVISOR_TRACE_CONSTANT]);
	ioctl(fd, int_to_cmd[REVISOR_CLEAR_ALL_INPUTS_CONSTANT]);
	ioctl(fd, int_to_cmd[REVISOR_UNLOAD_TEST_CONSTANT]);

	time_elapsed_nanos = timer_end(stime);

	return time_elapsed_nanos;
}

static inline long pre_test(int fd) {
	long time_elapsed_nanos = 0;
	struct timespec stime = {0};
	char* test = NULL;
	size_t size = 0;
	const char* input_test = "spectre_v1.bin";

	read_file(input_test, &test, &size);
	stime = timer_start();

	ioctl(fd, int_to_cmd[REVISOR_CHECKOUT_TEST_CONSTANT]);
	write(fd, test, size);

	time_elapsed_nanos = timer_end(stime);

	free(test);
	return time_elapsed_nanos;
}

static int T_operation(int fd) {
	const int number_of_tests = 1001;
	const int addative_factor = 10;
	const int initial_value = 1;
	int current_value = initial_value;
	struct pair {
		unsigned int repeats;
		long time_elapsed;
	};
	struct pair* results = (struct pair*)malloc(number_of_tests*sizeof(struct pair));

	if(NULL == results) return -1;

	memset(results, 0, number_of_tests*sizeof(struct pair));


	for(int i = 0; i < number_of_tests; ++i, current_value += addative_factor) {

		long total_time = pre_test(fd);

		results[i].repeats = current_value;

		for(int j = 0; j < current_value; ++j) {

			int ret = function_under_test_single(fd);

			if(0 > ret) {
				printf("failed with %d\n", ret);
				free(results);
				return ret;
			}

			total_time += ret;
		}

//		total_time += function_under_test_batch_indirect(fd, current_value);

		total_time += function_under_test_batch_direct(fd, current_value);

		total_time += post_test(fd);

		results[i].time_elapsed = total_time;
	}

	for(int i = 0; i < number_of_tests; ++i) {

		printf("Repeatitions: %d, Time elapsed: %ld\n", results[i].repeats, results[i].time_elapsed);
	}

	free(results);
	return 0;
}

struct pair {
	unsigned int repeats;
	long time_elapsed;
};
*/
static void print_usage(const char *prog_name) {
	printf("Usage: %s <device> <command [argument]|w file|r file> \n", prog_name);
	printf("\t<device>  : Path to the device (e.g., /dev/revizor_device)\n");
	printf("\t<command [arguemnt]|w file|r file> :\n");
	printf("\t		- IOCTL command number (integer)\n");
	printf("\t			[argument]: Optional argument for the IOCTL command (integer or hex)\n");
	printf("\t		- w for writing to the device\n");
	printf("\t			file: file with data to write to the device\n");
	printf("\t		- r for reading from the device\n");
	printf("\t			file: file to read device contents into\n");
	printf("\tExample: %s /dev/revizor_device 3 0x100\n", prog_name);
	printf("\tExample: %s /dev/revizor_device w /my/src/file/namr\n", prog_name);
	printf("\tExample: %s /dev/revizor_device r /my/dst/file/namr\n", prog_name);
}

static int read_device(int fd, char** buffer, size_t* size) {
	const size_t chunk_size = 1024 * 4;
	size_t total_size = 0;
	char* temp_buffer = NULL;
	char* new_buffer = NULL;
	ssize_t bytes_read = 0;
	*buffer = NULL;

	temp_buffer = malloc(chunk_size);
	if (NULL == temp_buffer) {
		perror("Error allocating memory");
		goto read_device_exit;
	}

	while ((bytes_read = read(fd, temp_buffer, chunk_size)) > 0) {

		new_buffer = realloc(*buffer, total_size + bytes_read);

		if (NULL == new_buffer) {
			perror("Error reallocating memory");
			goto read_device_free_buffers;
		}

		*buffer = new_buffer;
		memcpy(*buffer + total_size, temp_buffer, bytes_read);
		total_size += bytes_read;
	}

	if (0 > bytes_read) {
		perror("Error reading file");
		goto read_device_free_buffers;
	}

	free(temp_buffer);

	*size = total_size;
	return 0;

read_device_free_buffers:

	if(*buffer) {
		free(*buffer);
	}

	free(temp_buffer);

read_device_exit:
	return -1;
}

static int read_file(const char* file_path, char** buffer, size_t* size) {

	struct stat st = { 0 };
	int fd = -1;

	if (0 > stat(file_path, &st)) {
		perror("Error getting file size");
		goto read_file_failure;
	}

	*size = st.st_size;
	*buffer = malloc(*size);

	if (NULL == *buffer) {
		perror("Error allocating memory for file");
		goto read_file_failure;
	}

	fd = open(file_path, O_RDONLY);

	if (0 > fd) {
		perror("Error opening file");
		goto read_file_cleanup_free_memory;
	}

	if ((ssize_t)*size != read(fd, *buffer, *size)) {
		perror("Error reading file");
		goto read_file_cleanup_close_file;
	}

	close(fd);
	return 0;

read_file_cleanup_close_file:
	close(fd);
read_file_cleanup_free_memory:
	free(*buffer);
read_file_failure:
	return -1;
}

static int handle_returned_uint64(int fd, int command, const char* prompt) {
	uint64_t value= 0;
	int result = ioctl(fd, command, &value);
	if (0 <= result) {
	    printf("%s: %lu\n", prompt, value);
	}

	return result;
}

static void print_bits64(uint64_t u) {
	char str[66];
//	str[64] = '\n';
	str[64] = 0;
	for (int i = 63; i >= 0; i--) {
		str[i] = '0' + (u & 1);
		u >>= 1;
	}

	printf("%s", str);
}

static int handle_get_measurement(int fd, int command) {
	measurement_t measurement = { 0 };
	int result = ioctl(fd, command, &measurement);

	if (0 <= result) {

		printf("Measurement:\n");

		for(int i = 0; i < HTRACE_WIDTH; ++i) {
	    		printf("\thtrace %d: ", i);
    			print_bits64(measurement.htrace[i]);
		}

		printf("\n");

		for(int i = 0; i < NUM_PFC; ++i) {
			printf("\tpfc %d: %lu\n", i, measurement.pfc[i]);
		}

		printf("\tarchitectural memory access bitmap: ");

		for(int i = WIDTH_MEMORY_IDS - 1; i >= 0; --i) {
			print_bits64(measurement.memory_ids_bitmap[i]);
		}

		printf("\n");
	}

	return result;
}

static int write_operation(int fd, const char* filename) {

	char* file_data = NULL;
	size_t file_size = 0;
	int err = EXIT_SUCCESS;

	do {
		if(0 > read_file(filename, &file_data, &file_size)) {
			err = EXIT_FAILURE;
			break;
		}

		printf("File loaded: %s (%zu bytes)\n", filename, file_size);

		err = write(fd, file_data, file_size);
		if(0 > err) {
			err = EXIT_FAILURE;
			printf("Unable to write device. Error Number:  %d.\n", err);
			break;
		}

	} while(0);

	if(file_data) {
		free(file_data);
	}

	return err;
}

static int read_operation(int fd, const char* filename) {

	char* file_data = NULL;
	size_t file_size = 0;
	int file_fd = -1;
	int err = EXIT_SUCCESS;

	do {

		if(0 != read_device(fd, &file_data, &file_size)) {
			perror("Unable to read.\n");
			err = EXIT_FAILURE;
			break;
		}

		printf("%zu bytes were loaded from device\n", file_size);

		file_fd  = open(filename, O_WRONLY | O_CREAT);

		if (0 > file_fd) {
			printf("Error opening file %s", filename);
			err = EXIT_FAILURE;
			break;
		}

		err = write(file_fd, file_data, file_size);

		if(0 > err) {
			printf("Unable to write into %s. Error Number: %d.\n", filename, err);
			err = EXIT_FAILURE;
			break;
		}

	} while(0);

	if(file_data) {
		free(file_data);
	}

	if(-1 != file_fd) {
		close(file_fd);
	}

	return err;
}

static int serve_numerical_command_without_argument(int fd, int command) {

	int result = 0;

	switch (command) {
		case REVISOR_GET_TEST_LENGTH:
			result = handle_returned_uint64(fd, command, "Test Length");
			break;

		case REVISOR_MEASUREMENT:
			result = handle_get_measurement(fd, command);
			break;

		case REVISOR_GET_NUMBER_OF_INPUTS:
			result = handle_returned_uint64(fd, command, "Number Of Inputs");
			break;

		case REVISOR_ALLOCATE_INPUT:
			result = handle_returned_uint64(fd, command, "Allocated Input ID");
			break;

		case REVISOR_CHECKOUT_TEST:
		case REVISOR_UNLOAD_TEST:
		case REVISOR_TRACE:
		case REVISOR_CLEAR_ALL_INPUTS:
			result = ioctl(fd, command);
			break;

		default:
			printf("Missing arguments for command!\n");
			result = -2;

	}

	return result;
}

static int serve_numerical_command_with_argument(int fd, int command, uint64_t argument) {

	int result = 0;

	printf("Argument: 0x%lx\n", argument);
	switch(command) {
		case REVISOR_CHECKOUT_INPUT:
		case REVISOR_FREE_INPUT:
			result = ioctl(fd, command, &argument);
			break;

		default:
			printf("Too many argument were given to command!\n");
			result = -2;
	}

	return result;
}

static int serve_numerical_operation(int fd, int argc, char** argv) {

	int command_number = atoi(argv[2]);

	if (!(1 <= command_number && command_number <= 10)) {

		printf("Invalid command number: %d\n", command_number);
		return EXIT_FAILURE;
	}

	int command = int_to_cmd[command_number];
	int result = 0;
	int err = EXIT_SUCCESS;

	printf("Command: 0x%08x\n", command);
	printf("Command magic: %d\n", _IOC_TYPE(command));
	printf("Command number: %d\n", _IOC_NR(command));


	if (3 < argc) {

		result = serve_numerical_command_with_argument(fd, command, atoi(argv[3]));

	} else {

		result = serve_numerical_command_without_argument(fd, command);

	}

	if (0 <= result) {

		printf("Command executed successfully.\n");
		printf("Result: 0x%x\n", result);

	} else {
		err = EXIT_FAILURE;

		if (-1 == result) {
			perror("IOCTL error");
		}
	}

	return err;

}

static int serve_operation(int fd, int argc, char** argv) {

	int err = EXIT_SUCCESS;

	if(0 == strcmp("w", argv[2]) || 0 == strcmp("r", argv[2])) {

		if(4 > argc) {

			printf("Missing filename!\n");
			err = EXIT_FAILURE;
			goto serve_operation_failure;

		} else if ('w' == argv[2][0]) {

			err = write_operation(fd, argv[3]);

		} else {

			err = read_operation(fd, argv[3]);

		}

	} else if (0 == strcmp("T", argv[2])) {

		err = EXIT_FAILURE; //T_operation(fd);

	} else {

		err = serve_numerical_operation(fd, argc, argv);

	}

serve_operation_failure:

	return err;
}

int main(int argc, char **argv) {

	const char *device = NULL;
	int fd = -1;
	int err = EXIT_SUCCESS;

	if (3 > argc) {

		print_usage(argv[0]);
		err = EXIT_FAILURE;
		goto main_failure;
	}

	device = argv[1];
	printf("Device: %s\n", device);
	fd = open(device, O_RDWR);

	if (0 > fd) {

		perror("Error opening device");
		err = EXIT_FAILURE;
		goto main_failure;
	}

	err = serve_operation(fd, argc, argv);
	close(fd);

main_failure:

	return err;
}

