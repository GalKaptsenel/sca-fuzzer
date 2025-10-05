#include <executor_ioctl.h>
#include <executor_user_api.h>
#include <executor_utils.h>
#include <trace_writer.h>
#include <buffer.h>

static unsigned long int_to_cmd[] = {
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
	REVISOR_BATCHED_INPUTS,
	REVISOR_GET_AUX_BUFFER
};


static void print_usage(const char *prog_name) {
	printf("Usage: %s <device> <command [argument]|w file|r file| c batch_file> \n", prog_name);
	printf("\t<device>  : Path to the device (e.g., /dev/revizor_device)\n");
	printf("\t<command [arguemnt]|w file|r file> :\n");
	printf("\t		- IOCTL command number (integer)\n");
	printf("\t			[argument]: Optional argument for the IOCTL command (integer or hex)\n");
	printf("\t		- w for writing to the device\n");
	printf("\t			file: file with data to write to the device\n");
	printf("\t		- r for reading from the device\n");
	printf("\t			file: file to read device contents into\n");
	printf("\t		- c for uploading a batch file\n");
	printf("\t			btach_file: file containing the description of the batch\n");
	printf("\tExample: %s /dev/revizor_device 3 0x100\n", prog_name);
	printf("\tExample: %s /dev/revizor_device w /my/src/file/name\n", prog_name);
	printf("\tExample: %s /dev/revizor_device r /my/dst/file/name\n", prog_name);
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
		char buf[1000] = { 0 };
		snprintf(buf, sizeof(buf), "Error getting file size of: %s", file_path);
		perror(buf);
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

static void buffer_bits64(uint64_t u, char* buff) {
	buff[64] = 0;
	for (int i = 63; i >= 0; i--) {
		buff[i] = '0' + (u & 1);
		u >>= 1;
	}
}

static void print_bits64(uint64_t u) {
	char str[65] = { 0 };
	buffer_bits64(u, str);
	printf("%s", str);
}

static int handle_get_measurement(int fd, int command) {
	user_measurement_t measurement = { 0 };
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

struct buffer_t* get_aux_buffer(int fd) {
	if(0 > fd) return NULL;

	struct aux_buffer_ioctl req = {0};

	req.size = 0;
	req.data = NULL;

	if(0 > ioctl(fd, REVISOR_GET_AUX_BUFFER, &req)) {
		perror("ioctl REVISOR_GET_AUX_BUFFER (size query)");
		return NULL;
	}

	if(0 == req.size) return NULL;

	struct buffer_t* auxb = buffer_alloc(req.size);
	if(NULL == auxb) return NULL;

	req.data = auxb->addr;

	if(0 > ioctl(fd, REVISOR_GET_AUX_BUFFER, &req)) {
		perror("ioctl REVISOR_GET_AUX_BUFFER (data fetch)");
		buffer_free(auxb);
		return NULL;
	}

	auxb->data_size = req.size;
	return auxb;
}

static int handle_print_aux_buffer(int fd, size_t offset, size_t length) {
	struct buffer_t* auxb = get_aux_buffer(fd);
	if(NULL == auxb) {
		return -1;
	}

	buffer_dump_range(auxb, offset, length);
	buffer_free(auxb);
	return 0;
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

		file_fd  = open(filename, O_WRONLY | O_CREAT, 0644);

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

	switch (_IOC_NR(command)) {
		case REVISOR_GET_TEST_LENGTH_CONSTANT:
			
			result = handle_returned_uint64(fd, command, "Test Length");
			break;

		case REVISOR_MEASUREMENT_CONSTANT:
			result = handle_get_measurement(fd, command);
			break;

		case REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT:
			result = handle_returned_uint64(fd, command, "Number Of Inputs");
			break;

		case REVISOR_ALLOCATE_INPUT_CONSTANT:
			result = handle_returned_uint64(fd, command, "Allocated Input ID");
			break;

		case REVISOR_CHECKOUT_TEST_CONSTANT:
		case REVISOR_UNLOAD_TEST_CONSTANT:
		case REVISOR_TRACE_CONSTANT:
		case REVISOR_CLEAR_ALL_INPUTS_CONSTANT:
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

	switch(_IOC_NR(command)) {
		case REVISOR_CHECKOUT_INPUT_CONSTANT:
		case REVISOR_FREE_INPUT_CONSTANT:
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

	if (!(1 <= command_number && command_number <= 12)) {

		printf("Invalid command number: %d\n", command_number);
		return EXIT_FAILURE;
	}

	int command = int_to_cmd[command_number];
	int result = 0;
	int err = EXIT_SUCCESS;

	printf("Command: 0x%08x\n", command);
	printf("Command magic: %d\n", _IOC_TYPE(command));
	printf("Command number: %d\n", _IOC_NR(command));

	if(REVISOR_GET_AUX_BUFFER_CONSTANT == _IOC_NR(command)) {
		size_t offset = (4 <= argc) ? (size_t)atoi(argv[3]) : 0;
		size_t length = (5 <= argc) ? (size_t)atoi(argv[5]) : (size_t)-1;
		result = handle_print_aux_buffer(fd, offset, length);

	} else if (3 < argc) {

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

static int scenario_operation(int fd, 
		const char* input_paths[], 
		const unsigned int number_of_inputs, 
		const char* test_paths[], 
		const unsigned int number_of_tests, 
		const unsigned int repeats, 
		const char* output_filename) {

	int result = 0;
	int64_t* batch = (int64_t*)malloc(sizeof(int64_t) * number_of_inputs);

	if(NULL == batch) {
		result = -1;
		goto T_return;
	}

	result = serve_numerical_command_without_argument(fd, REVISOR_CLEAR_ALL_INPUTS); 
	if(0 >  result) {
		goto T_free_batch;
	}

	uint64_t total_number_of_traces = number_of_inputs * number_of_tests * repeats;

	struct trace_json* traces = (struct trace_json*)malloc(sizeof(struct trace_json) * total_number_of_traces);
	if(NULL == traces) {
		result = -2;
		goto T_free_batch;
	}

	for(unsigned int i = 0; i < total_number_of_traces; ++i) {
		init_trace_json(traces + i);
		traces[i].order = i;
	}

	cJSON** cjsons= (cJSON**)calloc(total_number_of_traces, sizeof(cJSON*));
	if(NULL == cjsons) {
		result = -3;
		goto T_free_traces;
	}

	for(unsigned int i = 0; i < number_of_inputs; ++i) {
		ioctl(fd, REVISOR_ALLOCATE_INPUT, batch + i);
		if(0 >  batch[i]) {
			result = -4;
			goto T_clear_all_inputs;
		}

	}

	for(unsigned int i = 0; i < number_of_inputs; ++i) {
		result = serve_numerical_command_with_argument(fd, REVISOR_CHECKOUT_INPUT, batch[i]); 
		if(0 >  result) {
			goto T_clear_all_inputs;
		}

		result = write_operation(fd, input_paths[i]);
		if(EXIT_FAILURE == result) {
			goto T_clear_all_inputs;
		}

		for(unsigned int j = 0; j < number_of_tests*repeats; ++j) {
			traces[i + j*number_of_inputs].iid = batch[i];
			traces[i + j*number_of_inputs].input_name = input_paths[i];
		}
	}

	for(unsigned int i = 0; i < repeats; ++i) {
		for(unsigned int j = 0 ; j < number_of_tests; ++j) {

			result = serve_numerical_command_without_argument(fd, REVISOR_CHECKOUT_TEST); 
			if(0 >  result) {
				goto T_unload_test_case;
			}

			result = write_operation(fd, test_paths[j]);
			if(EXIT_FAILURE == result) {
				printf("FAILED to load test. Result is: %u\n", result);
				goto T_unload_test_case;
			}

			result = serve_numerical_command_without_argument(fd, REVISOR_TRACE); 
			if(0 >  result) {
				goto T_unload_test_case;
			}

			for(unsigned int k = 0; k < number_of_inputs; ++k) {

				uint64_t index = (number_of_inputs*number_of_tests)*i + number_of_inputs*j + k;
				traces[index].test_name = test_paths[j];

				result = serve_numerical_command_with_argument(fd, REVISOR_CHECKOUT_INPUT, batch[k]); 
				if(0 >  result) {
					goto T_unload_test_case;
				}

				user_measurement_t measurement = { 0 };
				result = ioctl(fd, REVISOR_MEASUREMENT, &measurement);
				if(0 >  result) {
					goto T_unload_test_case;
				}

				for(unsigned int t = 0; t < HTRACE_WIDTH; ++t) {
					buffer_bits64(measurement.htrace[t], traces[index].hwtraces[t]);
				}
				
				for(unsigned int t = 0; t < NUM_PFC; ++t) {
					traces[index].pfcs[t] = measurement.pfc[t];
				}

				for(unsigned int t = 0; t < WIDTH_MEMORY_IDS; ++t) {
					buffer_bits64(measurement.memory_ids_bitmap[t], traces[index].memory_ids_bitmap[t]);
				}

				traces[index].aux_buffer = get_aux_buffer(fd);

				cjsons[index] = build_trace_json(traces + index);
				if(NULL == cjsons[index]) {
					result = -5;
					goto T_unload_test_case;
				}
			}
		}
	}

	if(NULL != output_filename) {
		if(0 > dump_jsons(output_filename, (const cJSON**)cjsons, total_number_of_traces)) {
			result = -6;
		}
	}

	for(uint64_t i = 0; i < total_number_of_traces; ++i) {
		cJSON_Delete(cjsons[i]);
	}
	
T_unload_test_case:
	serve_numerical_command_without_argument(fd, REVISOR_UNLOAD_TEST);
T_clear_all_inputs:
	serve_numerical_command_without_argument(fd, REVISOR_CLEAR_ALL_INPUTS);
	free(cjsons);
T_free_traces:
	for(unsigned int i = 0; i < total_number_of_traces; ++i) {
		if(traces[i].aux_buffer) {
			buffer_free(traces[i].aux_buffer);
			traces[i].aux_buffer = NULL;
		}
		release_trace_json(traces + i);
	}
	free(traces);
T_free_batch:
	free(batch);
T_return:
	return result;

}

static bool is_blank_line(const char* line) {
	while(*line) {
		if(!isspace((unsigned char)*line)) {
			return false;
		}

		++line;
	}

	return true;
}

#define MAX_LINE_LENGTH (1024)
#define INITIAL_CAPACITY (16)

static char** read_section_lines(const char* configuration_path, const char* section_name, unsigned int* line_count) {

	char** lines = NULL;

	if(NULL == configuration_path || NULL == section_name) return NULL;

	FILE* file = fopen(configuration_path, "r");
	if(NULL == file) {
		perror("Failed to open configuration file");
		goto read_section_return;
	}

	if(MAX_LINE_LENGTH - 3 < strlen(section_name)) {
		// Section name must start and end with '<' and '>' symbols
		fprintf(stderr, "section_name must be at length at most: %u", MAX_LINE_LENGTH - 2);
		goto read_section_fclose;
	}

	char line[MAX_LINE_LENGTH] = { 0 };
	char section_header[MAX_LINE_LENGTH] = { 0 };
	snprintf(section_header, sizeof(section_header), "<%s>", section_name);

	unsigned int capacity = INITIAL_CAPACITY;
	unsigned int count = 0;
	bool is_section = false;

	lines = (char**)malloc(sizeof(char*) * capacity);
	if(NULL == lines) {
		perror("Failed to alocate memory for lines");
		goto read_section_fclose;

	}

	lines[count] = NULL;

	while(fgets(line, sizeof(line), file)) {
		line[strcspn(line, "\r\n")] = 0;
		 
		if(!is_section) {
			is_section = (0 == strcmp(line, section_header));
		} else {
			if('<' == line[0] && '>' == line[strlen(line) - 1]) break;

			if(is_blank_line(line)) continue;

			if(count + 1 >= capacity) { // + 1 for NULL terminator in the send of array
				capacity *= 2;
				char** temp = realloc(lines, sizeof(char*) * capacity);
				if(NULL == temp) {
					perror("Failed to realocate memory for lines");
					goto read_section_free_lines;
				}

				lines = temp;
			}

			lines[count] = strdup(line);
			if(NULL == lines[count]) {
				perror("Failed to copy line from file");
				goto read_section_free_lines;
			}
			++count;
		}

	}

	lines[count] = NULL;
	if(NULL != line_count) {
		*line_count = count;
	}

	goto read_section_fclose;

read_section_free_lines:
	for(unsigned int i = 0; i < count; ++i) free(lines[i]);
	free(lines);
	lines = NULL;

read_section_fclose:
	fclose(file);
read_section_return:
	return lines;
}

static int configurable_operation(int fd, const char* configuration) {
	int err = EXIT_SUCCESS;
	if(NULL == configuration) {
		err = EXIT_FAILURE;
		printf("Error configurable_operation\n");
		goto configurable_operation_return;
	}

	unsigned int input_count = 0;
	const char** input_paths = (const char**)read_section_lines(configuration, "inputs", &input_count);
	if(NULL == input_paths) {
		err = EXIT_FAILURE;
		goto configurable_operation_return;
	}

	unsigned int test_count = 0;
	const char** test_paths = (const char**)read_section_lines(configuration, "tests", &test_count);
	if(NULL == test_paths) {
		err = EXIT_FAILURE;
		goto configurable_operation_free_inputs;
	}

	unsigned int repeats_count = 0;
	const char** repeats_strings = (const char**)read_section_lines(configuration, "repeats", &repeats_count);
	if(NULL == repeats_strings) {
		err = EXIT_FAILURE;
		goto configurable_operation_free_tests;
	}

	int repeats = 1;
	if(0 < repeats_count) {
		repeats = atoi(repeats_strings[0]);
		if(repeats < 0) {
			err = EXIT_FAILURE;
			goto configurable_operation_free_repeats;
		}
	}

	unsigned int output_filename_count = 0;
	const char** output_paths = (const char**)read_section_lines(configuration, "output", &output_filename_count);
	if(NULL == output_paths) {
		err = EXIT_FAILURE;
		goto configurable_operation_free_repeats;
	}

	const char* output_filename = NULL;
	if(0 < output_filename_count) {
		output_filename = output_paths[0];
	}

	err = scenario_operation(fd, input_paths, input_count, test_paths, test_count, (unsigned int)repeats, output_filename);

	free(output_paths);
configurable_operation_free_repeats:
	free(repeats_strings);
configurable_operation_free_tests:
	free(test_paths);
configurable_operation_free_inputs:
	free(input_paths);
configurable_operation_return:
	return err;
}

static int serve_operation(int fd, int argc, char** argv) {

	int err = EXIT_SUCCESS;

	if(0 == strcmp("w", argv[2]) || 0 == strcmp("r", argv[2]) || 0 == strcmp("c", argv[2])) {

		if(4 > argc) {

			printf("Missing filename!\n");
			err = EXIT_FAILURE;
			goto serve_operation_failure;

		} else if ('w' == argv[2][0]) {

			err = write_operation(fd, argv[3]);

		} else if ('r' == argv[2][0]) {

			err = read_operation(fd, argv[3]);

		} else {

			err = configurable_operation(fd, argv[3]);

		}

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

