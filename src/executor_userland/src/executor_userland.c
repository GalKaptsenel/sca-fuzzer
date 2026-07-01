#include <executor_ioctl.h>
#include <executor_pac_api.h>
#include <executor_mte_api.h>
#include <executor_user_api.h>
#include <executor_utils.h>

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
	REVISOR_SET_PAC_KEYS,
	REVISOR_GET_PAC_KEYS,
	REVISOR_PAC_SIGN,
	REVISOR_PAC_AUTH,
	REVISOR_PAC_XPAC,
	REVISOR_MTE_TAG_REGION
};

#define MAX_COMMAND_NUMBER ((int)(sizeof(int_to_cmd) / sizeof(int_to_cmd[0])) - 1)

static void print_usage(const char* prog_name) {
	printf("Usage: %s <device> <command [argument] | w file | r file>\n", prog_name);
	printf("\t<device>            : path to the device (e.g. /dev/executor)\n");
	printf("\t<command> [argument]: ioctl command number (1-%d), optional integer/hex argument\n", MAX_COMMAND_NUMBER);
	printf("\tw file              : write the file's contents to the device\n");
	printf("\tr file              : read the device's contents into the file\n");
	printf("\tPAC: 11 [10 hex keys|none], 12, 13|14 <mnemonic> <ptr> <ctx>, 15 <mnemonic> <ptr>\n");
	printf("\tExample: %s /dev/executor 4 0x3\n", prog_name);
	printf("\tExample: %s /dev/executor 11 0x1 0x0 0x2 0x0 0x3 0x0 0x4 0x0 0x5 0x0\n", prog_name);
	printf("\tExample: %s /dev/executor 11        (revert to live kernel keys)\n", prog_name);
	printf("\tExample: %s /dev/executor 13 pacia 0xffff000080000000 0x0\n", prog_name);
	printf("\tExample: %s /dev/executor w input.bin\n", prog_name);
	printf("\tExample: %s /dev/executor r dump.bin\n", prog_name);
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
	if (*buffer) {
		free(*buffer);
		*buffer = NULL;
	}
	free(temp_buffer);
read_device_exit:
	return -1;
}

static int read_file(const char* file_path, char** buffer, size_t* size) {
	struct stat st = { 0 };
	int fd = -1;

	if (0 > stat(file_path, &st)) {
		perror(file_path);
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

	size_t total = 0;
	while (total < *size) {
		ssize_t n = read(fd, *buffer + total, *size - total);
		if (0 > n) {
			perror("Error reading file");
			goto read_file_cleanup_close_file;
		}
		if (0 == n) {  // EOF before the stat'd size
			fprintf(stderr, "%s: short read (%zu of %zu bytes)\n", file_path, total, *size);
			goto read_file_cleanup_close_file;
		}
		total += (size_t)n;
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
	uint64_t value = 0;
	int result = ioctl(fd, command, &value);
	if (0 <= result) {
		printf("%s: %lu\n", prompt, value);
	}

	return result;
}

static void buffer_bits64(uint64_t u, char* buff) {
	buff[64] = 0;
	for (int i = 63; i >= 0; --i) {
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

		for (int i = 0; i < HTRACE_WIDTH; ++i) {
			printf("\thtrace %d: ", i);
			print_bits64(measurement.htrace[i]);
		}

		printf("\n");

		for (int i = 0; i < NUM_PFC; ++i) {
			printf("\tpfc %d: %lu\n", i, measurement.pfc[i]);
		}

		printf("\n");
	}

	return result;
}

static int write_operation(int fd, const char* filename) {
	char* file_data = NULL;
	size_t file_size = 0;
	int err = EXIT_SUCCESS;

	if (0 > read_file(filename, &file_data, &file_size)) {
		return EXIT_FAILURE;
	}

	printf("File loaded: %s (%zu bytes)\n", filename, file_size);

	if (0 > write(fd, file_data, file_size)) {
		perror("Unable to write device");
		err = EXIT_FAILURE;
	}

	free(file_data);
	return err;
}

static int read_operation(int fd, const char* filename) {
	char* file_data = NULL;
	size_t file_size = 0;
	int file_fd = -1;
	int err = EXIT_SUCCESS;

	if (0 != read_device(fd, &file_data, &file_size)) {
		perror("Unable to read");
		return EXIT_FAILURE;
	}

	printf("%zu bytes were loaded from device\n", file_size);

	file_fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (0 > file_fd) {
		fprintf(stderr, "Error opening file %s\n", filename);
		free(file_data);
		return EXIT_FAILURE;
	}

	if (0 > write(file_fd, file_data, file_size)) {
		fprintf(stderr, "Unable to write into %s\n", filename);
		err = EXIT_FAILURE;
	}

	free(file_data);
	close(file_fd);
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

	switch (_IOC_NR(command)) {
		case REVISOR_CHECKOUT_INPUT_CONSTANT:
		case REVISOR_FREE_INPUT_CONSTANT:
			result = ioctl(fd, command, &argument);
			break;

		default:
			printf("Too many arguments were given to command!\n");
			result = -2;
	}

	return result;
}

static void print_pac_keys(const struct pac_keys* k) {
	printf("\tAPIA %016lx:%016lx\n", k->apia_hi, k->apia_lo);
	printf("\tAPIB %016lx:%016lx\n", k->apib_hi, k->apib_lo);
	printf("\tAPDA %016lx:%016lx\n", k->apda_hi, k->apda_lo);
	printf("\tAPDB %016lx:%016lx\n", k->apdb_hi, k->apdb_lo);
	printf("\tAPGA %016lx:%016lx\n", k->apga_hi, k->apga_lo);
}

static bool is_pac_command(int command) {
	switch (_IOC_NR(command)) {
		case REVISOR_SET_PAC_KEYS_CONSTANT:
		case REVISOR_GET_PAC_KEYS_CONSTANT:
		case REVISOR_PAC_SIGN_CONSTANT:
		case REVISOR_PAC_AUTH_CONSTANT:
		case REVISOR_PAC_XPAC_CONSTANT:
			return true;
		default:
			return false;
	}
}

/* 16 MTE_TAG_REGION  <sandbox_offset> <tag0> [tag1 ...]  — one 4-bit tag per 16B granule,
 * starting at sandbox_offset bytes from lower_overflow. */
static int serve_mte_command(int fd, int command, int argc, char** argv) {
	if (5 > argc) {
		printf("Usage: <command> <sandbox_offset> <tag0> [tag1 ...]\n");
		return -2;
	}
	uint64_t n = (uint64_t)(argc - 4);
	if (n > MTE_TAG_MAX_GRANULES) {
		printf("too many granules: %lu (max %d)\n", n, MTE_TAG_MAX_GRANULES);
		return -2;
	}
	struct mte_tag_region_req req = { 0 };
	req.sandbox_offset = strtoull(argv[3], NULL, 0);
	req.n_granules = n;
	for (uint64_t i = 0; i < n; ++i) {
		req.tags[i] = (uint8_t)(strtoul(argv[4 + i], NULL, 0) & 0xF);
	}
	int result = ioctl(fd, command, &req);
	if (0 <= result) {
		printf("tagged %lu granule(s) at offset 0x%lx\n", n, req.sandbox_offset);
	}
	return result;
}

/*
 * PAC ioctls carry structs, so they take their own arguments:
 *   11 SET_PAC_KEYS  <10 hex words> | (none) to revert to live keys
 *   12 GET_PAC_KEYS
 *   13 PAC_SIGN      <mnemonic> <ptr> <ctx>
 *   14 PAC_AUTH      <mnemonic> <ptr> <ctx> [--force]  (default: XPAC-strip only;
 *                                             --force issues a real AUT* -- a wrong
 *                                             signature faults FEAT-FPAC silicon)
 *   15 PAC_XPAC      <mnemonic> <ptr>
 */
static int serve_pac_command(int fd, int command, int argc, char** argv) {
	switch (_IOC_NR(command)) {
		case REVISOR_GET_PAC_KEYS_CONSTANT: {
			struct pac_keys keys = { 0 };
			int result = ioctl(fd, command, &keys);
			if (0 <= result) {
				printf("PAC keys (hi:lo):\n");
				print_pac_keys(&keys);
			}
			return result;
		}

		case REVISOR_SET_PAC_KEYS_CONSTANT: {
			if (3 == argc) {
				printf("Reverting to the live kernel PAC keys\n");
				return ioctl(fd, command, NULL);
			}
			if (13 != argc) {
				printf("SET_PAC_KEYS needs 0 args (revert) or 10 hex words: "
				       "apia_lo apia_hi apib_lo apib_hi apda_lo apda_hi apdb_lo apdb_hi apga_lo apga_hi\n");
				return -2;
			}
			struct pac_keys keys = {
				.apia_lo = strtoull(argv[3],  NULL, 0), .apia_hi = strtoull(argv[4],  NULL, 0),
				.apib_lo = strtoull(argv[5],  NULL, 0), .apib_hi = strtoull(argv[6],  NULL, 0),
				.apda_lo = strtoull(argv[7],  NULL, 0), .apda_hi = strtoull(argv[8],  NULL, 0),
				.apdb_lo = strtoull(argv[9],  NULL, 0), .apdb_hi = strtoull(argv[10], NULL, 0),
				.apga_lo = strtoull(argv[11], NULL, 0), .apga_hi = strtoull(argv[12], NULL, 0),
			};
			return ioctl(fd, command, &keys);
		}

		case REVISOR_PAC_SIGN_CONSTANT: {
			if (6 != argc) {
				printf("Usage: <command> <mnemonic> <ptr> <ctx>\n");
				return -2;
			}
			struct pac_sign_req req = { 0 };
			strncpy(req.mnemonic, argv[3], sizeof(req.mnemonic) - 1);
			req.ptr = strtoull(argv[4], NULL, 0);
			req.ctx = strtoull(argv[5], NULL, 0);
			int result = ioctl(fd, command, &req);
			if (0 <= result) {
				printf("%s(0x%016lx, ctx=0x%016lx) = 0x%016lx\n",
				       req.mnemonic, req.ptr, req.ctx, req.result);
			}
			return result;
		}

		case REVISOR_PAC_AUTH_CONSTANT: {
			// A wrong signature makes a real AUT* FPAC-reset the box. Without --force, strip the
			// pointer with XPAC instead (the canonical value a successful AUT* would yield) --
			// non-faulting; issue the real AUT* only when explicitly forced.
			bool forced = (7 == argc) && (0 == strcmp(argv[6], "--force"));
			if (6 != argc && !forced) {
				printf("Usage: <command> <mnemonic> <ptr> <ctx> [--force]\n");
				return -2;
			}
			struct pac_sign_req req = { 0 };
			req.ptr = strtoull(argv[4], NULL, 0);
			req.ctx = strtoull(argv[5], NULL, 0);
			if (!forced) {
				const char* xpac_m = (0 == strncmp(argv[3], "auti", 4)) ? "xpaci"
				                   : (0 == strncmp(argv[3], "autd", 4)) ? "xpacd" : NULL;
				if (NULL == xpac_m) {
					printf("Unknown auth mnemonic %s\n", argv[3]);
					return -2;
				}
				strncpy(req.mnemonic, xpac_m, sizeof(req.mnemonic) - 1);
				int result = ioctl(fd, REVISOR_PAC_XPAC, &req);
				if (0 <= result) {
					printf("PAC_AUTH not forced: %s(0x%016lx) = 0x%016lx "
					       "(canonical; no real AUT* issued -- pass --force to authenticate)\n",
					       xpac_m, req.ptr, req.result);
				}
				return result;
			}
			printf("WARNING: issuing a real AUT*; a wrong signature will reset the box.\n");
			strncpy(req.mnemonic, argv[3], sizeof(req.mnemonic) - 1);
			int result = ioctl(fd, command, &req);
			if (0 <= result) {
				printf("%s(0x%016lx, ctx=0x%016lx) = 0x%016lx\n",
				       req.mnemonic, req.ptr, req.ctx, req.result);
			}
			return result;
		}

		case REVISOR_PAC_XPAC_CONSTANT: {
			if (5 != argc) {
				printf("Usage: <command> <mnemonic> <ptr>\n");
				return -2;
			}
			struct pac_sign_req req = { 0 };
			strncpy(req.mnemonic, argv[3], sizeof(req.mnemonic) - 1);
			req.ptr = strtoull(argv[4], NULL, 0);
			int result = ioctl(fd, command, &req);
			if (0 <= result) {
				printf("%s(0x%016lx) = 0x%016lx\n", req.mnemonic, req.ptr, req.result);
			}
			return result;
		}

		default:
			return -2;
	}
}

static int serve_numerical_operation(int fd, int argc, char** argv) {
	int command_number = atoi(argv[2]);

	if (!(1 <= command_number && command_number <= MAX_COMMAND_NUMBER)) {
		printf("Invalid command number: %d\n", command_number);
		return EXIT_FAILURE;
	}

	int command = int_to_cmd[command_number];
	int result = 0;
	int err = EXIT_SUCCESS;

	printf("Command: 0x%08x\n", command);
	printf("Command magic: %d\n", _IOC_TYPE(command));
	printf("Command number: %d\n", _IOC_NR(command));

	if (is_pac_command(command)) {
		result = serve_pac_command(fd, command, argc, argv);
	} else if (REVISOR_MTE_TAG_REGION_CONSTANT == _IOC_NR(command)) {
		result = serve_mte_command(fd, command, argc, argv);
	} else if (3 < argc) {
		result = serve_numerical_command_with_argument(fd, command, strtoull(argv[3], NULL, 0));
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
	bool is_write = (0 == strcmp("w", argv[2]));
	bool is_read = (0 == strcmp("r", argv[2]));

	if (is_write || is_read) {
		if (4 > argc) {
			printf("Missing filename!\n");
			return EXIT_FAILURE;
		}
		return is_write ? write_operation(fd, argv[3]) : read_operation(fd, argv[3]);
	}

	return serve_numerical_operation(fd, argc, argv);
}

int main(int argc, char** argv) {
	const char* device = NULL;
	int fd = -1;
	int err = EXIT_SUCCESS;

	if (3 > argc) {
		print_usage(argv[0]);
		return EXIT_FAILURE;
	}

	device = argv[1];
	printf("Device: %s\n", device);
	fd = open(device, O_RDWR);
	if (0 > fd) {
		perror("Error opening device");
		return EXIT_FAILURE;
	}

	err = serve_operation(fd, argc, argv);
	close(fd);
	return err;
}
