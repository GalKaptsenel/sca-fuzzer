#include <executor_ioctl.h>
#include <executor_pac_api.h>
#include <executor_user_api.h>
#include <executor_batch_format.h>
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
	REVISOR_PAC_SIGN,
	REVISOR_PAC_AUTH,
	REVISOR_PAC_XPAC
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

static int write_all_fd(int fd, const uint8_t* buf, size_t size) {
	size_t total = 0;
	while (total < size) {
		ssize_t n = write(fd, buf + total, size - total);
		if (0 > n) {
			perror("Error writing");
			return -1;
		}
		total += (size_t)n;
	}
	return 0;
}

/* Read a whole stream (e.g. stdin, which has no stat size) into a growable buffer. */
static uint64_t rd_u64(const uint8_t* p);   /* defined with the batch handlers below */

static int read_n_fd(int fd, void* buf, size_t n) {
	size_t got = 0;
	while (got < n) {
		ssize_t r = read(fd, (char*)buf + got, n - got);
		if (0 > r) { perror("Error reading"); return -1; }
		if (0 == r) { return -1; }   /* premature EOF */
		got += (size_t)r;
	}
	return 0;
}

static int append_read(int fd, char** buf, size_t* off, size_t n) {
	char* grown = realloc(*buf, *off + n);
	if (NULL == grown) { return -1; }
	*buf = grown;
	if (0 > read_n_fd(fd, *buf + *off, n)) { return -1; }
	*off += n;
	return 0;
}

static int read_batch_stream(int fd, char** buffer, size_t* size) {
	const size_t HDR = 4 * sizeof(uint64_t), DESC = 2 * sizeof(uint64_t), LEN = sizeof(uint64_t);
	char* buf = NULL;
	size_t off = 0;
	if (0 > append_read(fd, &buf, &off, HDR)) { goto fail; }
	uint64_t n_units = rd_u64((uint8_t*)buf + 2 * sizeof(uint64_t));
	size_t descs_at = off;
	if (0 > append_read(fd, &buf, &off, n_units * DESC)) { goto fail; }
	uint64_t total_inputs = 0, total_bodies = 0;
	for (uint64_t u = 0; u < n_units; ++u) {
		total_bodies += rd_u64((uint8_t*)buf + descs_at + u * DESC);
		total_inputs += rd_u64((uint8_t*)buf + descs_at + u * DESC + sizeof(uint64_t));
	}
	size_t lens_at = off;
	if (0 > append_read(fd, &buf, &off, total_inputs * LEN)) { goto fail; }
	for (uint64_t i = 0; i < total_inputs; ++i) {
		total_bodies += rd_u64((uint8_t*)buf + lens_at + i * LEN);
	}
	if (0 > append_read(fd, &buf, &off, total_bodies)) { goto fail; }
	*buffer = buf;
	*size = off;
	return 0;
fail:
	free(buf);
	return -1;
}

/* Read the batch request from stdin ("-") or a file. */
static int read_batch_input(const char* name, char** buffer, size_t* size) {
	int fd = (0 == strcmp("-", name)) ? STDIN_FILENO : open(name, O_RDONLY);
	if (0 > fd) { fprintf(stderr, "batch: cannot open %s\n", name); return -1; }
	int rc = read_batch_stream(fd, buffer, size);
	if (STDIN_FILENO != fd) { close(fd); }
	return rc;
}

static int write_batch_output(const char* name, const uint8_t* buf, size_t size) {
	if (0 == strcmp("-", name)) {
		return write_all_fd(STDOUT_FILENO, buf, size);
	}
	int fd = open(name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (0 > fd) {
		fprintf(stderr, "Error opening file %s\n", name);
		return -1;
	}
	int rc = write_all_fd(fd, buf, size);
	close(fd);
	return rc;
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

/* Fill `keys` from 10 hex words at argv[base..base+9] (apia_lo apia_hi ... apga_lo apga_hi). */
static void parse_pac_keys(char** argv, int base, struct pac_keys* keys) {
	keys->apia_lo = strtoull(argv[base + 0], NULL, 0);
	keys->apia_hi = strtoull(argv[base + 1], NULL, 0);
	keys->apib_lo = strtoull(argv[base + 2], NULL, 0);
	keys->apib_hi = strtoull(argv[base + 3], NULL, 0);
	keys->apda_lo = strtoull(argv[base + 4], NULL, 0);
	keys->apda_hi = strtoull(argv[base + 5], NULL, 0);
	keys->apdb_lo = strtoull(argv[base + 6], NULL, 0);
	keys->apdb_hi = strtoull(argv[base + 7], NULL, 0);
	keys->apga_lo = strtoull(argv[base + 8], NULL, 0);
	keys->apga_hi = strtoull(argv[base + 9], NULL, 0);
}

static bool is_pac_command(int command) {
	switch (_IOC_NR(command)) {
		case REVISOR_PAC_SIGN_CONSTANT:
		case REVISOR_PAC_AUTH_CONSTANT:
		case REVISOR_PAC_XPAC_CONSTANT:
			return true;
		default:
			return false;
	}
}

/*
 * PAC ioctls carry structs, so they take their own arguments. Sign/auth carry the keys with the
 * request (the kernel keeps no key state); XPAC is key-independent:
 *   11 PAC_SIGN  <mnemonic> <ptr> <ctx> <10 hex keys>
 *   12 PAC_AUTH  <mnemonic> <ptr> <ctx> [--force] <10 hex keys>  (default: XPAC-strip only;
 *                                        --force issues a real AUT* -- a wrong signature faults
 *                                        FEAT-FPAC silicon)
 *   13 PAC_XPAC  <mnemonic> <ptr>
 * The 10 keys are: apia_lo apia_hi apib_lo apib_hi apda_lo apda_hi apdb_lo apdb_hi apga_lo apga_hi
 */
static int serve_pac_command(int fd, int command, int argc, char** argv) {
	switch (_IOC_NR(command)) {
		case REVISOR_PAC_SIGN_CONSTANT: {
			if (16 != argc) {
				printf("Usage: <command> <mnemonic> <ptr> <ctx> <10 hex keys>\n");
				return -2;
			}
			struct pac_sign_req req = { 0 };
			strncpy(req.mnemonic, argv[3], sizeof(req.mnemonic) - 1);
			req.ptr = strtoull(argv[4], NULL, 0);
			req.ctx = strtoull(argv[5], NULL, 0);
			parse_pac_keys(argv, 6, &req.keys);
			req.keys_present = 1;
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
			bool forced = (17 == argc) && (0 == strcmp(argv[6], "--force"));
			int keys_base = forced ? 7 : 6;
			if (16 != argc && !forced) {
				printf("Usage: <command> <mnemonic> <ptr> <ctx> [--force] <10 hex keys>\n");
				return -2;
			}
			struct pac_sign_req req = { 0 };
			req.ptr = strtoull(argv[4], NULL, 0);
			req.ctx = strtoull(argv[5], NULL, 0);
			parse_pac_keys(argv, keys_base, &req.keys);
			req.keys_present = 1;
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

/* ---- batch: measure a whole super-batch in one invocation (executor_batch_format.h) ---------- */

#define BATCH_RESULT_WORDS (1 + NUM_PFC)   /* htrace + pfc[NUM_PFC] per (input, rep) */

static uint64_t rd_u64(const uint8_t* p) { uint64_t v; memcpy(&v, p, sizeof(v)); return v; }
static void wr_u64(uint8_t* p, uint64_t v) { memcpy(p, &v, sizeof(v)); }

/* Load one unit's test case and inputs, trace it n_reps times, and write this unit's results —
 * [input][rep] { htrace, pfc[NUM_PFC] } — into `out`. Fails fast (returns -1) on any device error. */
static int batch_run_unit(int fd, const uint8_t* tc, uint64_t tc_len,
                          const uint8_t** inputs, const uint64_t* input_lens, uint64_t n_inputs,
                          uint64_t n_reps, uint8_t* out) {
	if (0 > ioctl(fd, REVISOR_CHECKOUT_TEST) || (ssize_t)tc_len != write(fd, tc, tc_len)) {
		return -1;
	}
	if (0 > ioctl(fd, REVISOR_CLEAR_ALL_INPUTS)) {
		return -1;
	}

	uint64_t* iids = malloc(n_inputs * sizeof(uint64_t));
	if (NULL == iids && 0 != n_inputs) {
		return -1;
	}
	for (uint64_t i = 0; i < n_inputs; ++i) {
		uint64_t iid = 0;
		if (0 > ioctl(fd, REVISOR_ALLOCATE_INPUT, &iid) ||
		    0 > ioctl(fd, REVISOR_CHECKOUT_INPUT, &iid) ||
		    (ssize_t)input_lens[i] != write(fd, inputs[i], input_lens[i])) {
			free(iids);
			return -1;
		}
		iids[i] = iid;
	}

	for (uint64_t r = 0; r < n_reps; ++r) {
		if (0 > ioctl(fd, REVISOR_TRACE)) {
			free(iids);
			return -1;
		}
		for (uint64_t i = 0; i < n_inputs; ++i) {
			user_measurement_t m = { 0 };
			if (0 > ioctl(fd, REVISOR_CHECKOUT_INPUT, &iids[i]) ||
			    0 > ioctl(fd, REVISOR_MEASUREMENT, &m)) {
				free(iids);
				return -1;
			}
			uint8_t* slot = out + (i * n_reps + r) * BATCH_RESULT_WORDS * sizeof(uint64_t);
			wr_u64(slot, m.htrace[0]);
			for (int k = 0; k < NUM_PFC; ++k) {
				wr_u64(slot + (1 + k) * sizeof(uint64_t), m.pfc[k]);
			}
		}
	}
	free(iids);
	return 0;
}

static int serve_batch_command(int fd, const char* in_file, const char* out_file) {
	char* req = NULL;
	size_t req_size = 0;
	if (0 > read_batch_input(in_file, &req, &req_size)) {
		return EXIT_FAILURE;
	}
	const uint8_t* p = (const uint8_t*)req;

	int err = EXIT_FAILURE;
	uint64_t* tc_len = NULL;
	uint64_t* n_inputs = NULL;
	uint64_t* input_len = NULL;
	uint8_t* res = NULL;

	if (req_size < 4 * sizeof(uint64_t) ||
	    REVISOR_BATCH_REQUEST_MAGIC != rd_u64(p) ||
	    REVISOR_BATCH_VERSION != rd_u64(p + 8)) {
		fprintf(stderr, "batch: bad request header\n");
		goto out;
	}
	uint64_t n_units = rd_u64(p + 16);
	uint64_t n_reps  = rd_u64(p + 24);
	size_t off = 4 * sizeof(uint64_t);

	tc_len = malloc(n_units * sizeof(uint64_t));
	n_inputs = malloc(n_units * sizeof(uint64_t));
	if ((NULL == tc_len || NULL == n_inputs) && 0 != n_units) {
		goto out;
	}
	uint64_t total_inputs = 0;
	for (uint64_t u = 0; u < n_units; ++u) {
		tc_len[u] = rd_u64(p + off);
		n_inputs[u] = rd_u64(p + off + 8);
		off += 2 * sizeof(uint64_t);
		total_inputs += n_inputs[u];
	}

	input_len = malloc(total_inputs * sizeof(uint64_t));
	if (NULL == input_len && 0 != total_inputs) {
		goto out;
	}
	for (uint64_t j = 0; j < total_inputs; ++j) {
		input_len[j] = rd_u64(p + off);
		off += sizeof(uint64_t);
	}

	size_t res_size = 4 * sizeof(uint64_t) + n_units * sizeof(uint64_t) +
	                  total_inputs * n_reps * BATCH_RESULT_WORDS * sizeof(uint64_t);
	res = malloc(res_size);
	if (NULL == res) {
		goto out;
	}
	wr_u64(res, REVISOR_BATCH_RESPONSE_MAGIC);
	wr_u64(res + 8, REVISOR_BATCH_VERSION);
	wr_u64(res + 16, n_units);
	wr_u64(res + 24, n_reps);
	for (uint64_t u = 0; u < n_units; ++u) {
		wr_u64(res + 4 * sizeof(uint64_t) + u * sizeof(uint64_t), n_inputs[u]);
	}
	uint8_t* res_meas = res + 4 * sizeof(uint64_t) + n_units * sizeof(uint64_t);

	size_t input_idx = 0;
	size_t meas_off = 0;
	for (uint64_t u = 0; u < n_units; ++u) {
		const uint8_t* tc = p + off;
		off += tc_len[u];
		const uint8_t** inputs = malloc(n_inputs[u] * sizeof(uint8_t*));
		if (NULL == inputs && 0 != n_inputs[u]) {
			goto out;
		}
		for (uint64_t i = 0; i < n_inputs[u]; ++i) {
			inputs[i] = p + off;
			off += input_len[input_idx + i];
		}
		int rc = batch_run_unit(fd, tc, tc_len[u], inputs, input_len + input_idx,
		                        n_inputs[u], n_reps, res_meas + meas_off);
		free(inputs);
		if (0 != rc) {
			fprintf(stderr, "batch: unit %lu measurement failed\n", u);
			goto out;
		}
		meas_off += n_inputs[u] * n_reps * BATCH_RESULT_WORDS * sizeof(uint64_t);
		input_idx += n_inputs[u];
	}

	if (0 == write_batch_output(out_file, res, res_size)) {
		err = EXIT_SUCCESS;
	}

out:
	free(res);
	free(input_len);
	free(n_inputs);
	free(tc_len);
	free(req);
	return err;
}

static int serve_operation(int fd, int argc, char** argv) {
	if (0 == strcmp("batch", argv[2])) {
		if (5 > argc) {
			printf("Usage: <device> batch <request-file> <result-file>\n");
			return EXIT_FAILURE;
		}
		return serve_batch_command(fd, argv[3], argv[4]);
	}

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
	fd = open(device, O_RDWR);
	if (0 > fd) {
		perror("Error opening device");
		return EXIT_FAILURE;
	}

	err = serve_operation(fd, argc, argv);
	close(fd);
	return err;
}
