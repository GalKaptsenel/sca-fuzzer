#include <stdbool.h>
#include <errno.h>
#include "simulation_input.h"
#include "userapi/executor_input_format.h"

static int read_full(int fd, void* buf, size_t size) {
	uint8_t *p = buf;
	size_t off = 0;

	while (off < size) {
		ssize_t n = read(fd, p + off, size - off);
		if (0 >= n) {
			return -1;
		}

		off += (size_t)n;
	}

	return 0;
}

int simulation_input_validate_header(const struct input_header* hdr) {
	if (NULL == hdr) return -1;

	if (RVZRCE_MAGIC != hdr->magic) return -1;

	if (RVZRCE_VERSION != hdr->version) return -1;

	if (RVZR_ARCH_X86_64 != hdr->arch &&
	    RVZR_ARCH_AARCH64 != hdr->arch) {
		return -1;
	}

	if (0 != hdr->reserved) return -1;

	if ((hdr->flags & RVZR_FLAG_HAS_CODE) && 0 == hdr->code_size) {
		return -1;
	}

	if ((hdr->flags & RVZR_FLAG_HAS_INPUT) && 0 == hdr->input_init_size) {
		return -1;
	}

	if (MAX_PAYLOAD_SIZE < hdr->code_size || MAX_PAYLOAD_SIZE < hdr->input_init_size) {
		return -1;
	}

	return 0;
}

size_t simulation_input_payload_size(const struct input_header* hdr) {
	if(NULL == hdr) return 0;
	return hdr->code_size + hdr->input_init_size;
}

/* Parse the shared input initialization (executor_input_format) into sim_input: memory = main || faulty,
 * regs = gpr, plus optional MTE tags (unpacked) and PAC keys. Returns 0, or -1 on a malformed input_init
 * or a missing/mis-sized required section. */
static int parse_input_init(const uint8_t* input_init, size_t input_init_size, struct simulation_input* sim_input) {
	const struct revisor_input_section* sec;
	const struct revisor_input_section* main_sec;
	const struct revisor_input_section* faulty_sec;
	const struct revisor_input_section* gpr_sec;

	if (!revisor_input_header_valid(input_init, input_init_size)) {
		return -1;
	}

	main_sec   = revisor_input_find_section(input_init, REVISOR_SEC_MEMORY_MAIN);
	faulty_sec = revisor_input_find_section(input_init, REVISOR_SEC_MEMORY_FAULTY);
	gpr_sec    = revisor_input_find_section(input_init, REVISOR_SEC_GPR);
	if (NULL == main_sec || NULL == faulty_sec || NULL == gpr_sec) {
		return -1;
	}

	/* memory image is the contiguous main || faulty span. */
	sim_input->mem_size = main_sec->length + faulty_sec->length;
	sim_input->memory = malloc(sim_input->mem_size);
	if (NULL == sim_input->memory) {
		return -1;
	}
	memcpy(sim_input->memory, input_init + main_sec->offset, main_sec->length);
	memcpy(sim_input->memory + main_sec->length, input_init + faulty_sec->offset, faulty_sec->length);

	sim_input->regs_size = gpr_sec->length;
	sim_input->regs = malloc(sim_input->regs_size);
	if (NULL == sim_input->regs) {
		return -1;
	}
	memcpy(sim_input->regs, input_init + gpr_sec->offset, gpr_sec->length);

	/* optional MTE tags: two 4-bit tags per byte, low nibble first; one tag per 16B granule. */
	sec = revisor_input_find_section(input_init, REVISOR_SEC_MTE_TAGS);
	if (NULL != sec) {
		size_t count = sim_input->mem_size / 16;
		const uint8_t* packed = input_init + sec->offset;
		if ((count + 1) / 2 != sec->length) {
			return -1;
		}
		sim_input->mte_tags = malloc(count);
		if (NULL == sim_input->mte_tags) {
			return -1;
		}
		for (size_t i = 0; i < count; ++i) {
			uint8_t byte = packed[i / 2];
			sim_input->mte_tags[i] = (0 == (i & 1)) ? (byte & 0xF) : (byte >> 4);
		}
		sim_input->mte_tag_count = count;
	}

	/* optional per-input PAC keys. */
	sec = revisor_input_find_section(input_init, REVISOR_SEC_PAC_KEYS);
	if (NULL != sec) {
		if (sizeof(sim_input->pac_keys) != sec->length) {
			return -1;
		}
		memcpy(sim_input->pac_keys, input_init + sec->offset, sizeof(sim_input->pac_keys));
		sim_input->pac_keys_present = true;
	}

	return 0;
}

int simulation_input_load_fd(int fd, struct simulation_input* sim_input) {
	uint8_t* input_init = NULL;

	memset(sim_input, 0, sizeof(*sim_input));

	if (0 > read_full(fd, &sim_input->hdr, sizeof(sim_input->hdr))) {
		goto load_fd_fail;
	}

	if (0 > simulation_input_validate_header(&sim_input->hdr)) {
		goto load_fd_fail;
	}

	if (sim_input->hdr.flags & RVZR_FLAG_HAS_CODE) {
		sim_input->code = malloc(sim_input->hdr.code_size);
		if (NULL == sim_input->code) {
			goto load_fd_fail;
		}

		if (0 > read_full(fd, sim_input->code, sim_input->hdr.code_size)) {
			goto load_fd_fail;
		}
	}

	if (sim_input->hdr.flags & RVZR_FLAG_HAS_INPUT) {
		input_init = malloc(sim_input->hdr.input_init_size);
		if (NULL == input_init) {
			goto load_fd_fail;
		}

		if (0 > read_full(fd, input_init, sim_input->hdr.input_init_size)) {
			goto load_fd_fail;
		}

		if (0 > parse_input_init(input_init, sim_input->hdr.input_init_size, sim_input)) {
			goto load_fd_fail;
		}
	}

	free(input_init);
	return 0;

load_fd_fail:
	free(input_init);
	simulation_input_free(sim_input);
	return -1;
}

int simulation_input_load_path(const char* path, struct simulation_input* sim_input) {
	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		return -1;
	}

	int ret = simulation_input_load_fd(fd, sim_input);
	close(fd);
	return ret;
}

static bool safe_file_read(FILE* f, void* buff, size_t size) {
	size_t result = 0;
	while(true) {
		result = fread(buff, size, 1, f);
		if(1 == result) {
		       	return true;
		}
		if(ferror(f) && EINTR == errno) {
			clearerr(f);
			continue;
		} else if(feof(f)) {
			return false;
		}
		return false;
	}
}

static int receive_payload_from_file(FILE* f, void* payload, size_t buffer_size) {
	if(NULL == f || NULL == payload) return -1;

	struct header hdr = { 0 };
	if(!safe_file_read(f, &hdr, sizeof(hdr))) {
		return -1;
	}

	if(0 == hdr.length) {
		return 0;
	}

	if(buffer_size < hdr.length) {
		return -2;
	}

	if(!safe_file_read(f, payload, hdr.length)) {
		return -1;
	}

	return hdr.length;
}

static uint8_t tmp_buffer[MAX_PAYLOAD_SIZE] = { 0 };
int simulation_input_from_file(FILE* f, struct simulation_input* sim_input) {
	if(NULL == f || NULL == sim_input) return -1;

	int ret = receive_payload_from_file(f, tmp_buffer, sizeof(tmp_buffer));
	if(ret <= 0) return ret;
	size_t payload_len = (size_t)ret;

	if (payload_len < sizeof(sim_input->hdr)) {
		fprintf(stderr, "Payload smaller than header!\n");
		return -1;
	}

	const uint8_t* current_ptr = tmp_buffer;
	memset(sim_input, 0, sizeof(*sim_input));
	memcpy(&sim_input->hdr, current_ptr, sizeof(sim_input->hdr));
	current_ptr += sizeof(sim_input->hdr);

	if (0 > simulation_input_validate_header(&sim_input->hdr)) {
		fprintf(stderr, "Input validation failed!\n");
		ret = -1;
		goto simulation_input_from_file_err;
	}

	if (payload_len < sizeof(sim_input->hdr) + sim_input->hdr.code_size + sim_input->hdr.input_init_size) {
		fprintf(stderr, "Payload truncated!\n");
		ret = -1;
		goto simulation_input_from_file_err;
	}

	if (sim_input->hdr.flags & RVZR_FLAG_HAS_CODE) {
		sim_input->code = malloc(sim_input->hdr.code_size);
		if (NULL == sim_input->code) {
			ret = -1;
			goto simulation_input_from_file_err;
		}

		memcpy(sim_input->code, current_ptr, sim_input->hdr.code_size);
		current_ptr += sim_input->hdr.code_size;
	}

	if (sim_input->hdr.flags & RVZR_FLAG_HAS_INPUT) {
		if (0 > parse_input_init(current_ptr, sim_input->hdr.input_init_size, sim_input)) {
			fprintf(stderr, "Malformed input initialization!\n");
			ret = -1;
			goto simulation_input_from_file_err;
		}
		current_ptr += sim_input->hdr.input_init_size;
	}

	goto simulation_input_from_file_success;

simulation_input_from_file_err:
	simulation_input_free(sim_input);
simulation_input_from_file_success:
	memset(tmp_buffer, 0, payload_len);
	current_ptr = NULL;

	return ret;
}

void simulation_input_free(struct simulation_input* sim_input) {
	if (NULL == sim_input) {
		return;
	}

	free(sim_input->code);
	free(sim_input->memory);
	free(sim_input->regs);
	free(sim_input->mte_tags);

	memset(sim_input, 0, sizeof(*sim_input));
}
