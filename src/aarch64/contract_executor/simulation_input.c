#include "simulation_input.h"

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

/* ============================
 * Validation
 * ============================ */

int simulation_input_validate_header(const struct input_header* hdr) {
	if (NULL == hdr) return -1;
	
	if (RVZR_MAGIC != hdr->magic) return -1;
	
	if (RVZR_VERSION != hdr->version) return -1;
	
	if (RVZR_ARCH_X86_64 != hdr->arch &&
	    RVZR_ARCH_AARCH64 != hdr->arch) {
		return -1;
	}
	
	if (0 != hdr->reserved) return -1;
	
	if ((hdr->flags & RVZR_FLAG_HAS_CODE) && hdr->code_size == 0) {
		return -1;
	}
	
	if ((hdr->flags & RVZR_FLAG_HAS_MEMORY) && hdr->mem_size == 0) {
		return -1;
	}
	
	if ((hdr->flags & RVZR_FLAG_HAS_REGS) && hdr->regs_size == 0) {
		return -1;
	}
	
	return 0;
}

size_t simulation_input_payload_size(const struct input_header* hdr) {
	if(NULL == hdr) return 0;
	return hdr->code_size + hdr->mem_size + hdr->regs_size;
}

/* ============================
 * Loader
 * ============================ */

int simulation_input_load_fd(int fd, struct simulation_input* sim_input) {
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

	if (sim_input->hdr.flags & RVZR_FLAG_HAS_MEMORY) {
		sim_input->memory = malloc(sim_input->hdr.mem_size);
		if (NULL == sim_input->memory) {
			goto load_fd_fail;
		}

		if (0 > read_full(fd, sim_input->memory, sim_input->hdr.mem_size)) {
			goto load_fd_fail;
		}
	}

	if (sim_input->hdr.flags & RVZR_FLAG_HAS_REGS) {
		sim_input->regs = malloc(sim_input->hdr.regs_size);
		if (NULL == sim_input->regs) {
			goto load_fd_fail;
		}

		if (0 > read_full(fd, sim_input->regs, sim_input->hdr.regs_size)) {
			goto load_fd_fail;
		}
	}

	return 0;

load_fd_fail:
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

static uint8_t tmp_buffer[REQ_RING_SIZE] = { 0 };
int simulation_input_load_shm(struct shm_region* shm, struct simulation_input* sim_input) {
	if(NULL == shm || NULL == sim_input) return -1;

	int ret = 0;

        uint32_t type = 0;
        uint32_t len = 0;
        ring_recv(shm, &shm->req, &type, tmp_buffer, &len);

	const uint8_t* current_ptr = tmp_buffer;
	memset(sim_input, 0, sizeof(*sim_input));
	memcpy(&sim_input->hdr, current_ptr, sizeof(sim_input->hdr));
	current_ptr += sizeof(sim_input->hdr);


	if (0 > simulation_input_validate_header(&sim_input->hdr)) {
		fprintf(stderr, "[C] Input validation failed!\n");
		ret = -1;
		goto load_shm_fail;
	}

	if (sim_input->hdr.flags & RVZR_FLAG_HAS_CODE) {
		sim_input->code = malloc(sim_input->hdr.code_size);
		if (NULL == sim_input->code) {
			ret = -1;
			fprintf(stderr, "[C] failed malloc code!\n");
			goto load_shm_fail;
		}

		memcpy(sim_input->code, current_ptr, sim_input->hdr.code_size);
		current_ptr += sim_input->hdr.code_size;
	}

	if (sim_input->hdr.flags & RVZR_FLAG_HAS_MEMORY) {
		sim_input->memory = malloc(sim_input->hdr.mem_size);
		if (NULL == sim_input->memory) {
			fprintf(stderr, "[C] failed malloc memory!\n");
			ret = -1;
			goto load_shm_fail;
		}

		memcpy(sim_input->memory, current_ptr, sim_input->hdr.mem_size);
		current_ptr += sim_input->hdr.mem_size;
	}

	if (sim_input->hdr.flags & RVZR_FLAG_HAS_REGS) {
		sim_input->regs = malloc(sim_input->hdr.regs_size);
		if (NULL == sim_input->regs) {
			fprintf(stderr, "[C] failed malloc regs!\n");
			ret = -1;
			goto load_shm_fail;
		}
		memcpy(sim_input->regs, current_ptr, sim_input->hdr.regs_size);
		current_ptr += sim_input->hdr.regs_size;
	}

	goto load_shm_success;

load_shm_fail:
	simulation_input_free(sim_input);

load_shm_success:
	memset(tmp_buffer, 0, current_ptr - tmp_buffer);
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

	memset(sim_input, 0, sizeof(*sim_input));
}

