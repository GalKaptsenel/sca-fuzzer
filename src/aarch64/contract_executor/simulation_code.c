#include "simulation_code.h"

int simulation_code_init(const struct simulation_input* sim_input,
		struct simulation_code *out) {
	memset(out, 0, sizeof(*out));

	out->code_size = sim_input->hdr.code_size;

	void* req_code_base = NULL;
	if(CONFIG_FLAG_REQ_CODE_BASE_VIRT | sim_input->hdr.config.flags) {
		req_code_base = (void*)sim_input->hdr.config.requested_code_base_virt;
	}

	/* RWX simulation code */
	out->code = mmap(
			req_code_base,
			out->code_size,
			PROT_READ | PROT_WRITE | PROT_EXEC,
			MAP_PRIVATE | MAP_ANONYMOUS,
			-1,
			0
		);

	if (MAP_FAILED == out->code) {
		return -1;
	}

    return 0;
}

void simulation_code_free(struct simulation_code *code) {
	if (NULL == code) return;

	if (NULL != code->code) {
		munmap(code->code, code->code_size);
	}

	code->code = NULL;
	code->code_size = 0;
}

