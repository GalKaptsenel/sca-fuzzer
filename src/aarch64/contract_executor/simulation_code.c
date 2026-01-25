#include "simulation_code.h"

int simulation_code_init(const struct simulation_input* sim_input,
		struct simulation_code *out) {
	memset(out, 0, sizeof(*out));

	out->code_size = sim_input->hdr.code_size;

	/* RWX simulation code */
	out->code = mmap(
			NULL,
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

