#include "main.h"


static int sim_code_init(const struct simulation_input* sim_input,
                         struct sim_code *out) {
	memset(out, 0, sizeof(*out));

	out->code_size = sim_input->hdr.code_size;

	/* RWX simulation code */
	out->simulation_code = mmap(
			NULL,
			out->code_size,
			PROT_READ | PROT_WRITE | PROT_EXEC,
			MAP_PRIVATE | MAP_ANONYMOUS,
			-1,
			0
		);

	if (MAP_FAILED == out->simulation_code) {
		return -1;
	}

    return 0;
}

void sim_code_free(struct sim_code *code) {
	if (NULL == code) return;

	if (NULL != code->simulation_code) {
		munmap(code->simulation_code, code->code_size);
	}

	code->simulation_code = NULL;
	code->code_size = 0;
}

