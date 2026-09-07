#ifndef SIMULATION_CODE_H
#define SIMULATION_CODE_H

#include <sys/mman.h>
#include <stdint.h>
#include <stddef.h>
#include "simulation_input.h"

struct simulation_code {
    void*	code;  /* RWX: [instructions: code_size][read-only tables: data_size][RET][trampoline] */
    size_t	code_size;  /* instruction bytes — the only region that is hooked/executed */
    size_t	data_size;  /* read-only dispatch-table bytes right after the instructions (never hooked) */
};

int simulation_code_init(const struct simulation_input* sim_input, 
		struct simulation_code *out, size_t additional_space_alloc);

void simulation_code_free(struct simulation_code *code, size_t additional_space_alloc);

#endif // SIMULATION_CODE_H

