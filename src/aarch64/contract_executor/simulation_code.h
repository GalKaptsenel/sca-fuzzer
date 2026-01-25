#ifndef SIMULATION_CODE_H
#define SIMULATION_CODE_H

#include <sys/mman.h>
#include <stdint.h>
#include <stddef.h>
#include "simulation_input.h"

struct simulation_code {
    void*	code;  /* RWX */
    size_t	code_size;
};

int simulation_code_init(const struct simulation_input* sim_input,
		struct simulation_code *out);

void simulation_code_free(struct simulation_code *code);

#endif // SIMULATION_CODE_H

