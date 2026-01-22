#ifndef SIM_CODE_H
#define SIM_CODE_H

#include "main.h"

struct sim_code {
    void*	simulation_code;  /* RWX */
    size_t	code_size;
};

int sim_code_init(const struct simulation_input* sim_input,
                                struct sim_code *out);

void sim_code_free(struct sim_code *code);

#endif

