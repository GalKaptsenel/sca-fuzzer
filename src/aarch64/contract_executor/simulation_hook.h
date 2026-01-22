#ifndef SIMULATION_HOOK_H
#define SIMULATION_HOOK_H

#include "main.h"

int hook_aarch64_instructions(
		const struct simulation_input* sim_input,
		struct sim_code* sc,
		void *hook_addr
);

void base_hook_c(struct cpu_state *state);


#endif // SIMULATION_HOOK_H
