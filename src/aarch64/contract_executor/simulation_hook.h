#ifndef SIMULATION_HOOK_H
#define SIMULATION_HOOK_H

#include <inttypes.h>
#include <stdio.h>
#include <stdbool.h>
#include "instruction_encodings.h"
#include "simulation_state.h"

struct simulation_input;
struct simulation_code;

int hook_aarch64_instructions(
		const struct simulation_input* sim_input,
		struct simulation_code* sc,
		void* hook_addr,
		size_t hook_size
);

void base_hook_c(struct cpu_state *state);
bool out_of_simulation(struct cpu_state* state);

typedef void* (*simulation_hook_fn)(struct simulation_state* sim_state);

void* stdout_print_hook(struct simulation_state* sim_state);
void* handle_ret_hook(struct simulation_state* sim_state);

#endif // SIMULATION_HOOK_H
