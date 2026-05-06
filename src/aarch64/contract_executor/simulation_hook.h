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

/* Debug: last CPU state seen by base_hook_c and the original instruction at that PC */
extern volatile struct cpu_state g_last_hook_cpu_state;
extern volatile uint32_t g_last_hook_orig_instr;
void ce_debug_print_last_sim_state(FILE *out);

#endif // SIMULATION_HOOK_H
