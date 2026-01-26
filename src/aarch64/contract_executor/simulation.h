#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdint.h>
#include "simulation_code.h"
#include "simulation_input.h"
#include "simulation_hook.h"

#define MAX_HOOKS (1024)

struct simulation {
	uintptr_t return_address;
	struct simulation_input sim_input;
	struct simulation_code sim_code;
	uint8_t* simulation_memory;
	size_t n_hooks;
	simulation_hook_fn hooks[MAX_HOOKS];
};

extern struct simulation simulation;

ssize_t install_hook(struct simulation* simulation, size_t max_hooks, simulation_hook_fn hook);
ssize_t install_hooks(struct simulation* simulation, size_t max_hooks, simulation_hook_fn* hooks, size_t new_hooks_length);

#endif // SIMULATION_H
