#include "simulation.h"
#include "simulation_hook.h"

struct simulation simulation = { 0 };

ssize_t install_hook(struct simulation* simulation, size_t max_hooks, simulation_hook_fn hook) {
	if(NULL == simulation || NULL == hook) return -1;
	if(simulation->n_hooks + 1 > max_hooks) return -1;
	simulation->hooks[simulation->n_hooks] = hook;
	++simulation->n_hooks;
	return simulation->n_hooks;
}

ssize_t install_hooks(struct simulation* simulation, size_t max_hooks, simulation_hook_fn* hooks, size_t new_hooks_length) {
	if(NULL == simulation || NULL == hooks) return -1;
	size_t current_length = simulation->n_hooks;
	for(size_t i = 0; i < max_hooks && i < new_hooks_length; ++i) {
		if(-1 == install_hook(simulation, max_hooks, hooks[i])) {
			fprintf(stderr, "Unable to install hook!\n");
			simulation->n_hooks = current_length;
			for(size_t j = current_length; j < simulation->n_hooks; ++j) simulation->hooks[j] = 0;
			return -1;
		}
	}

	return simulation->n_hooks;
}


