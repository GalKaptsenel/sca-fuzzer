#include "main.h"
#include "simulation.h"

extern void base_hook();
extern void base_hook_end();
extern uint64_t base_hook_c_target;

simulation_hook_fn hooks_to_install[] = {
	stdout_print_hook,
	handle_ret_hook,
};

int main(int argc, char** argv) {

	const char *input_path = (argc > 1) ? argv[1] : "-";  // "-" = stdin

	int ret = 0;

	const size_t base_hook_size = (size_t)(base_hook_end - base_hook);
	base_hook_c_target = (uint64_t)base_hook_c;

//struct simulation simulation = { 0 };

	if (NULL == input_path || 0 == strcmp(input_path, "-")) {
		ret = simulation_input_load_fd(STDIN_FILENO, &simulation.sim_input);
		if (0 > ret) {
			fprintf(stderr, "Failed to load input from STDIN\n");
			goto main_out;
		}
	} else {
		ret = simulation_input_load(input_path, &simulation.sim_input);
		if (0 > ret) {
			fprintf(stderr, "Failed to load input from %s\n", input_path);
			goto main_out;
		}
	}

	ret = simulation_code_init(&simulation.sim_input, &simulation.sim_code);
	if (ret < 0) {
		fprintf(stderr, "Failed to allocate simulation code\n");
		goto main_input_free;
	}

	printf("Loaded test case (%zu bytes)\n", simulation.sim_code.code_size);

	if(RVZR_ARCH_AARCH64 == simulation.sim_input.hdr.arch) {
		hook_aarch64_instructions(&simulation.sim_input, &simulation.sim_code, base_hook, base_hook_size);
	} else {
		fprintf(stderr, "ONLY AARCH64 arch currently supported for simulation\n");
		ret = -1;
		goto main_code_free;
	}

	simulation.simulation_memory = (uint8_t*)malloc(simulation.sim_input.hdr.mem_size);
	if(NULL == simulation.simulation_memory) {
		fprintf(stderr, "was unable to allocate enough memory for sandbox\n");
		ret = -1;
		goto main_code_free;
	}
	memcpy(simulation.simulation_memory, simulation.sim_input.memory, simulation.sim_input.hdr.mem_size);

	simulation.code_tmp = (uint8_t*)malloc(simulation.sim_input.hdr.code_size);
	if(NULL == simulation.code_tmp) {
		fprintf(stderr, "was unable to allocate enough memory for code tmp variable\n");
		ret = -1;
		goto main_sim_mem_free;
	}

	simulation.n_hooks = 0;
	memset(simulation.hooks, 0, sizeof(simulation.hooks));
	simulation.return_address = 0;

	simulation.n_hooks = install_hooks(&simulation, MAX_HOOKS, hooks_to_install, (sizeof(hooks_to_install)/sizeof(hooks_to_install[0])));
	
	asm volatile (
			"adr x9, 1f\n"
			"str x9, %0\n"
			"blr %1\n"
			"1:\n"
			: "=m"(simulation.return_address)
			: "r"(simulation.sim_code.code)
			: "x9", "memory", "cc"
		);

	free(simulation.code_tmp);
main_sim_mem_free:
	free(simulation.simulation_memory);
main_code_free:
	simulation_code_free(&simulation.sim_code);
main_input_free:
	simulation_input_free(&simulation.sim_input);
main_out:
	return ret;
}
