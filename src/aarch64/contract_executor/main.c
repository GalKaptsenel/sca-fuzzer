#include "main.h"

int main(int argc, char** argv) {

	const char *input_path = (argc > 1) ? argv[1] : "-";  // "-" = stdin

	int ret = 0;

	if (NULL == input_path || 0 == strcmp(input_path, "-")) {
		ret = simulation_input_load_fd(STDIN_FILENO, &simulation.sim_input);
		if (0 > ret) {
			fprintf(stderr, "Failed to load input from STDIN\n");
			return ret;
		}
	} else {
		ret = simulation_input_load(input_path, &simulation.sim_input);
		if (0 > ret) {
			fprintf(stderr, "Failed to load input from %s\n", input_path);
			return ret;
		}
	}

	ret = sim_code_init(&simulation.sim_input, &simulation.sim_code);
	if (ret < 0) {
		fprintf(stderr, "Failed to allocate simulation code\n");
		simulation_input_free(&simulation.sim_input);
		return ret;
	}

	printf("Loaded test case (%zu bytes)\n", simulation.sim_code.code_size);

	if(RVZR_ARCH_AARCH64 == simulation.sim_input.hdr.arch) {
		hook_aarch64_instructions(&simulation.sim_input, &simulation.sim_code, &base_hook);
	} else {
		fprintf(stderr, "ONLY AARCH64 arch currently supported for simulation\n");
		sim_code_free(&simulation.sim_code);
		simulation_input_free(&simulation.sim_input);
		return -1;
	}

	simulation.simulation_memory = (uint8_t*)malloc(simulation.sim_input.hdr.mem_size);
	if(NULL == simulation.simulation_memory) {
		fprintf(stderr, "was unable to allocate enough memory for sandbox\n");
		sim_code_free(&simulation.sim_code);
		simulation_input_free(&simulation.sim_input);
		return -1;
	}
	memcpy(simulation.simulation_memory, simulation.sim_input.code, simulation.sim_input.hdr.code_size);

	simulation.code_tmp = (uint8_t*)malloc(simulation.sim_input.hdr.code_size);
	if(NULL == simulation.code_tmp) {
		fprintf(stderr, "was unable to allocate enough memory for code tmp variable\n");
		sim_code_free(&simulation.sim_code);
		simulation_input_free(&simulation.sim_input);
		free(simulation.simulation_memory);
		return -1;
	}


	// Example: copy code from canonical buffer to execution memory
	// memcpy(sc.simulation_code, tc.code, sc.code_size);

	// run_simulation(sc.simulation_code, sc.code_size); // your execution function

	sim_code_free(&simulation.sim_code);
	simulation_input_free(&simulation.sim_input);
	free(simulation.simulation_memory);
	free(simulation.code_tmp);

	return 0;
}
