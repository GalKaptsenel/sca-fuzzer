#include "main.h"
#include "simulation.h"
#include "simulation_execution_clause_hook.h"
#include "tage_py.h"

extern void base_hook();
extern void base_hook_end();
extern uint64_t base_hook_c_target;

simulation_hook_fn hooks_to_install[] = {
	execution_clause_hook,
	stdout_print_hook,
	handle_ret_hook,
};

int main(int argc, char** argv) {
	int ret = 0;
	ret = python_init();
	if(ret < 0) {
		fprintf(stderr, "Failed to init python interpreter\n");
		goto main_out;
	}
	const char* shm_name = (argc > 1) ? argv[1] : NULL;
	struct shm_region* shm = init_shm(shm_name);
	assert(NULL != shm && "init_shm should never return NULL");

	const size_t base_hook_size = (size_t)(base_hook_end - base_hook);
	base_hook_c_target = (uint64_t)base_hook_c;

	while(1) {
		ret = simulation_input_load_shm(shm, &simulation.sim_input);

		if (0 > ret) {
			fprintf(stderr, "Failed to load input from shared memory\n");
			goto main_destroy_shm;
		}

		ret = simulation_code_init(&simulation.sim_input, &simulation.sim_code);
		if (ret < 0) {
			fprintf(stderr, "Failed to allocate simulation code\n");
			goto main_input_free;
		}
	
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
	
		simulation.n_hooks = 0;
		memset(simulation.hooks, 0, sizeof(simulation.hooks));
		simulation.return_address = 0;
	
		simulation.n_hooks = install_hooks(&simulation, MAX_HOOKS, hooks_to_install, (sizeof(hooks_to_install)/sizeof(hooks_to_install[0])));
		
		uint64_t* regs_blob = (uint64_t*)simulation.sim_input.regs;
	
		asm volatile (
				"mov x29, %2\n"
				"adr x9, 1f\n"
				"str x9, %0\n"
				"mov x0, %3\n"
				"mov x1, %4\n"
				"mov x2, %5\n"
				"mov x3, %6\n"
				"mov x4, %7\n"
				"mov x5, %8\n"
				"mov x6, %9\n"
				"mov x7, %10\n"
				"mov x8, %11\n"
				"blr %1\n"
				"1:\n"
				: "=m"(simulation.return_address)
				: "r"(simulation.sim_code.code), "r"(simulation.simulation_memory),
				"r"(regs_blob[0]), "r"(regs_blob[1]), "r"(regs_blob[2]), "r"(regs_blob[3]),
				"r"(regs_blob[4]), "r"(regs_blob[5]), "r"(regs_blob[6]), "r"(regs_blob[7]),
				"r"(regs_blob[8])
				: "x9", "x29", "memory", "cc", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x10", "x30"
			);
	}


	free(simulation.simulation_memory);
main_code_free:
	simulation_code_free(&simulation.sim_code);
main_input_free:
	simulation_input_free(&simulation.sim_input);
main_destroy_shm:
	destroy_shm(shm);
	python_finalize();
main_out:
	return ret;
}
