#include "main.h"


int hook_aarch64_instructions(
		const struct simulation_input* sim_input,
		struct sim_code* sc,
		void *hook_addr) {

	if(NULL == sim_input || NULL == sc || NULL == hook_addr) return -1;
	if (sim_input->hdr.code_size % 4 != 0) return -1;

	sc->code_size = sim_input->hdr.code_size;

	size_t n_instructions = size / 4;
	uint32_t* sim_code = (uint32_t*)sc->simulation_code;

	for (size_t i = 0; i < n_instructions; ++i) {
		uintptr_t pc = sim_code + i;

        	uint32_t bl = encode_bl(pc, (uintptr_t)hook_addr);
        	if (0 == bl) {
			return -1;
		}

		sim_code[i] = bl;
	}

	__builtin___clear_cache((char*)sim_code, (char*)sim_code + sc->code_size);
	return 0;
}

static inline uint32_t pc_to_orig_instruction(uintptr_t pc) {
	uintptr_t pc_offset  = pc - (uintptr_t)simulation.sim_code.simulation_code;
	if (pc_offset % 4 != 0 || pc_offset >= simulation.sim_input.hdr.code_size) {
		__builtin_trap(); // sanity check
	}

	uint32_t* saved_code_ptr = (uint32_t*)(pc_offset + (uintptr_t)simulation.sim_input.code);
	return *saved_code_ptr;
}


void base_hook_c(struct cpu_state *state) {
	if (NULL == state) __builtin_trap();

	struct simulation_state sim_state = { 0 };

	uintptr_t current_ret_address = state->pc; // By default return to the currently hooked instruction
	*((uint32_t*)state->pc) = pc_to_orig_instruction(state->pc); // Fix the hooked instruction
	state->lr = simulation.return_address; // The return address of the original code

	sim_state.cpu_state = *state;
	sim_state.memory = simulation.simulation_memory;

	memcpy(simulation.code_tmp, simulation.sim_code.simulation_code, simulation.sim_input.hdr.code_size); // store current code state
	memcpy(simulation.sim_code.simulation_code, simulation.sim_input.code, simulation.sim_input.hdr.code_size); // restore original code

	for (size_t i = 0; i < simulation.sim_code.n_listeners; ++i) {
		// Listeners can change the code flow by returning the next address to execute
		void* continue_simulation_from = simulation.sim_code.listeners[i](sim_state);
		if(NULL != continue_simulation_from) {
			current_ret_address = (uintptr_t)continue_simulation_from;
		}
	}

	memcpy(simulation.sim_input.code, simulation.simulation_code, simulation.sim_input.hdr.code_size); // copy changes from listeners

	memcpy(simulation.sim_code.simulation_code, simulation.code_tmp, simulation.sim_input.hdr.code_size); // restore code state for simulation

	simulation.return_address = sim_state.cpu_state.lr; // copy changes from listeners to the LR register

	__builtin___clear_cache((char*)simulation.sim_code.simulation_code,
			(char*)simulation.sim_code.simulation_code + simulation.sim_input.hdr.code_size);

	*state = sim_state.cpu_state;
	state->lr = current_ret_address;
}

