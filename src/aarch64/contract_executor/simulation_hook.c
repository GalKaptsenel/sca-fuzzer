#include "simulation_hook.h"
#include "simulation.h"


int hook_aarch64_instructions(
		const struct simulation_input* sim_input,
		struct simulation_code* sc,
		void *hook_addr,
		size_t hook_size) {

	if(NULL == sim_input || NULL == sc || NULL == hook_addr) return -1;
	if (sim_input->hdr.code_size % 4 != 0) return -1;

	sc->code_size = sim_input->hdr.code_size;

	size_t n_instructions = (sc->code_size / 4) + 1; // Add space for 1 additional instruction for our use (we would use this for  a default RET instruction at the end of the testcase)
	uint32_t* sim_code = (uint32_t*)sc->code;
	void* copied_hook_addr = sim_code + n_instructions; 
	memcpy(copied_hook_addr, hook_addr, hook_size); 

	for (size_t i = 0; i < n_instructions; ++i) {
		uintptr_t pc = (uintptr_t)(sim_code + i);

        	uint32_t bl = encode_bl(pc, (uintptr_t)copied_hook_addr);
        	if (0 == bl) {
			return -1;
		}

		sim_code[i] = bl;
	}

	__builtin___clear_cache((char*)sim_code, (char*)copied_hook_addr + hook_size);
	return 0;
}

static inline uint32_t pc_to_orig_instruction(uintptr_t pc) {
	uintptr_t pc_offset  = pc - (uintptr_t)simulation.sim_code.code;
	if (pc_offset % 4 != 0 || pc_offset >= simulation.sim_input.hdr.code_size) {
		__builtin_trap(); // sanity check
	}

	uint32_t* saved_code_ptr = (uint32_t*)(pc_offset + (uintptr_t)simulation.sim_input.code);
	return *saved_code_ptr;
}


void base_hook_c(struct cpu_state *state) {
	if (NULL == state) __builtin_trap();

	struct simulation_state sim_state = { 0 };

	uintptr_t current_ret_address = state->pc; // By default, return to the currently hooked instruction

	if(state->pc - (uintptr_t)simulation.sim_code.code < simulation.sim_input.hdr.code_size) {
		*((uint32_t*)state->pc) = pc_to_orig_instruction(state->pc); // Fix the hooked instruction

	} else {
		// If we finished the test case area manually append a RET instruction
		*((uint32_t*)state->pc) = 0xd65f03c0; // RET
	}

	state->lr = simulation.return_address; // For the hooks, fake the LR register to point to the return address of the code when it runs wihtout simulation

	sim_state.cpu_state = *state;
	sim_state.memory = simulation.simulation_memory;

	memcpy(simulation.code_tmp, simulation.sim_code.code, simulation.sim_input.hdr.code_size); // store current code state
	memcpy(simulation.sim_code.code, simulation.sim_input.code, simulation.sim_input.hdr.code_size); // restore original code

	for (size_t i = 0; i < simulation.n_hooks; ++i) {
		// Listeners can change the code flow by returning the next address to execute
		void* continue_simulation_from = simulation.hooks[i](&sim_state);
		if(NULL != continue_simulation_from) {
			current_ret_address = (uintptr_t)continue_simulation_from;
		}
	}

	memcpy(simulation.sim_input.code, simulation.sim_code.code, simulation.sim_input.hdr.code_size); // copy changes from hooks

	memcpy(simulation.sim_code.code, simulation.code_tmp, simulation.sim_input.hdr.code_size); // restore code state for simulation

	simulation.return_address = sim_state.cpu_state.lr; // copy changes from hooks to the LR register

	__builtin___clear_cache((char*)simulation.sim_code.code,
			(char*)simulation.sim_code.code + simulation.sim_input.hdr.code_size);

	*state = sim_state.cpu_state;
	state->lr = current_ret_address;
}

void* stdout_print_hook(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;
	printf("X0 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x0);
	printf("X1 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x1);
	printf("X2 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x2);
	printf("X3 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x3);
	printf("X4 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x4);
	printf("X5 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x5);
	printf("X6 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x6);
	printf("X7 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x7);
	printf("X8 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x8);
	printf("X9 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x9);
	printf("X10 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x10);
	printf("X11 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x11);
	printf("X12 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x12);
	printf("X13 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x13);
	printf("X14 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x14);
	printf("X15 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x15);
	printf("X16 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x16);
	printf("X17 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x17);
	printf("X18 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x18);
	printf("X19 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x19);
	printf("X20 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x20);
	printf("X21 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x21);
	printf("X22 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x22);
	printf("X23 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x23);
	printf("X24 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x24);
	printf("X25 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x25);
	printf("X26 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x26);
	printf("X27 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x27);
	printf("X28 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x28);
	printf("X29 (FP) = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x29);
	printf("X30 (LR) = 0x%016" PRIxPTR "\n", sim_state->cpu_state.lr);
	printf("SP = 0x%016" PRIxPTR "\n", sim_state->cpu_state.sp);
	printf("PC = 0x%016" PRIxPTR "\n", sim_state->cpu_state.pc);
	uintptr_t nzcv = sim_state->cpu_state.nzcv;
	uint32_t N = (nzcv >> 31) & 1;
	uint32_t Z = (nzcv >> 30) & 1;
	uint32_t C = (nzcv >> 29) & 1;
	uint32_t V = (nzcv >> 28) & 1;
	printf("NZCV   = 0x%016" PRIxPTR "  [N=%u Z=%u C=%u V=%u]  (%c%c%c%c)\n",
           nzcv,
           N, Z, C, V,
           N ? 'N' : '-',
           Z ? 'Z' : '-',
           C ? 'C' : '-',
           V ? 'V' : '-');
	return NULL;
}

void* handle_ret_hook(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;
	if(0xd65f03c0 == *(uint32_t*)sim_state->cpu_state.pc) return (void*)sim_state->cpu_state.lr;
	return NULL;
}

