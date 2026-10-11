#include "instruction_emulator.h"
#include "simulation.h"

void* instruction_emulator_hook(struct simulation_state* sim_state) {
	const uint8_t* raw = simulation.sim_input.emul_table;
	if (NULL == sim_state || NULL == raw) {
		return NULL;
	}
	const struct emul_entry* table = (const struct emul_entry*)raw;
	size_t count = simulation.sim_input.emul_table_size / sizeof(struct emul_entry);
	uintptr_t off = sim_state->cpu_state.pc - (uintptr_t)simulation.sim_code.code;

	for (size_t i = 0; i < count; ++i) {
		if (table[i].offset == off) {
			/* live code (read by downstream hooks + native exec) and input copy (read by the logger) */
			((uint32_t*)simulation.sim_code.code)[off / 4] = table[i].word;
			((uint32_t*)simulation.sim_input.code)[off / 4] = table[i].word;
			return NULL;
		}
	}
	return NULL;
}
