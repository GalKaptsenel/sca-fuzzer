#ifndef INSTRUCTION_EMULATOR_H
#define INSTRUCTION_EMULATOR_H

#include <stdint.h>
#include <stddef.h>
#include "simulation_state.h"

/* Maps a code byte offset to the instruction the CE emulates there in place of the placeholder NOP;
 * the CE's ordinary per-instruction machinery then traces it, with no knowledge of why it stands in.
 * Entries run after the speculation engine, so they are data-processing ops, not branches. */
struct emul_entry {
	uint32_t offset;
	uint32_t word;
};

static inline int emul_table_valid(const uint8_t* table, size_t size, size_t code_size) {
	if (0 != size % sizeof(struct emul_entry)) {
		return 0;
	}
	const struct emul_entry* entries = (const struct emul_entry*)table;
	for (size_t i = 0; i < size / sizeof(struct emul_entry); ++i) {
		if (0 != entries[i].offset % 4 || (uint64_t)entries[i].offset + 4 > code_size) {
			return 0;
		}
	}
	return 1;
}

/* Install BEFORE the per-instruction emulation hooks and the trace logger. */
void* instruction_emulator_hook(struct simulation_state* sim_state);

#endif /* INSTRUCTION_EMULATOR_H */
