#ifndef INSTRUCTION_EMULATOR_H
#define INSTRUCTION_EMULATOR_H

#include <stdint.h>
#include <stddef.h>
#include "simulation_state.h"

/* The emulation table materializes the genuine instructions of a sealing site in place, during the
 * trace. Each row is (code byte offset of the first slot, total slot size in bytes, op-emulator id).
 * On reaching an unprocessed row the CE dispatches by `op` to a general op-emulator that reads the
 * state + the site's real instruction (the last slot), computes the value (QARMA signature, MTE cell
 * tag), and writes the genuine instructions into the junk slots, then runs them as ordinary code. The
 * CE knows nothing about sealing — only how to emulate these ops. The table is OPTIONAL and orthogonal
 * to the speculation contract; the caller chooses both. */

enum emul_op {
	EMUL_OP_NONE      = 0,
	EMUL_OP_PAC_AUTH  = 1,   /* [MOVK(sig) x n, AUT*]: sign the pointer, write the sig MOVKs          */
	EMUL_OP_MTE_RETAG = 2,   /* [ADDG set, access, ADDG back]: retag the pointer to the cell tag       */
};

struct emul_row {
	uint32_t offset;   /* byte offset of the first slot in the code */
	uint32_t size;     /* total bytes of the site's slots (>= 4; last 4 bytes are the real op) */
	uint32_t op;       /* enum emul_op */
};

static inline int emul_table_valid(const uint8_t* table, size_t size, size_t code_size) {
	if (0 != size % sizeof(struct emul_row)) {
		return 0;
	}
	const struct emul_row* rows = (const struct emul_row*)table;
	for (size_t i = 0; i < size / sizeof(struct emul_row); ++i) {
		if (0 != rows[i].offset % 4 || 0 == rows[i].size || 0 != rows[i].size % 4 ||
		    (uint64_t)rows[i].offset + rows[i].size > code_size) {
			return 0;
		}
	}
	return 1;
}

/* Reset the per-message "row already emulated" state. Called once before each trace. */
void emul_reset(void);

/* Per-instruction hook: when PC reaches an unprocessed row's offset, run its op-emulator (planting the
 * genuine instructions in place) and mark the row processed. Install BEFORE the per-op PAC/MTE hooks
 * and the trace logger. */
void* instruction_emulator_hook(struct simulation_state* sim_state);

#endif /* INSTRUCTION_EMULATOR_H */
