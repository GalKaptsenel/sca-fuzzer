#ifndef INSTRUCTION_EMULATOR_H
#define INSTRUCTION_EMULATOR_H

#include <stdint.h>
#include <stddef.h>
#include "simulation_state.h"

/* The emulation table materializes the genuine instructions of a sealing site in place, during the
 * trace. Each row is (code byte offset of the first slot, total slot size in bytes, op-emulator id).
 * The dispatcher is a pure event router: from PC vs row geometry it raises two phases per row and
 * forwards them to the op-emulator named by `op`. The op-emulator owns its whole site — it decides
 * whether and which slots to fill with idle UDF, and plants the genuine instructions (QARMA signature,
 * MTE retag) for the state currently in the registers. There is no stored per-row state: both phases
 * are a pure function of PC, so re-reaches and speculation rollback need no bookkeeping. The CE knows
 * nothing about sealing — only how to route to these ops. The table is OPTIONAL and orthogonal to the
 * speculation contract; the caller chooses both. */

enum emul_op {
	EMUL_OP_NONE      = 0,
	EMUL_OP_PAC_AUTH  = 1,   /* slots [MOVK(sig) x n], anchor AUT*: write the sig MOVKs so AUT* verifies */
	EMUL_OP_MTE_RETAG = 2,   /* slot [ADDG], anchor access: retag the pointer to the accessed cell tag   */
	EMUL_OP_MTE_DEMERGE = 3, /* slot [ADDG] after the access: re-converge the pointer tag to the cell tag */
};

struct emul_row {
	uint32_t offset;   /* byte offset of the first slot the row owns */
	uint32_t size;     /* bytes of the slots the row owns (>= 4); the op's anchor sits at offset+size */
	uint32_t op;       /* enum emul_op */
};

static inline int emul_table_valid(const uint8_t* table, size_t size, size_t code_size) {
	if (0 != size % sizeof(struct emul_row)) {
		return 0;
	}
	const struct emul_row* rows = (const struct emul_row*)table;
	for (size_t i = 0; i < size / sizeof(struct emul_row); ++i) {
		/* The anchor instruction sits at offset+size, so it (its 4 bytes) must itself lie in the code. */
		if (0 != rows[i].offset % 4 || 0 == rows[i].size || 0 != rows[i].size % 4 ||
		    (uint64_t)rows[i].offset + rows[i].size + 4 > code_size) {
			return 0;
		}
	}
	return 1;
}

/* The site-lifecycle phase the dispatcher hands an op-emulator. IDLE: restore the site's idle filler
 * (the op decides whether and which slots to UDF); MATERIALIZE: plant the genuine instructions for the
 * state currently in the registers. */
typedef enum {
	EMUL_PHASE_IDLE = 0,
	EMUL_PHASE_MATERIALIZE = 1,
} emul_phase_t;

/* The signature every op-emulator implements; the dispatcher routes by op id through this type. */
typedef void (*emul_op_fn)(struct simulation_state* sim_state, uintptr_t offset, uint32_t size,
                           emul_phase_t phase);

/* The idle filler (UDF #0) an op-emulator plants into the `n_slots` 4-byte slots it owns, starting at
 * code word `first_slot`. A slot that ever executes un-materialized then faults loudly. */
void emul_udf_fill(uint32_t first_slot, uint32_t n_slots);

/* Set every row's site to its idle filler. Called once before each trace. */
void emul_reset(void);

/* Per-instruction hook: route the two site phases to the owning op-emulator — MATERIALIZE when PC is at
 * a row's first slot, IDLE when PC has just left the row's last slot. Install BEFORE the per-op PAC/MTE
 * hooks and the trace logger. */
void* instruction_emulator_hook(struct simulation_state* sim_state);

#endif /* INSTRUCTION_EMULATOR_H */
