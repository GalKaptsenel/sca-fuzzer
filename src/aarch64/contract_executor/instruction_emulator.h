#ifndef INSTRUCTION_EMULATOR_H
#define INSTRUCTION_EMULATOR_H

#include <stdint.h>
#include <stddef.h>
#include "simulation_state.h"

/* The emulation table supplies a sealing site's genuine per-input value during the trace, by injecting
 * it into the GST image (sim_input.simd); base_hook then reloads the vector file from that image, so the
 * site's fixed UMOV reads the injected value. The site's code is fixed (UMOV + apply + anchor) and is
 * never patched. Each row names the site's first slot (the MATERIALIZE trigger), the byte span to its
 * anchor (the committed AUT* / the guarded access, at offset+size, which the op decodes to find the
 * pointer/context register), and the GST lane byte offset(s) the op writes. There is no stored per-row
 * state: MATERIALIZE is a pure function of PC, so re-reaches and speculation rollback need no
 * bookkeeping (the op is idempotent per reach, the lane persists, the resolve image is discarded after).
 * The CE knows nothing about sealing — only how to route to these ops. The table is OPTIONAL and
 * orthogonal to the speculation contract; the caller chooses both. */

enum emul_op {
	EMUL_OP_NONE      = 0,
	EMUL_OP_PAC_AUTH  = 1,   /* anchor AUT*: write the signed pointer to gst_offset so the AUT* verifies */
	EMUL_OP_MTE_RETAG = 2,   /* anchor access: write the accessed cell's tag to gst_offset (applied) and
	                          * gst_offset2 (the after-access re-converge lane)                           */
};

struct emul_row {
	uint32_t offset;      /* byte offset of the first slot (the MATERIALIZE trigger) */
	uint32_t size;        /* bytes from the first slot to the anchor; the anchor sits at offset+size */
	uint32_t op;          /* enum emul_op */
	uint32_t gst_offset;  /* byte offset into the GST image of the lane the op writes */
	uint32_t gst_offset2; /* MTE: the after-access re-converge lane; unused (0) for PAC */
};

static inline int emul_table_valid(const uint8_t* table, size_t size, size_t code_size,
                                   size_t simd_size) {
	if (0 != size % sizeof(struct emul_row)) {
		return 0;
	}
	const struct emul_row* rows = (const struct emul_row*)table;
	for (size_t i = 0; i < size / sizeof(struct emul_row); ++i) {
		/* The anchor instruction sits at offset+size, so it (its 4 bytes) must itself lie in the code. */
		if (0 != rows[i].offset % 4 || 0 == rows[i].size || 0 != rows[i].size % 4 ||
		    (uint64_t)rows[i].offset + rows[i].size + 4 > code_size ||
		    rows[i].gst_offset >= simd_size || rows[i].gst_offset2 >= simd_size) {
			return 0;
		}
	}
	return 1;
}

/* The site-lifecycle phase the dispatcher hands an op-emulator. MATERIALIZE: inject the genuine value
 * for the state currently in the registers. IDLE: nothing to do (the site's code is fixed). */
typedef enum {
	EMUL_PHASE_IDLE = 0,
	EMUL_PHASE_MATERIALIZE = 1,
} emul_phase_t;

/* The signature every op-emulator implements; the dispatcher routes by op id through this type. */
typedef void (*emul_op_fn)(struct simulation_state* sim_state, const struct emul_row* row,
                           emul_phase_t phase);

/* Per-instruction hook: route MATERIALIZE to the owning op-emulator when PC is at a row's first slot.
 * Install BEFORE the per-op PAC/MTE hooks and the trace logger. */
void* instruction_emulator_hook(struct simulation_state* sim_state);

#endif /* INSTRUCTION_EMULATOR_H */
