#ifndef MTE_SEAL_PLUGIN_H
#define MTE_SEAL_PLUGIN_H

#include "simulation_state.h"
#include "instruction_emulator.h"

/* op-emulator for EMUL_OP_MTE_RETAG: the guarded access is the anchor at offset+size. MATERIALIZE reads
 * the accessed cell's allocation tag and injects it into the applied lane (row->gvt_offset) and the
 * after-access re-converge lane (row->gvt_offset2), so the fixed BFI slots set the pointer tag [59:56]
 * to it and the tag check matches. IDLE: nothing (the site's code is fixed). */
void mte_emulate_retag(struct simulation_state *sim_state, const struct emul_row *row,
		       emul_phase_t phase);

#endif // MTE_SEAL_PLUGIN_H
