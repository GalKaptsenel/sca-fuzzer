#ifndef MTE_SEAL_PLUGIN_H
#define MTE_SEAL_PLUGIN_H

#include "simulation_state.h"
#include "instruction_emulator.h"

/* op-emulator for EMUL_OP_MTE_RETAG: the row owns the retag ADDG slot(s); the guarded access is the
 * anchor just past them. MATERIALIZE retags the access's base register so the effective address's tag
 * matches the accessed cell's allocation tag; IDLE fills the slot(s) with UDF. */
void mte_emulate_retag(struct simulation_state *sim_state, uintptr_t offset, uint32_t size,
		       emul_phase_t phase);

/* op-emulator for EMUL_OP_MTE_DEMERGE: the de-merge ADDG after a sealed access; re-converges the
 * pointer tag to the accessed cell's tag. The sealed access sits immediately before the slot. */
void mte_emulate_demerge(struct simulation_state *sim_state, uintptr_t offset, uint32_t size,
			 emul_phase_t phase);

#endif // MTE_SEAL_PLUGIN_H
