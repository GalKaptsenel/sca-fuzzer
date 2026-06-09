#include "execution_clause_bpas.h"
#include "instruction_encodings.h"   /* is_memory_access, is_store */
#include "simulation_input.h"        /* EXEC_CLAUSE_BPAS */
#include <stdlib.h>
#include <string.h>

/* Two-phase store bypass:
 *   phase A (at a store): snapshot memory, then let the HW execute the store (writeback included);
 *   phase B (next instruction): undo the store's memory write so the speculative window reads the
 *     stale value, checkpointing the POST-store state as the architectural return — so the store
 *     is never re-executed and rollback stays uniform.
 * Whole-memory snapshot/restore => STR / STP / writeback / atomics are all handled by the HW. */
static uint8_t* bpas_preimage     = NULL;   /* snapshot of memory just before a pending store */
static size_t   bpas_preimage_cap = 0;
static int      bpas_pending      = 0;
static uint64_t bpas_pending_nesting;       /* nesting at which the pending store was seen */
static uint64_t bpas_index;

static void bpas_on_init(uint64_t index) { bpas_index = index; }

static void bpas_on_reset(void) {
	bpas_pending = 0;
	size_t msz = spec_memory_size();          /* grow only if a sim ever needs more (cold path) */
	if (bpas_preimage_cap < msz) {
		uint8_t* grown = realloc(bpas_preimage, msz);
		if (NULL == grown) __builtin_trap();   // OOM
		bpas_preimage = grown;
		bpas_preimage_cap = msz;
	}
}

static void* bpas_on_instruction(struct simulation_state* sim_state) {
	/* Phase B: apply the pending bypass — but only if no rollback has happened since phase A
	 * (a changed nesting depth means our snapshot belongs to a now-discarded context). */
	if (bpas_pending) {
		if (spec_nesting() == bpas_pending_nesting) {
			spec_push_frame(sim_state, sim_state->cpu_state.pc, bpas_index); /* post-store = arch return */
			memcpy(sim_state->memory, bpas_preimage, spec_memory_size());    /* undo store -> stale */
		}
		bpas_pending = 0;
	}

	/* Phase A: snapshot before a store, let the HW execute it, undo it on the next instruction. */
	uint32_t insn = *(uint32_t*)sim_state->cpu_state.pc;
	if (spec_nesting() < spec_max_nesting() && is_memory_access(insn) && is_store(insn)) {
		memcpy(bpas_preimage, sim_state->memory, spec_memory_size());
		bpas_pending = 1;
		bpas_pending_nesting = spec_nesting();
	}
	return NULL;
}

const struct execution_clause_descriptor bpas_execution_clause = {
	.name           = "bpas",
	.clause_bit     = EXEC_CLAUSE_BPAS,
	.on_init        = bpas_on_init,
	.on_reset       = bpas_on_reset,
	.on_instruction = bpas_on_instruction,
	.on_rollback    = NULL,
};
