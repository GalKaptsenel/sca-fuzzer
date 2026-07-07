#include "execution_clause_cond.h"
#include "branch_speculation.h"
#include "instruction_encodings.h"   /* barrier_fences_control */
#include "simulation_input.h"   /* EXEC_CLAUSE_COND */

static uint64_t cond_index;

static void cond_on_init(uint64_t index) { cond_index = index; }

/* Conditional-branch misprediction: at every conditional branch, take the mispredicted path. */
static void* cond_on_instruction(struct simulation_state* sim_state) {
	if (!is_conditional_branch(*(uint32_t*)sim_state->cpu_state.pc)) return NULL;
	return branch_speculate(sim_state, /*mispredict=*/1, cond_index);
}

/* A speculation barrier stops the mispredicted path: revert through the oldest open misprediction
 * window and everything nested in it. */
static uint64_t cond_on_barrier(struct simulation_state* sim_state) {
	if (!barrier_fences_control(*(uint32_t*)sim_state->cpu_state.pc)) return SPEC_NO_REVERT;
	return spec_oldest_frame_of_owner(cond_index);
}

const struct execution_clause_descriptor cond_execution_clause = {
	.name           = "cond",
	.clause_bit     = EXEC_CLAUSE_COND,
	.on_init        = cond_on_init,
	.on_reset       = NULL,
	.on_instruction = cond_on_instruction,
	.on_barrier     = cond_on_barrier,
	.on_rollback    = NULL,
};
