#include "execution_clause_bpu.h"
#include "branch_speculation.h"
#include "branch_predictors.h"
#include "instruction_encodings.h"   /* evaluate_cond_target */
#include "simulation_input.h"        /* EXEC_CLAUSE_BPU */
#include "simulation.h"              /* simulation.sim_input.hdr.config */
#include "simulation_execution_clause_hook.h"  /* spec_nesting */

/* Mispredict branches per the predictor selected by the input (config.branch_predictor).
 * No default — an unknown/none selection on a BPU run traps. */
static const struct branch_predictor* g_predictor = NULL;
static uint64_t bpu_index;

static void bpu_on_init(uint64_t index) { bpu_index = index; }

static void bpu_on_reset(void) {
	g_predictor = branch_predictor_by_id(simulation.sim_input.hdr.config.branch_predictor);
	if (NULL == g_predictor) __builtin_trap();   // BPU enabled but no/unknown predictor selected
	if (g_predictor->init)  g_predictor->init();  // idempotent (predictor self-guards)
	if (g_predictor->reset) g_predictor->reset();
}

static void* bpu_on_instruction(struct simulation_state* sim_state) {
	uint32_t insn = *(uint32_t*)sim_state->cpu_state.pc;
	if (!is_conditional_branch(insn)) return NULL;

	uintptr_t arch_next = cond_branch_architectural_next(&sim_state->cpu_state);
	uintptr_t target    = evaluate_cond_target(sim_state->cpu_state.pc, insn);
	int arch_taken = (target == arch_next);
	int predicted  = g_predictor->predict(sim_state->cpu_state.pc);

	/* Train only on the architectural path: wrong-path (speculated) branches never retire. */
	if (0 == spec_nesting()) {
		g_predictor->update(sim_state->cpu_state.pc, arch_taken, arch_next);
	}

	/* Mispredict iff the predictor disagrees with the architectural direction. */
	int mispredict = !((arch_taken && predicted) || (!arch_taken && !predicted));
	return branch_speculate(sim_state, mispredict, bpu_index);
}

const struct execution_clause_descriptor bpu_execution_clause = {
	.name           = "bpu",
	.clause_bit     = EXEC_CLAUSE_BPU,
	.on_init        = bpu_on_init,
	.on_reset       = bpu_on_reset,
	.on_instruction = bpu_on_instruction,
	.on_rollback    = NULL,
};
