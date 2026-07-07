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
	uintptr_t pc = sim_state->cpu_state.pc;
	uint32_t insn = *(uint32_t*)pc;
	if (!is_conditional_branch(insn)) return NULL;

	uintptr_t target = evaluate_cond_target(pc, insn);
	int arch_taken = cond_branch_is_taken(&sim_state->cpu_state);
	int predicted  = g_predictor->predict(pc);
	int mispredict = (0 != predicted) != (0 != arch_taken);

	/* Train the tables on the architectural path only: wrong-path (speculated) branches never
	 * retire, so they must not train. */
	if (0 == spec_nesting()) {
		g_predictor->update(pc, arch_taken, target);
	}

	/* Speculative global history. If this misprediction opens a window we dive down the predicted
	 * (wrong) path, so checkpoint the pre-branch history and advance it with the PREDICTED
	 * direction; bpu_on_rollback() then restores it and re-advances with the resolved direction.
	 * Otherwise we continue on the architectural path, so advance with the architectural direction.
	 * The guard mirrors branch_speculate()'s push condition exactly, keeping the history checkpoint
	 * 1:1 with the engine's speculation frames. */
	if (mispredict && spec_nesting() < spec_max_nesting()) {
		g_predictor->checkpoint();
		g_predictor->advance(pc, predicted, target);
	} else {
		g_predictor->advance(pc, arch_taken, target);
	}

	return branch_speculate(sim_state, mispredict, bpu_index);
}

/* Recover the speculative history when a misprediction window unwinds. on_rollback replaces the
 * engine's default checkpoint reload, so reload the architectural checkpoint here, then — with the
 * CPU state restored to the branch (pc + registers) — recompute the resolved direction and
 * restore + re-advance the history: rollback() undoes the wrong-path advance and advance(arch_taken)
 * reapplies the correct one, leaving exactly the history a correct prediction would have produced. */
static void bpu_on_rollback(struct simulation_state* sim_state,
                            const struct execution_checkpoint_desc* frame) {
	spec_reload_checkpoint(sim_state, frame);

	uintptr_t pc = sim_state->cpu_state.pc;
	uint32_t insn = *(uint32_t*)pc;
	uintptr_t target = evaluate_cond_target(pc, insn);
	int arch_taken = cond_branch_is_taken(&sim_state->cpu_state);

	g_predictor->rollback();
	g_predictor->advance(pc, arch_taken, target);
}

/* A speculation barrier stops the mispredicted path: revert through the oldest open misprediction
 * window and everything nested in it. bpu_on_rollback then restores predictor history per frame. */
static uint64_t bpu_on_barrier(struct simulation_state* sim_state) {
	if (!barrier_fences_control(*(uint32_t*)sim_state->cpu_state.pc)) return SPEC_NO_REVERT;
	return spec_oldest_frame_of_owner(bpu_index);
}

const struct execution_clause_descriptor bpu_execution_clause = {
	.name           = "bpu",
	.clause_bit     = EXEC_CLAUSE_BPU,
	.on_init        = bpu_on_init,
	.on_reset       = bpu_on_reset,
	.on_instruction = bpu_on_instruction,
	.on_barrier     = bpu_on_barrier,
	.on_rollback    = bpu_on_rollback,
};
