#include "execution_clause_bpu.h"
#include "branch_speculation.h"
#include "instruction_encodings.h"   /* evaluate_cond_target */
#include "simulation_input.h"        /* EXEC_CLAUSE_BPU */

/* Mispredict branches according to an injected predictor; the concrete model (e.g. Neoverse-N3)
 * is wired in at the composition root. No default — using the clause without a predictor traps. */
static const struct branch_predictor* g_predictor = NULL;
static uint64_t bpu_index;
static int      predictor_inited = 0;

void bpu_clause_set_predictor(const struct branch_predictor* predictor) {
	g_predictor = predictor;
}

static void bpu_on_init(uint64_t index) { bpu_index = index; }

/* Construct lazily and once (only reached when the BPU clause is enabled), then reset per run.
 * Eagerly constructing at injection would force the predictor's setup on every CE invocation,
 * including non-BPU ones. */
static void bpu_on_reset(void) {
	if (NULL == g_predictor) __builtin_trap();   // BPU enabled but no predictor injected
	if (!predictor_inited) {
		if (g_predictor->init) g_predictor->init();
		predictor_inited = 1;
	}
	if (g_predictor->reset) g_predictor->reset();
}

static void* bpu_on_instruction(struct simulation_state* sim_state) {
	uint32_t insn = *(uint32_t*)sim_state->cpu_state.pc;
	if (!is_conditional_branch(insn)) return NULL;

	uintptr_t arch_next = cond_branch_architectural_next(&sim_state->cpu_state);
	uintptr_t target    = evaluate_cond_target(sim_state->cpu_state.pc, insn);
	int arch_taken = (target == arch_next);
	int predicted  = g_predictor->predict(sim_state->cpu_state.pc);
	g_predictor->update(sim_state->cpu_state.pc, arch_taken, arch_next);

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
