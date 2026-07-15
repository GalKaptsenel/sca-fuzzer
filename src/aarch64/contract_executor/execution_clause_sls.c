#include "execution_clause_sls.h"
#include "branch_speculation.h"       /* is_conditional_branch, cond_branch_architectural_next */
#include "instruction_encodings.h"    /* classify_branch, decode_b_target, barrier_fences_control */
#include "simulation_input.h"         /* EXEC_CLAUSE_SLS */
#include "simulation_execution_clause_hook.h"

static uint64_t sls_index;

static void sls_on_init(uint64_t index) { sls_index = index; }

/* Straight-line speculation: at a branch, the frontend keeps fetching sequentially past it, so explore
 * pc+4 as a window resuming at the branch's architectural successor. Unconditional B (target != pc+4)
 * and taken conditional branches (pc+4 is the not-taken side) contribute pc+4; a not-taken conditional
 * already runs pc+4 architecturally, so nothing to add. RET/BL/BLR/indirect are not modelled yet. */
static void sls_on_instruction(struct simulation_state* sim_state) {
	uintptr_t pc = sim_state->cpu_state.pc;
	uint32_t insn = *(uint32_t*)pc;

	uintptr_t arch_next;
	if (BRANCH_B == classify_branch(insn)) arch_next = decode_b_target(pc, insn);
	else if (is_conditional_branch(insn)) arch_next = cond_branch_architectural_next(&sim_state->cpu_state);
	else return;

	uintptr_t straight = pc + 4;
	if (straight == arch_next) return;
	if (spec_nesting() >= spec_max_nesting()) return;
	spec_request_window(straight, arch_next, sls_index);
}

/* A control-fencing barrier stops the straight-line path: revert through the oldest open SLS window. */
static uint64_t sls_on_barrier(struct simulation_state* sim_state) {
	if (!barrier_fences_control(*(uint32_t*)sim_state->cpu_state.pc)) return SPEC_NO_REVERT;
	return spec_oldest_frame_of_owner(sls_index);
}

const struct execution_clause_descriptor sls_execution_clause = {
	.name           = "sls",
	.clause_bit     = EXEC_CLAUSE_SLS,
	.on_init        = sls_on_init,
	.on_reset       = NULL,
	.on_instruction = sls_on_instruction,
	.on_barrier     = sls_on_barrier,
	.on_rollback    = NULL,
};
