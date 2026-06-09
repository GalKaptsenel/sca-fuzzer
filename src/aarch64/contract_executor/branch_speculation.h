#ifndef BRANCH_SPECULATION_H
#define BRANCH_SPECULATION_H

#include <stdint.h>
#include "simulation_state.h"

/* Conditional-branch resolution + the shared misprediction fork, used by every
 * branch-misprediction execution clause (cond, bpu). Pure branch semantics here; the
 * per-clause decision of WHETHER to mispredict lives in the clause. */

/* True for the conditional branches we model (B.cond / CBZ / CBNZ / TBZ / TBNZ). */
int is_conditional_branch(uint32_t insn);

/* Architectural next PC of the conditional branch at `state->pc` (resolves the real direction). */
uintptr_t cond_branch_architectural_next(const struct cpu_state* state);

/* Drive the fork: if `mispredict` (and depth allows), checkpoint (owned by `owner`) + schedule
 * the architectural successor for rollback and return the mispredicted target; otherwise return
 * the architectural next PC. Returns the PC the simulator should continue from. */
void* branch_speculate(struct simulation_state* sim_state, int mispredict, uint64_t owner);

#endif // BRANCH_SPECULATION_H
