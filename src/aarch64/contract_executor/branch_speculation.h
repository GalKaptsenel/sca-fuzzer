#ifndef BRANCH_SPECULATION_H
#define BRANCH_SPECULATION_H

#include <stdint.h>
#include <stdbool.h>
#include "simulation_state.h"

/* Conditional-branch resolution + the shared misprediction fork, used by every
 * branch-misprediction execution clause (cond, bpu). Pure branch semantics here; the
 * per-clause decision of WHETHER to mispredict lives in the clause. */

/* True for the conditional branches we model (B.cond / CBZ / CBNZ / TBZ / TBNZ). */
int is_conditional_branch(uint32_t insn);

/* Architectural next PC of the conditional branch at `state->pc` (resolves the real direction). */
uintptr_t cond_branch_architectural_next(const struct cpu_state* state);

/* Resolved direction (true iff architecturally taken) of the conditional branch at `state->pc`,
 * read directly from the condition/register test — correct even when the taken-target is pc+4. */
bool cond_branch_is_taken(const struct cpu_state* state);

/* Open a misprediction window for the conditional branch at `sim_state->pc` (depth permitting):
 * explore the mispredicted target, resuming at the architectural successor; owned by `owner`. */
void branch_speculate(struct simulation_state* sim_state, uint64_t owner);

#endif // BRANCH_SPECULATION_H
