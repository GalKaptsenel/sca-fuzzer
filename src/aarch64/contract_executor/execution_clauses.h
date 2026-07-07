#ifndef EXECUTION_CLAUSES_H
#define EXECUTION_CLAUSES_H

#include <stdint.h>
#include "simulation_state.h"
#include "simulation_execution_clause_hook.h"  /* struct execution_checkpoint_desc + engine API */

/* One composable execution clause. The engine enables the subset whose clause_bit is set in
 * the run's execution_clauses bitmask and drives them through these callbacks. Each clause
 * lives in its own file and exposes one such descriptor; the registry collects them. */
struct execution_clause_descriptor {
	const char* name;
	uint32_t    clause_bit;                            /* EXEC_CLAUSE_* that enables this clause */
	void  (*on_init)(uint64_t index);                  /* CE-start: record index, one-time setup */
	void  (*on_reset)(void);                           /* per-simulation (re)init                 */
	/* Per instruction; may speculate (spec_push_frame). Returns a PC to redirect to, or NULL.
	 * If several enabled clauses redirect on one instruction, they must agree (engine traps). */
	void* (*on_instruction)(struct simulation_state* sim_state);
	/* Barrier honoring (called only when EXEC_CLAUSE_BARRIER is also enabled). If the current
	 * instruction is a barrier that fences this clause's speculation AND this clause has an open
	 * window, return the id of the oldest such window to revert through; otherwise SPEC_NO_REVERT.
	 * NULL => this clause is barrier-agnostic. */
	uint64_t (*on_barrier)(struct simulation_state* sim_state);
	/* Recover a frame this clause pushed; NULL => engine reloads the frame's checkpoint. */
	void  (*on_rollback)(struct simulation_state* sim_state,
	                     const struct execution_checkpoint_desc* frame);
};

int                                   execution_clause_count(void);
const struct execution_clause_descriptor* execution_clause_at(int index);

/* True iff `clauses` (an EXEC_CLAUSE_* bitmask) is a supported combination. Arbitrary mixes
 * are rejected (e.g. two branch models). */
int execution_clauses_supported(uint64_t clauses);

#endif // EXECUTION_CLAUSES_H
