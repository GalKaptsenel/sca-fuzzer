#ifndef SIMULATION_EXECUTION_CLAUSE_H
#define SIMULATION_EXECUTION_CLAUSE_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "simulation.h"
#include "simulation_state.h"
#include "instruction_encodings.h"

struct execution_checkpoint {
	struct cpu_state cpu_state;
	uint8_t* memory;
	uint8_t* tags;     /* MTE tag-memory snapshot, or NULL when not in MTE-test mode */
};

/* A pushed speculation frame: where/how to resume when the window unwinds. `owner` is the
 * index of the execution clause that pushed it, so the engine can route rollback back to it. */
struct execution_checkpoint_desc {
	uint64_t nesting;
	uintptr_t return_addr;
	uint64_t checkpoint_id;
	uint64_t owner;          /* execution-clause registry index */
	uint64_t start_instr;    /* instr_count when this window opened (for the per-window cap) */
};

struct execution_mgmt {
	uint64_t current_nesting;
	uint64_t max_nesting;
	uint64_t memory_size;
	uint64_t stack_top;
	struct execution_checkpoint_desc stack[4096];
	uint64_t max_checkpoints;
	uint64_t current_checkpoint_id;
	struct execution_checkpoint* checkpoints_array;
	uint64_t instr_count;    /* instructions simulated so far (monotonic) */
	uint64_t max_instr;      /* per-window instruction cap (max_misspred_instructions; 0 = no cap) */
};

void* execution_clause_hook(struct simulation_state* sim_state);
void* log_instr_execution_cluase_hook(struct simulation_state* sim_state);
void reset_execution_clause_state(void);
int is_in_speculation(void);

/* ===================== Speculation engine API =====================
 * Generic mechanism used by the execution clauses (execution_clause_*.c). The engine knows
 * nothing about specific clauses beyond this. */
uint64_t spec_nesting(void);      /* current speculation depth                    */
uint64_t spec_max_nesting(void);  /* configured depth cap                         */
uint64_t spec_memory_size(void);  /* size of the simulated memory image (bytes)   */

/* Sentinel "no revert requested" returned by an on_barrier callback. */
#define SPEC_NO_REVERT UINT64_MAX

/* Id (stack index) of the oldest open speculation frame owned by `owner`, or SPEC_NO_REVERT if the
 * clause has no open window. A barrier honoring clause returns this from on_barrier to have the
 * engine revert (unwind) through that frame and everything nested inside it. */
uint64_t spec_oldest_frame_of_owner(uint64_t owner);

/* Checkpoint the current state and schedule a rollback to `return_addr` at window end, tagging
 * the frame with `owner` (the pushing clause's registry index) for rollback routing.
 * At rollback the return address is loaded into LR (the harness continues from there). */
void spec_push_frame(struct simulation_state* sim_state, uintptr_t return_addr, uint64_t owner);

/* Reload the architectural checkpoint a frame captured (cpu_state + memory), restoring state to the
 * branch that pushed it. Exposed so a clause's on_rollback can restore the base state before applying
 * its own recovery (e.g. the BPU restoring its speculative history from the reloaded branch state). */
void spec_reload_checkpoint(struct simulation_state* sim_state, const struct execution_checkpoint_desc* frame);

#endif // SIMULATION_EXECUTION_CLAUSE_H
