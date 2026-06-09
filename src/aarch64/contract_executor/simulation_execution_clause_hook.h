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
};

/* A pushed speculation frame: where/how to resume when the window unwinds. `owner` is the
 * index of the execution clause that pushed it, so the engine can route rollback back to it. */
struct execution_checkpoint_desc {
	uint64_t nesting;
	uintptr_t return_addr;
	uint64_t checkpoint_id;
	uint64_t owner;          /* execution-clause registry index */
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

/* Checkpoint the current state and schedule a rollback to `return_addr` at window end, tagging
 * the frame with `owner` (the pushing clause's registry index) for rollback routing.
 * At rollback the return address is loaded into LR (the harness continues from there). */
void spec_push_frame(struct simulation_state* sim_state, uintptr_t return_addr, uint64_t owner);

#endif // SIMULATION_EXECUTION_CLAUSE_H
