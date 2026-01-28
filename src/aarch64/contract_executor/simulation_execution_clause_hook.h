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
#include "tage_py.h"

struct execution_checkpoint {
	struct cpu_state cpu_state;
	uint8_t* memory;
};

struct execution_checkpoint_desc {
	uint64_t nesting;
	uintptr_t return_addr;
	uint64_t checkpoint_id;
	uint64_t reserved;
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

#endif // SIMULATION_EXECUTION_CLAUSE_H
