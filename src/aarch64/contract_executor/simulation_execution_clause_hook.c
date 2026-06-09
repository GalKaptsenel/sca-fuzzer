#include "simulation_execution_clause_hook.h"
#include "simulation.h"
#include "simulation_output.h"
#include "execution_clauses.h"

/* Generic speculation engine: a checkpoint/rollback stack + nesting counter. Per-instruction
 * behavior lives in the execution clauses, dispatched via the registry. */

static struct execution_mgmt mgmt = { 0 };
static int initialized = 0;
static int clauses_inited = 0;

static uint64_t take_checkpoint(struct simulation_state* sim_state) {
	if(mgmt.current_checkpoint_id >= mgmt.max_checkpoints) {
		__builtin_trap();
	}
	mgmt.checkpoints_array[mgmt.current_checkpoint_id].cpu_state = sim_state->cpu_state;
	void* checkpoint_memory = malloc(mgmt.memory_size);
	if(NULL == checkpoint_memory) {
		fprintf(stderr, "OUT OF MEMORY - Unable to allocate a new checkpoint!");
		__builtin_trap();
	}
	mgmt.checkpoints_array[mgmt.current_checkpoint_id].memory = checkpoint_memory;
	memcpy(checkpoint_memory, sim_state->memory, mgmt.memory_size);
	return mgmt.current_checkpoint_id++;
}

static void reload_checkpoint(struct simulation_state* sim_state, uint64_t checkpoint_id) {
	if(checkpoint_id >= mgmt.max_checkpoints) {
		__builtin_trap();
	}
	sim_state->cpu_state = mgmt.checkpoints_array[checkpoint_id].cpu_state;
	memcpy(sim_state->memory, mgmt.checkpoints_array[checkpoint_id].memory, mgmt.memory_size);
}

static void destroy_execution_clause(void) {
	if(initialized) {
		for(size_t i = 0; i < mgmt.current_checkpoint_id; ++i) {
			free(mgmt.checkpoints_array[i].memory);
			mgmt.checkpoints_array[i].memory = NULL;
		}
		free(mgmt.checkpoints_array);
		mgmt.checkpoints_array = NULL;
		initialized = 0;
	}
}

uint64_t spec_nesting(void)     { return mgmt.current_nesting; }
uint64_t spec_max_nesting(void) { return mgmt.max_nesting; }
uint64_t spec_memory_size(void) { return mgmt.memory_size; }

void spec_push_frame(struct simulation_state* sim_state, uintptr_t return_addr, uint64_t owner) {
	mgmt.stack[mgmt.stack_top].nesting       = mgmt.current_nesting;
	mgmt.stack[mgmt.stack_top].return_addr   = return_addr;
	mgmt.stack[mgmt.stack_top].checkpoint_id = take_checkpoint(sim_state);
	mgmt.stack[mgmt.stack_top].owner         = owner;
	++mgmt.stack_top;
	++mgmt.current_nesting;
}

static void init_clauses_once(void) {
	if(clauses_inited) return;
	int n = execution_clause_count();
	for(int i = 0; i < n; ++i) {
		const struct execution_clause_descriptor* ecd = execution_clause_at(i);
		if(ecd->on_init) ecd->on_init((uint64_t)i);
	}
	clauses_inited = 1;
}

static void ensure_initialized(void) {
	if(initialized) return;
	mgmt.current_nesting = 0;
	mgmt.max_nesting = simulation.sim_input.hdr.config.max_misspred_branch_nesting;
	mgmt.stack_top = 0;
	memset(mgmt.stack, 0, sizeof(mgmt.stack));
	mgmt.max_checkpoints = 1024;
	mgmt.current_checkpoint_id = 0;
	mgmt.memory_size = simulation.sim_input.hdr.mem_size + 0x1000; // + overflow page
	size_t checkpoints_array_size = mgmt.max_checkpoints * sizeof(struct execution_checkpoint);
	mgmt.checkpoints_array = malloc(checkpoints_array_size);
	if(NULL == mgmt.checkpoints_array) return;
	memset(mgmt.checkpoints_array, 0, checkpoints_array_size);

	uint64_t clauses = simulation.sim_input.hdr.config.execution_clauses;
	if(!execution_clauses_supported(clauses)) {
		fprintf(stderr, "Unsupported execution_clauses bitmask: 0x%lx\n", (unsigned long)clauses);
		__builtin_trap();
	}

	int n = execution_clause_count();
	for(int i = 0; i < n; ++i) {
		const struct execution_clause_descriptor* ecd = execution_clause_at(i);
		if((ecd->clause_bit & clauses) && ecd->on_reset) ecd->on_reset();
	}
	initialized = 1;
}

static void* handle_window_end(struct simulation_state* sim_state) {
	if(0 == mgmt.current_nesting) {
		destroy_execution_clause();
		return NULL;
	}
	if(0 == mgmt.stack_top) __builtin_trap();
	--mgmt.stack_top;
	struct execution_checkpoint_desc* frame = &mgmt.stack[mgmt.stack_top];
	mgmt.current_nesting = frame->nesting;

	const struct execution_clause_descriptor* ecd = execution_clause_at((int)frame->owner);
	if(ecd->on_rollback) {
		ecd->on_rollback(sim_state, frame);
	} else {
		reload_checkpoint(sim_state, frame->checkpoint_id);
	}

	/* Checkpoints are strictly LIFO with the frame stack, so the one being popped is the most
	 * recent: free it and reuse the slot. This bounds total checkpoints by live nesting depth
	 * (not by the number of speculative excursions). */
	if(frame->checkpoint_id != mgmt.current_checkpoint_id - 1) __builtin_trap();
	free(mgmt.checkpoints_array[frame->checkpoint_id].memory);
	mgmt.checkpoints_array[frame->checkpoint_id].memory = NULL;
	--mgmt.current_checkpoint_id;

	return (void*)frame->return_addr;
}

void* log_instr_execution_cluase_hook(struct simulation_state* sim_state) {
	log_instr_with_speculation_nesting(sim_state, mgmt.current_nesting);
	return NULL;
}

void* execution_clause_hook(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;

	init_clauses_once();
	ensure_initialized();
	if(!initialized) return NULL;

	if(out_of_simulation(&sim_state->cpu_state)) {
		return handle_window_end(sim_state);
	}

	uint64_t clauses = simulation.sim_input.hdr.config.execution_clauses;
	void* redirect = NULL;
	int n = execution_clause_count();
	for(int i = 0; i < n; ++i) {
		const struct execution_clause_descriptor* ecd = execution_clause_at(i);
		if(!(ecd->clause_bit & clauses)) continue;
		void* r = ecd->on_instruction(sim_state);
		if(NULL != r) {
			if(NULL != redirect && redirect != r) __builtin_trap(); // clauses disagree on redirect
			redirect = r;
		}
	}
	return redirect;
}

void reset_execution_clause_state(void) {
	destroy_execution_clause();
}

int is_in_speculation(void) {
	return initialized && (mgmt.current_nesting > 0);
}
