#include "simulation_execution_clause_hook.h"
#include "simulation.h"
#include "tage_py.h"

static inline bool get_bit(uint32_t val, int n) {
	return (val >> n) & 1;
}

static bool condition_passed(uint32_t cond, uint32_t nzcv) {
	bool N = get_bit(nzcv, 31);
	bool Z = get_bit(nzcv, 30);
	bool C = get_bit(nzcv, 29);
	bool V = get_bit(nzcv, 28);

	switch (cond) {
		case 0x0: return Z;			// EQ
		case 0x1: return !Z;			// NE
		case 0x2: return C;			// CS\HS
		case 0x3: return !C;			// CC\LO
		case 0x4: return N;			// MI
		case 0x5: return !N;			// PL
		case 0x6: return V;			// VS
		case 0x7: return !V;			// VC
		case 0x8: return !Z && C;		// HI
		case 0x9: return Z || !C;		// LS
		case 0xA: return N == V;		// GE
		case 0xB: return N != V;		// LT
		case 0xC: return !Z && (N == V);	// GT
		case 0xD: return Z || (N != V);		// LE
		case 0xE: return true;			// AL (always)
		case 0xF: return false;			// NV (always)
		default:  return false;			// invalid
	}
}

static uintptr_t gpr_x(const struct cpu_state* state, uint64_t n) {
	switch (n) {
		case 0: return state->gprs.x0;
		case 1: return state->gprs.x1;
		case 2: return state->gprs.x2;
		case 3: return state->gprs.x3;
		case 4: return state->gprs.x4;
		case 5: return state->gprs.x5;
		case 6: return state->gprs.x6;
		case 7: return state->gprs.x7;
		case 8: return state->gprs.x8;
		case 9: return state->gprs.x9;
		case 10: return state->gprs.x10;
		case 11: return state->gprs.x11;
		case 12: return state->gprs.x12;
		case 13: return state->gprs.x13;
		case 14: return state->gprs.x14;
		case 15: return state->gprs.x15;
		case 16: return state->gprs.x16;
		case 17: return state->gprs.x17;
		case 18: return state->gprs.x18;
		case 19: return state->gprs.x19;
		case 20: return state->gprs.x20;
		case 21: return state->gprs.x21;
		case 22: return state->gprs.x22;
		case 23: return state->gprs.x23;
		case 24: return state->gprs.x24;
		case 25: return state->gprs.x25;
		case 26: return state->gprs.x26;
		case 27: return state->gprs.x27;
		case 28: return state->gprs.x28;
		case 29: return state->gprs.x29;
		case 30: return state->lr;
		default:  return 0;			// invalid
	}
}

static uintptr_t evaluate_cond_branch_bool(const struct cpu_state* state, bool req_direction) {
	uintptr_t pc = state->pc;
	uint32_t insn = *(uint32_t*)pc;
	branch_type_t btype = classify_branch(insn);
	uintptr_t target = evaluate_cond_target(pc, insn);

	if(BRANCH_B_COND == btype) {
		uint32_t cond = insn & 0xF;
		if (condition_passed(cond, state->nzcv)) {
			return req_direction ? target : pc + 4;
		} else {
			return req_direction ? pc + 4: target;
		}
	}

	if (BRANCH_CBZ == btype || BRANCH_CBNZ == btype) {
		uint32_t Rt = insn & 0x1F;
		bool zero = gpr_x(state, Rt) == 0;
		bool is_cbz = BRANCH_CBZ == btype;
		bool take = is_cbz ? zero : !zero;
		if(take) {
			return req_direction ? target : pc + 4;
		} else {
			return req_direction ? pc + 4 : target;
		}
	}

	if (BRANCH_TBZ == btype || BRANCH_TBNZ == btype) {
		uint32_t Rt = insn & 0x1F;
		uint32_t b5 = (insn >> 19) & 0x1F;     // bit to test
		bool bit_set = (gpr_x(state, Rt) >> b5) & 1;
		bool is_tbz = BRANCH_TBZ == btype;
		bool take = is_tbz ? !bit_set : bit_set;
		if(take) {
			return req_direction ? target : pc + 4;
		} else {
			return req_direction ? pc + 4 : target;
		}
	}

	// Not a conditional branch we recognize
	return 0;
}

static uintptr_t evaluate_cond_branch_taken(const struct cpu_state* state) {
	return evaluate_cond_branch_bool(state, true);
}
static uintptr_t evaluate_cond_branch_not_taken(const struct cpu_state* state) {
	return evaluate_cond_branch_bool(state, false);
}

static struct execution_mgmt mgmt = { 0 };
static int initialized = 0;

static uint64_t take_checkpoint(struct simulation_state* sim_state) {
	if(mgmt.current_checkpoint_id >= mgmt.max_checkpoints) {
		__builtin_trap(); // sanity check
	}
	mgmt.checkpoints_array[mgmt.current_checkpoint_id].cpu_state = sim_state->cpu_state;
	void* checkpoint_memory = malloc(mgmt.memory_size);
	if(NULL == checkpoint_memory) {
		fprintf(stderr, "OUT OF MEMORY - Unable to allocate a new checkpoint!");
		__builtin_trap(); // sanity check
	}
	mgmt.checkpoints_array[mgmt.current_checkpoint_id].memory = checkpoint_memory;
	memcpy(checkpoint_memory, sim_state->memory, mgmt.memory_size);
	uint64_t checkpoint_id = mgmt.current_checkpoint_id;
	++mgmt.current_checkpoint_id;
	return checkpoint_id;
}

static void reload_checkpoint(struct simulation_state* sim_state, uint64_t checkpoint_id) {
	if(checkpoint_id >= mgmt.max_checkpoints) {
		__builtin_trap(); // sanity check
	}

	sim_state->cpu_state = mgmt.checkpoints_array[checkpoint_id].cpu_state;
	memcpy(sim_state->memory, mgmt.checkpoints_array[checkpoint_id].memory, mgmt.memory_size);
}

static void initialize_director() {
	fprintf(stderr, "[LOG] init director\n");
	if(0 != tagebp_init("src/aarch64/contract_executor", "bootstrap_director", 2, 4, 4096)) {
		__builtin_trap(); // sanity check
	}
}

static void destroy_director() {
	fprintf(stderr, "[LOG] destroy director instance\n");
	tagebp_destroy_instance();
	fprintf(stderr, "[LOG] destroy director instance finished\n");
}

static uintptr_t director_predict(uintptr_t pc) {
	uintptr_t res = tagebp_predict(pc);
	fprintf(stderr, "[LOG] predict director at 0x%" PRIxPTR ": 0x%" PRIxPTR "\n", pc, res);
	return res;
}

static void director_update(uintptr_t pc, bool taken) {
	fprintf(stderr, "[LOG] predict update at 0x%" PRIxPTR " with %s\n", pc, taken ? "taken" : "not-taken");
	tagebp_update(pc, taken);
}
static void destroy_execution_clause() {
	// When adding support for call and ret, I should make this a hook, that catches RET + no calls inside stack + no speculation
	if(initialized) {
		destroy_director();
		for(size_t i = 0; i < mgmt.current_checkpoint_id; ++i) {
			free(mgmt.checkpoints_array[i].memory);
			mgmt.checkpoints_array[i].memory = NULL;
		}
		free(mgmt.checkpoints_array);
		mgmt.checkpoints_array = NULL;
		initialized = 0;
	}
}

void* execution_clause_hook(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;

	if(!initialized) {
		mgmt.current_nesting = 0;
		mgmt.max_nesting = simulation.sim_input.hdr.config.max_misspred_branch_nesting;
		mgmt.stack_top = 0;
		memset(mgmt.stack, 0, sizeof(mgmt.stack));
		mgmt.max_checkpoints = 1024;
		mgmt.current_checkpoint_id = 0;
		mgmt.memory_size = simulation.sim_input.hdr.mem_size;
		size_t checkpoints_array_size = mgmt.max_checkpoints * sizeof(struct execution_checkpoint);
		mgmt.checkpoints_array = malloc(checkpoints_array_size);
		if(NULL == mgmt.checkpoints_array) return NULL;
		memset(mgmt.checkpoints_array, 0, checkpoints_array_size);

		initialize_director();

		initialized = 1;
	}

	if(out_of_simulation(&sim_state->cpu_state)) {
		if(0 == mgmt.current_nesting) {
			destroy_execution_clause();
			return NULL;
		}

		if(0 == mgmt.stack_top) __builtin_trap(); // Should never happen
		--mgmt.stack_top;
		mgmt.current_nesting = mgmt.stack[mgmt.stack_top].nesting;
		reload_checkpoint(sim_state, mgmt.stack[mgmt.stack_top].checkpoint_id);
		return (void*)mgmt.stack[mgmt.stack_top].return_addr;
	}

	uint32_t branch_type = classify_branch(*(uint32_t*)sim_state->cpu_state.pc);
	if(BRANCH_NONE == branch_type || BRANCH_B == branch_type || BRANCH_BL == branch_type || BRANCH_BLR == branch_type) return NULL;

	if(mgmt.max_nesting <= mgmt.current_nesting) {
		if(0 == mgmt.current_nesting) __builtin_trap(); // Should never happen
		if(0 == mgmt.stack_top) __builtin_trap(); // Should never happen
		--mgmt.stack_top;
		mgmt.current_nesting = mgmt.stack[mgmt.stack_top].nesting;
		reload_checkpoint(sim_state, mgmt.stack[mgmt.stack_top].checkpoint_id);
		return (void*)mgmt.stack[mgmt.stack_top].return_addr;
	}

	const uintptr_t target_addr = evaluate_cond_target(sim_state->cpu_state.pc, *(uint32_t*)sim_state->cpu_state.pc);
	// check what the director says, if it is the same as real direction, then no deeper speculation needed, otherwise speculate to the mispredicted direction
	const uintptr_t taken_addr = evaluate_cond_branch_taken(&sim_state->cpu_state);
	const uintptr_t predicted = director_predict(sim_state->cpu_state.pc);
	director_update(sim_state->cpu_state.pc, target_addr == taken_addr);
	if((target_addr == taken_addr && predicted) || (target_addr != taken_addr && !predicted)) {
		fprintf(stderr, "[LOG] currect prediction\n");
		return (void*)taken_addr;
	}

	fprintf(stderr, "[LOG] incurrect prediction\n");
	if(0 == mgmt.current_nesting) {
		mgmt.stack[mgmt.stack_top].nesting = 0;
		++mgmt.current_nesting;
	} else {
		++mgmt.current_nesting;
		mgmt.stack[mgmt.stack_top].nesting = mgmt.current_nesting;
	}
	
	uint64_t checkpoint_id = take_checkpoint(sim_state);
	mgmt.stack[mgmt.stack_top].return_addr = taken_addr;
	mgmt.stack[mgmt.stack_top].checkpoint_id = checkpoint_id;
	mgmt.stack[mgmt.stack_top].reserved = 0;
	++mgmt.stack_top;

	fprintf(stderr, "[LOG] returning from hook\n");
	return (void*)evaluate_cond_branch_not_taken(&sim_state->cpu_state);
}

