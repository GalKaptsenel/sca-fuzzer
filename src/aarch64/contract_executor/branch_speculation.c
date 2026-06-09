#include "branch_speculation.h"
#include "simulation_execution_clause_hook.h"  /* spec_nesting / spec_max_nesting / spec_push_frame */
#include "instruction_encodings.h"              /* classify_branch, evaluate_cond_target, BRANCH_* */
#include "utils.h"                              /* get_bit */
#include <stdbool.h>

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
		case 0xE: return true;			// AL
		case 0xF: return true;			// NV behaves as AL
		default:  return false;			// invalid
	}
}

static uintptr_t evaluate_cond_branch_bool(const struct cpu_state* state, bool req_direction) {
	uintptr_t pc = state->pc;
	uint32_t insn = *(uint32_t*)pc;
	branch_type_t btype = classify_branch(insn);
	uintptr_t target = evaluate_cond_target(pc, insn);

	if(BRANCH_B_COND == btype) {
		uint32_t cond = insn & 0xF;
		bool arch_take = condition_passed(cond, state->nzcv) == req_direction;
		return arch_take ? target : pc + 4;
	}

	if (BRANCH_CBZ == btype || BRANCH_CBNZ == btype) {
		uint32_t Rt = insn & 0x1F;
		uint32_t is_64bit = (insn >> 31) & 1;
		uint64_t value = cpu_state_read_gpr_zr(state, Rt);
		value = is_64bit ? value : (uint32_t)value;
		bool take = (0 == value) == (BRANCH_CBZ == btype);
		bool arch_take = take == req_direction;
		return arch_take ? target : pc + 4;
	}

	if (BRANCH_TBZ == btype || BRANCH_TBNZ == btype) {
		uint32_t Rt     = insn & 0x1F;
		uint32_t b5_msb = (insn >> 31) & 0x1;
		uint32_t b40    = (insn >> 19) & 0x1F;
		uint32_t bit_num = (b5_msb << 5) | b40;
		bool bit_set = (cpu_state_read_gpr_zr(state, Rt) >> bit_num) & 1;
		bool take = (0 == bit_set) == (BRANCH_TBZ == btype);
		bool arch_take = take == req_direction;
		return arch_take ? target : pc + 4;
	}

	return 0;	// not a conditional branch we recognize
}

static uintptr_t cond_branch_mispredicted_next(const struct cpu_state* state) {
	return evaluate_cond_branch_bool(state, false);
}

uintptr_t cond_branch_architectural_next(const struct cpu_state* state) {
	return evaluate_cond_branch_bool(state, true);
}

int is_conditional_branch(uint32_t insn) {
	switch (classify_branch(insn)) {
		case BRANCH_B_COND:
		case BRANCH_CBZ:
		case BRANCH_CBNZ:
		case BRANCH_TBZ:
		case BRANCH_TBNZ:
			return 1;
		default:
			return 0;
	}
}

void* branch_speculate(struct simulation_state* sim_state, int mispredict, uint64_t owner) {
	uintptr_t arch_next = cond_branch_architectural_next(&sim_state->cpu_state);
	if (!mispredict || spec_nesting() >= spec_max_nesting()) {
		return (void*)arch_next;
	}
	spec_push_frame(sim_state, arch_next, owner);
	return (void*)cond_branch_mispredicted_next(&sim_state->cpu_state);
}
