#include "simulation_hook.h"
#include "simulation.h"
#include "simulation_execution_clause_hook.h"
#include "instruction_encodings.h"

#include "simulation_output.h"

volatile struct cpu_state g_last_hook_cpu_state;
volatile uint32_t g_last_hook_orig_instr;

void ce_debug_print_last_sim_state(FILE *out) {
	struct cpu_state s;
	memcpy((void*)&s, (const void*)&g_last_hook_cpu_state, sizeof(s));
	uint32_t instr = g_last_hook_orig_instr;
	fprintf(out, "[CE DEBUG] Last simulated CPU state at hook entry:\n");
	fprintf(out, "  PC    = 0x%016" PRIxPTR "\n", s.pc);
	fprintf(out, "  LR    = 0x%016" PRIxPTR "\n", s.lr);
	fprintf(out, "  SP    = 0x%016" PRIxPTR "\n", s.sp);
	uintptr_t nzcv = s.nzcv;
	fprintf(out, "  NZCV  = 0x%016" PRIxPTR "  [N=%u Z=%u C=%u V=%u]\n", nzcv,
		(unsigned)((nzcv >> 31) & 1), (unsigned)((nzcv >> 30) & 1),
		(unsigned)((nzcv >> 29) & 1), (unsigned)((nzcv >> 28) & 1));
	for (int i = 0; i <= 29; i++) {
		fprintf(out, "  X%-2d   = 0x%016" PRIxPTR "\n", i, s.gpr[29 - i]);
	}
	fprintf(out, "  INSTR = 0x%08x  (original instruction at simulated PC)\n", instr);
	fflush(out);
}

//static uint32_t* emit_stub(uintptr_t src_pc, uint32_t* stub, uint32_t** hole) {
//
//	size_t idx = 0;
//
//	stub[idx++] = 0xA9BF7BFF; // stp xzr, x30 [sp, #-16]!
//
//	// x30 <- src_pc + 4
//	idx += emit_mov64((uintptr_t)(stub + idx), 30, src_pc + 4);
//
//	
//	// b hook --> hole to be filled
//	*hole = stub + idx;
//	++idx;
//
//	stub[idx++] = 0xA8C17BFF; // LDP xzr, x30, [sp], 16
//
//	// b srcpc - to be filled
//
//	stub[idx] = encode_b((uintptr_t)(stub + idx), src_pc + 4);
//	++idx;
//
//	return stub + idx;
//}
//
//#define MAX_INSTRUCTIONS (4096UL)
//static uint32_t* holes[MAX_INSTRUCTIONS] = { 0 };

typedef struct {
	void *addr;        // Address of the instruction (kept for debugging)
	uint32_t original; // Original instruction; Rn is extracted from this by apply_fixups
} fixup_t;

#define MAX_FIXUPS 1024
static fixup_t fixups[MAX_FIXUPS] = { 0 };
static size_t fixup_count = 0;

static void revert_log_fixup(void) {
	memset(fixups, 0, fixup_count * sizeof(fixups[0]));
	fixup_count = 0;
}
static void log_fixup(void *addr, uint32_t original) {
	if (fixup_count >= MAX_FIXUPS) return;
	fixups[fixup_count].addr = addr;
	fixups[fixup_count].original = original;
	++fixup_count;
}

static void apply_fixups(struct cpu_state* state) {
	for (size_t i = 0; i < fixup_count; ++i) {
		uint32_t rn = get_rn(fixups[i].original);
		uintptr_t uaddr = cpu_state_read_base_reg(state, rn);
		cpu_state_write_base_reg(state, rn, (uintptr_t)uaddr2kaddr((void*)uaddr));
	}
	revert_log_fixup();
}

static void* inner_hook_aarch64_instructions(struct simulation_code* sc) {
	if(NULL == sc) return NULL;

	size_t n_instructions = (sc->code_size / 4) + 1; // Add space for 1 additional instruction for our use (we would use this for  a default RET instruction at the end of the testcase)
	
	uint32_t* sim_code = (uint32_t*)sc->code;

	void* hook = sim_code + n_instructions;

	for (size_t i = 0; i < n_instructions; ++i) {
		uintptr_t pc = (uintptr_t)(sim_code + i);
        	uint32_t bl = encode_bl(pc, (uintptr_t)hook);
        	if (0 == bl) {
			return NULL;
		}

		sim_code[i] = bl;
	}

	__builtin___clear_cache((char*)sim_code, (char*)hook);
	return hook;
}
int hook_aarch64_instructions(
		const struct simulation_input* sim_input,
		struct simulation_code* sc,
		void *hook_addr,
		size_t hook_size) {

	if(NULL == sim_input || NULL == sc || NULL == hook_addr) return -1;
	if (sim_input->hdr.code_size % 4 != 0) return -1;

	void* copied_hook_addr = inner_hook_aarch64_instructions(sc);
	if(NULL == copied_hook_addr) return -1;

	memcpy(copied_hook_addr, hook_addr, hook_size); 

	__builtin___clear_cache((char*)copied_hook_addr, (char*)copied_hook_addr + hook_size);
	return 0;
}

static inline uint32_t pc_to_orig_instruction(uintptr_t pc) {
	uintptr_t pc_offset  = pc - (uintptr_t)simulation.sim_code.code;
	if (pc_offset % 4 != 0 || pc_offset >= simulation.sim_input.hdr.code_size) {
		__builtin_trap(); // sanity check
	}

	uint32_t* saved_code_ptr = (uint32_t*)(pc_offset + (uintptr_t)simulation.sim_input.code);
	return *saved_code_ptr;
}

bool out_of_simulation(struct cpu_state* state) {
	if(NULL == state) __builtin_trap(); // sanity_check
	return (state->pc - (uintptr_t)simulation.sim_code.code) >= simulation.sim_input.hdr.code_size;
}

void base_hook_c(struct cpu_state* state) {
	if (NULL == state) __builtin_trap();
	/* apply_fixups MUST run first: it restores kaddr in any Rn that was
	 * translated to uaddr for the previous memory instruction.  Every
	 * subsequent reader of *state (debug snapshot, hook loop, everything)
	 * must see kaddr, never uaddr.  Do not move any read of *state above
	 * this call. */
	apply_fixups(state);
	g_last_hook_cpu_state = *state;
	g_last_hook_orig_instr = out_of_simulation(state) ? 0xd65f03c0 : pc_to_orig_instruction(state->pc);

	struct simulation_state sim_state = { 0 };

	uintptr_t current_ret_address = state->pc; // By default, return to the currently hooked instruction

	state->lr = simulation.return_address; // For the hooks, fake the LR register to point to the return address of the code when it runs wihtout simulation

	sim_state.cpu_state = *state;
	sim_state.memory = simulation.simulation_memory;

	// NOTICE: We write "outside" the simulated code, but we explicitly malloced 1 additional instruction spacew after the simulated code
	memcpy(simulation.sim_code.code, simulation.sim_input.code, simulation.sim_input.hdr.code_size); // restore original code
	*((uint32_t*)((uintptr_t)simulation.sim_code.code + simulation.sim_input.hdr.code_size)) = 0xd65f03c0; // Manually insert RET

	for (size_t i = 0; i < simulation.n_hooks; ++i) {
		// Listeners can change the code flow by returning the next address to execute
		void* continue_simulation_from = simulation.hooks[i](&sim_state);
		if(NULL != continue_simulation_from) {
			current_ret_address = (uintptr_t)continue_simulation_from;
		}
	}

	memcpy(simulation.sim_input.code, simulation.sim_code.code, simulation.sim_input.hdr.code_size); // copy changes from hooks

	if(NULL == inner_hook_aarch64_instructions(&simulation.sim_code)) {
		__builtin_trap();
	}

	simulation.return_address = sim_state.cpu_state.lr; // copy changes from hooks to the LR register

	*state = sim_state.cpu_state;
	state->lr = current_ret_address;

	if(!out_of_simulation(state)) {
		*((uint32_t*)state->pc) = pc_to_orig_instruction(state->pc); // Fix the hooked instruction

	} else {
		*((uint32_t*)state->pc) = 0xd65f03c0; // Manually insert RET
	}

	__builtin___clear_cache((char*)state->pc, (char*)state->pc + 4); /* must precede the translation below */

	/* LAST write to *state — nothing must follow this block. */
	if(is_memory_access(*(uint32_t*)state->pc)) {
		uint32_t rn = get_rn(*(uint32_t*)state->pc);
		uintptr_t kaddr = cpu_state_read_base_reg(state, rn);
		cpu_state_write_base_reg(state, rn, (uintptr_t)kaddr2uaddr((void*)kaddr));
		log_fixup((void*)state->pc, *(uint32_t*)state->pc);
	}
}

void* stdout_print_hook(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;
	fprintf(stderr, "X0 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x0);
	fprintf(stderr, "X1 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x1);
	fprintf(stderr, "X2 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x2);
	fprintf(stderr, "X3 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x3);
	fprintf(stderr, "X4 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x4);
	fprintf(stderr, "X5 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x5);
	fprintf(stderr, "X6 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x6);
	fprintf(stderr, "X7 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x7);
	fprintf(stderr, "X8 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x8);
	fprintf(stderr, "X9 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x9);
	fprintf(stderr, "X10 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x10);
	fprintf(stderr, "X11 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x11);
	fprintf(stderr, "X12 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x12);
	fprintf(stderr, "X13 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x13);
	fprintf(stderr, "X14 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x14);
	fprintf(stderr, "X15 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x15);
	fprintf(stderr, "X16 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x16);
	fprintf(stderr, "X17 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x17);
	fprintf(stderr, "X18 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x18);
	fprintf(stderr, "X19 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x19);
	fprintf(stderr, "X20 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x20);
	fprintf(stderr, "X21 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x21);
	fprintf(stderr, "X22 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x22);
	fprintf(stderr, "X23 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x23);
	fprintf(stderr, "X24 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x24);
	fprintf(stderr, "X25 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x25);
	fprintf(stderr, "X26 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x26);
	fprintf(stderr, "X27 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x27);
	fprintf(stderr, "X28 = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x28);
	fprintf(stderr, "X29 (FP) = 0x%016" PRIxPTR "\n", sim_state->cpu_state.gprs.x29);
	fprintf(stderr, "X30 (LR) = 0x%016" PRIxPTR "\n", sim_state->cpu_state.lr);
	fprintf(stderr, "SP = 0x%016" PRIxPTR "\n", sim_state->cpu_state.sp);
	fprintf(stderr, "PC = 0x%016" PRIxPTR "\n", sim_state->cpu_state.pc);
	fprintf(stderr, "\tINSTR = 0x%08x\n", *(uint32_t*)sim_state->cpu_state.pc);
	uintptr_t nzcv = sim_state->cpu_state.nzcv;
	uint32_t N = (nzcv >> 31) & 1;
	uint32_t Z = (nzcv >> 30) & 1;
	uint32_t C = (nzcv >> 29) & 1;
	uint32_t V = (nzcv >> 28) & 1;
	fprintf(stderr, "NZCV   = 0x%016" PRIxPTR "  [N=%u Z=%u C=%u V=%u]  (%c%c%c%c)\n",
           nzcv,
           N, Z, C, V,
           N ? 'N' : '-',
           Z ? 'Z' : '-',
           C ? 'C' : '-',
           V ? 'V' : '-');
	return NULL;
}

void* handle_ret_hook(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;
	if(0xd65f03c0 == *(uint32_t*)sim_state->cpu_state.pc) { // Identify RET
	       	return (void*)sim_state->cpu_state.lr;
	}
	return NULL;
}

