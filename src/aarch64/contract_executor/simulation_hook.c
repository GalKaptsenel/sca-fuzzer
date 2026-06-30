#include "simulation_hook.h"
#include "simulation.h"
#include "simulation_execution_clause_hook.h"
#include "instruction_encodings.h"
#include "mte_tag_plugin.h"

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

/*
 * Post-instruction repairs queued by base_hook_c and applied by the next hook's apply_fixups. A
 * fixup is composable: it may carry a REGISTER fixup (rewrite a base register — revert its
 * kaddr<->uaddr translation and/or correct its MTE tag) and/or a MEMORY fixup (restore bytes a
 * native store clobbered). What to do is decided at log time; apply_fixups just applies it.
 */
typedef struct {
	void *addr;                  /* instruction address (debug only) */
	struct {
		int      apply;      /* rewrite `reg` */
		uint32_t reg;        /* base register index */
		int      revert;     /* uaddr -> kaddr */
		int      retag;      /* overwrite tag bits [59:56] with `tag` */
		uint8_t  tag;
	} reg_fix;
	struct {
		void    *uaddr;      /* NULL = none; else restore `size` bytes of `value` here */
		uint64_t value;
		uint32_t size;
	} mem_fix;
} fixup_t;

#define MAX_FIXUPS 1024
static fixup_t fixups[MAX_FIXUPS] = { 0 };
static size_t fixup_count = 0;

static void revert_log_fixup(void) {
	memset(fixups, 0, fixup_count * sizeof(fixups[0]));
	fixup_count = 0;
}
static void log_fixup(fixup_t f) {
	if (fixup_count >= MAX_FIXUPS) return;
	fixups[fixup_count++] = f;
}

static void apply_fixups(struct cpu_state* state) {
	for (size_t i = 0; i < fixup_count; ++i) {
		fixup_t* f = &fixups[i];
		if (NULL != f->mem_fix.uaddr) {
			memcpy(f->mem_fix.uaddr, &f->mem_fix.value, f->mem_fix.size);
		}
		if (f->reg_fix.apply) {
			uintptr_t v = cpu_state_read_base_reg(state, f->reg_fix.reg);
			if (f->reg_fix.revert) {
				v = (uintptr_t)uaddr2kaddr((void*)v);
			}
			if (f->reg_fix.retag) {
				v = (v & ~(0xFull << 56)) | ((uint64_t)f->reg_fix.tag << 56);
			}
			cpu_state_write_base_reg(state, f->reg_fix.reg, v);
		}
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

	/* LAST write to *state — nothing must follow this block. MTE memory-tag ops (LDG/STG/STGP) are
	 * software-emulated and skipped from native execution, so they must NOT have their base register
	 * translated kaddr->uaddr here (the emulator already produced the architectural result from the
	 * kaddr; translating would leave a uaddr in the base register). */
	if(is_memory_access(*(uint32_t*)state->pc) && !mte_is_mem_tag_access(*(uint32_t*)state->pc)
	   && !is_literal_pc_relative(*(uint32_t*)state->pc)) {
		uint32_t inst = *(uint32_t*)state->pc;
		uint32_t rn = get_rn(inst);
		uintptr_t kaddr = cpu_state_read_base_reg(state, rn);

		fixup_t f = { 0 };
		f.addr = (void*)state->pc;
		f.reg_fix.reg = rn;

		/* A load that writes its result into Rn leaves loaded data there, not a pointer: leave it. */
		int load_aliases = is_load(inst) &&
		    (get_rt(inst) == rn || (is_pair_load_store(inst) && get_rt2(inst) == rn));
		f.reg_fix.apply  = !load_aliases;
		f.reg_fix.revert = !load_aliases;

		/* MTE-test mode (not SP): give the pointer the accessed cell's tag before the access, as the
		 * genuine ADDG does, so a store of it writes the tagged value. The cell lookup masks the tag. */
		uint8_t cell_tag = 0;
		int retag = !load_aliases && mte_tagmem_active() && rn != 31;
		if (retag) {
			trace_cpu_state_t tcs0 = { 0 };
			for (uint32_t r = 0; r < 31; ++r) { tcs0.gpr[r] = cpu_state_read_base_reg(state, r); }
			tcs0.sp = state->sp;
			tcs0.pc = state->pc;
			mem_access_info_t ea = parse_memory_access_instruction(inst, &tcs0);
			cell_tag = mte_tagmem_tag_at((uintptr_t)ea.effective_address);
			kaddr = (kaddr & ~(0xFull << 56)) | ((uintptr_t)cell_tag << 56);
			f.reg_fix.retag = 1;   /* the revert recovers the canonical kaddr (TBI dropped the tag) */
			f.reg_fix.tag = cell_tag;
		}

		cpu_state_write_base_reg(state, rn, (uintptr_t)kaddr2uaddr((void*)kaddr));

		/* A store whose data register aliases the base writes kaddr (the tagged architectural pointer),
		 * not the uaddr; restore those bytes after the native store. */
		if (is_store(inst)) {
			int pair = is_pair_load_store(inst);
			int rt_aliases = (get_rt(inst) == rn);
			int rt2_aliases = (pair && get_rt2(inst) == rn);
			if (rt_aliases || rt2_aliases) {
				trace_cpu_state_t tcs = { 0 };
				for (uint32_t r = 0; r < 31; ++r) { tcs.gpr[r] = cpu_state_read_base_reg(state, r); }
				tcs.sp = state->sp; tcs.pc = state->pc;
				mem_access_info_t mi = parse_memory_access_instruction(inst, &tcs);
				f.mem_fix.value = (uint64_t)kaddr;
				f.mem_fix.size  = (uint32_t)mi.data_size;
				f.mem_fix.uaddr = rt_aliases ? (void*)mi.effective_address
				                             : (void*)(mi.effective_address + mi.data_size);
			}
		}
		log_fixup(f);
	}
}

void* handle_ret_hook(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;
	if(0xd65f03c0 == *(uint32_t*)sim_state->cpu_state.pc) { // Identify RET
	       	return (void*)sim_state->cpu_state.lr;
	}
	return NULL;
}

