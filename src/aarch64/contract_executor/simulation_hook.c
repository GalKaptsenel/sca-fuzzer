#include "simulation_hook.h"
#include "simulation.h"
#include "simulation_execution_clause_hook.h"
#include "instruction_encodings.h"
#include "call_stack.h"
#include "mte_tag_plugin.h"

#include "simulation_output.h"

volatile struct cpu_state g_last_hook_cpu_state;
volatile uint32_t g_last_hook_orig_instr;

/* Architectural stack pointer, tracked in software. The real (host) SP is used by the per-instruction
 * trampoline and is unrelated to the test case's SP, so we model the sandbox SP here: it is seeded to
 * the sandbox stack top per test case, presented to the hooks as cpu_state.sp, and updated by the
 * emulated frame spills. It rides cpu_state.sp through the speculation checkpoint, so speculative
 * spills roll back with their window and SP never drifts off the stack. */
static uintptr_t g_arch_sp = 0;
void arch_sp_reset(uintptr_t stack_top) { g_arch_sp = stack_top; }

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
 * Post-instruction repairs queued by base_hook_c, applied by the next hook. Each entry is ONE action:
 *   REG - restore a base register: reg = orig + (current - uaddr); preserves the exact top byte / MTE
 *         tag on a non-writeback access and the writeback increment otherwise.
 *   MEM - restore `size` bytes of `value` at `uaddr` (undo a native store that wrote the uaddr).
 * One instruction may queue several (a base-register restore plus one MEM restore per stored element
 * whose data register aliased the base). Held in a cyclic buffer, drained tail -> head.
 */
typedef enum { FIXUP_REG = 1, FIXUP_MEM } fixup_kind_t;

typedef struct {
	fixup_kind_t kind;
	union {
		struct { uint32_t reg; uintptr_t orig; uintptr_t uaddr; } reg;
		struct { void *uaddr; uint64_t value; uint32_t size; }    mem;
	};
} fixup_t;

#define MAX_FIXUPS 1024
static fixup_t fixups[MAX_FIXUPS];
static size_t fix_head = 0;   /* next slot to write */
static size_t fix_tail = 0;   /* next slot to apply */

static void push_fixup(fixup_t f) {
	size_t next = (fix_head + 1) % MAX_FIXUPS;
	if (next == fix_tail) __builtin_trap();
	fixups[fix_head] = f;
	fix_head = next;
}
static void log_reg_fixup(uint32_t reg, uintptr_t orig, uintptr_t uaddr) {
	fixup_t f = { .kind = FIXUP_REG };
	f.reg.reg = reg; f.reg.orig = orig; f.reg.uaddr = uaddr;
	push_fixup(f);
}
static void log_mem_fixup(void *uaddr, uint64_t value, uint32_t size) {
	fixup_t f = { .kind = FIXUP_MEM };
	f.mem.uaddr = uaddr; f.mem.value = value; f.mem.size = size;
	push_fixup(f);
}

static void apply_fixups(struct cpu_state* state) {
	while (fix_tail != fix_head) {                 /* drain the ring, wrapping, until empty */
		fixup_t* f = &fixups[fix_tail];
		if (FIXUP_MEM == f->kind) {
			memcpy(f->mem.uaddr, &f->mem.value, f->mem.size);
		} else if (FIXUP_REG == f->kind) {
			uintptr_t cur = cpu_state_read_base_reg(state, f->reg.reg);
			cpu_state_write_base_reg(state, f->reg.reg, f->reg.orig + (cur - f->reg.uaddr));
		}
		fix_tail = (fix_tail + 1) % MAX_FIXUPS;
	}
}

/* Sign-extend the low `size` bytes of v to 64 bits. */
static uint64_t sext_from(uint64_t v, uint32_t size) {
	uint32_t bits = size * 8;
	if (bits >= 64) return v;
	uint64_t m = 1ull << (bits - 1);
	return (v ^ m) - m;
}

/* When an atomic store's source register aliases the base (which we set to uaddr for native
 * execution), the native store wrote a uaddr-derived value. Recompute the value the store WOULD have
 * written with kaddr, so the mem fixup can restore it. `uea` is the translated (sim-memory) address,
 * still holding the pre-op value. Returns 1 and sets *out when a fix is needed. Covers SWP, the LSE
 * arithmetic RMW family (LD/ST<op>), and CAS/CASP element 0. Register 31 is XZR (value 0). */
static int atomic_kaddr_store_value(uint32_t inst, uint32_t rn, uint64_t kaddr, void* uea,
                                    uint32_t size, struct cpu_state* state, uint64_t* out) {
	uint64_t mask = (size >= 8) ? ~0ull : ((1ull << (size * 8)) - 1);
	uint64_t old = 0; memcpy(&old, uea, size); old &= mask;

	if (is_regular_load_store(inst)) {            /* LSE: SWP / LD|ST<op>, source = Rs[20:16] */
		uint32_t rs = (inst >> 16) & 0x1F;
		if (rs != rn || rs == AARCH64_XZR_REG) return 0;   /* not aliased, or XZR source */
		uint64_t s = kaddr & mask;
		uint32_t o3 = (inst >> 15) & 1, opc = (inst >> 12) & 7;
		uint64_t v;
		if (o3 && opc == 0)      v = s;                                              /* SWP    */
		else switch (opc) {
			case 0: v = old + s; break;                                         /* LDADD  */
			case 1: v = old & ~s; break;                                        /* LDCLR  */
			case 2: v = old ^ s; break;                                         /* LDEOR  */
			case 3: v = old | s; break;                                         /* LDSET  */
			case 4: v = sext_from(old,size) > sext_from(s,size) ? old : s; break;/* LDSMAX */
			case 5: v = sext_from(old,size) < sext_from(s,size) ? old : s; break;/* LDSMIN */
			case 6: v = old > s ? old : s; break;                               /* LDUMAX */
			default: v = old < s ? old : s; break;                              /* LDUMIN */
		}
		*out = v & mask; return 1;
	}

	/* CAS / CASP: store Rt (element 0) iff [EA] == Rs. */
	uint32_t rs = (inst >> 16) & 0x1F, rt = inst & 0x1F;
	if (rs != rn && rt != rn) return 0;
	uint64_t rsv = ((rs == rn) ? kaddr : (rs == AARCH64_XZR_REG ? 0 : cpu_state_read_base_reg(state, rs))) & mask;
	uint64_t rtv = ((rt == rn) ? kaddr : (rt == AARCH64_XZR_REG ? 0 : cpu_state_read_base_reg(state, rt))) & mask;
	*out = (old == rsv) ? rtv : old; return 1;
}

static void* inner_hook_aarch64_instructions(struct simulation_code* sc) {
	if(NULL == sc) return NULL;

	uint32_t* sim_code = (uint32_t*)sc->code;
	size_t instr_words = sc->code_size / 4;
	size_t end_slot    = (sc->code_size + sc->data_size) / 4;   // one word past code + tables

	// Executable layout: [instructions][read-only tables][end slot (4B)][trampoline]. The trampoline
	// (and the terminal slot before it) live past code + data, in additional_space_alloc, so hooking
	// only ever rewrites instruction words and the terminal slot, never the dispatch tables.
	void* hook = (uint8_t*)sc->code + (end_slot + 1) * 4;

	for (size_t i = 0; i < instr_words; ++i) {
		uintptr_t pc = (uintptr_t)(sim_code + i);
		uint32_t bl = encode_bl(pc, (uintptr_t)hook);
		if (0 == bl) {
			return NULL;
		}
		sim_code[i] = bl;
	}

	// Keep the read-only dispatch tables intact for LDRSW: base_hook_c restores only the instruction
	// region, so refresh the tables from the original payload here (a no-op when there are none).
	if (0 != sc->data_size) {
		memcpy((uint8_t*)sc->code + sc->code_size,
		       (const uint8_t*)simulation.sim_input.code + sc->code_size, sc->data_size);
	}

	// Hook the terminal slot just past code + tables (when data_size==0 this is the original "+1" slot):
	// B .test_case_exit lands here, so the trampoline fires, out_of_simulation trips and the trace ends —
	// otherwise execution would fall through into a stale native RET and loop.
	uint32_t end_bl = encode_bl((uintptr_t)(sim_code + end_slot), (uintptr_t)hook);
	if (0 == end_bl) {
		return NULL;
	}
	sim_code[end_slot] = end_bl;

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

/* True when addr falls inside the executable image (instructions + read-only dispatch tables). A load
 * of such an address is the ADR+LDRSW dispatch sequence reading its own jump table — a code/rodata read,
 * not a sandbox access — so it is neither rebased into the sandbox nor recorded in the contract. */
static inline int addr_in_sim_code(uintptr_t addr) {
	uintptr_t lo = (uintptr_t)simulation.sim_code.code;
	uintptr_t hi = lo + simulation.sim_code.code_size + simulation.sim_code.data_size;
	return addr >= lo && addr < hi;
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
	sim_state.cpu_state.sp = g_arch_sp;   /* hooks see the architectural sandbox SP, not the host SP */
	sim_state.memory = simulation.simulation_memory;

	// Restore the original instructions; the read-only tables sit after them and are never touched here.
	// The default-RET slot follows code + data (in the additional scratch), so it can't clobber a table.
	memcpy(simulation.sim_code.code, simulation.sim_input.code, simulation.sim_input.hdr.code_size); // restore original code
	*((uint32_t*)((uintptr_t)simulation.sim_code.code + simulation.sim_input.hdr.code_size
	              + simulation.sim_input.hdr.data_size)) = 0xd65f03c0; // Manually insert RET past code+data

	for (size_t i = 0; i < simulation.n_hooks; ++i) {
		// Listeners can change the code flow by returning the next address to execute
		void* continue_simulation_from = simulation.hooks[i](&sim_state);
		if(NULL != continue_simulation_from) {
			current_ret_address = (uintptr_t)continue_simulation_from;
		}
	}

	g_arch_sp = sim_state.cpu_state.sp;   /* persist frame-spill writebacks and speculative rollbacks */

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
	/* SP-based accesses are only ever function-frame spills (the generator never emits sp as a data
	 * base): leave them native on the host stack instead of rebasing into the sandbox buffer — they
	 * are harness instrumentation, kept out of the contract just like the kernel keeps them in
	 * upper_overflow, outside the probed sets. */
	if(is_memory_access(*(uint32_t*)state->pc) && !mte_is_mem_tag_access(*(uint32_t*)state->pc)
	   && !is_literal_pc_relative(*(uint32_t*)state->pc)
	   && get_rn(*(uint32_t*)state->pc) != AARCH64_SP_REG) {
		uint32_t inst = *(uint32_t*)state->pc;
		uint32_t rn = get_rn(inst);
		uintptr_t base_orig = cpu_state_read_base_reg(state, rn);
		uintptr_t restore_val = base_orig;

		trace_cpu_state_t tcs = { 0 };
		for (uint32_t r = 0; r < 31; ++r) { tcs.gpr[r] = cpu_state_read_base_reg(state, r); }
		tcs.sp = state->sp; tcs.pc = state->pc;
		mem_access_info_t mi = parse_memory_access_instruction(inst, &tcs);
		uintptr_t ea = (uintptr_t)mi.effective_address;

		/* A load that writes its result into Rn (Rt or, for a pair, Rt2) leaves loaded data there,
		 * not a pointer: don't restore it. */
		int load_aliases = mi.is_load && rn != AARCH64_SP_REG &&
		    (mi.target_register == rn || (mi.is_pair && mi.rt2_register == rn));

		/* MTE-test mode (not SP): give the pointer the accessed cell's tag before the access, as the
		 * genuine ADDG does, so a store of it writes the tagged value. */
		if (!load_aliases && mte_tagmem_active() && rn != AARCH64_SP_REG) {
			uint8_t cell_tag = mte_tagmem_tag_at(ea);
			restore_val = (base_orig & ~(0xFull << 56)) | ((uintptr_t)cell_tag << 56);
		}

		/* Rebase so the hardware-computed EA lands exactly on the translated (clean) EA; the native
		 * access then never relies on EL0 TBI to drop a stray top byte. A load from the executable image
		 * (the ADR+LDRSW dispatch reading its own jump table) already points into sim_code, so it needs no
		 * rebase — leave the base as-is, exactly like the SP frame-spill exemption. */
		if (!addr_in_sim_code(ea)) {
			uintptr_t new_base = (uintptr_t)kaddr2uaddr((void*)ea) - ea + base_orig;
			cpu_state_write_base_reg(state, rn, new_base);
			if (!load_aliases) log_reg_fixup(rn, restore_val, new_base);
		}

		/* A store whose data register aliases the base wrote the uaddr, not the kaddr; restore those
		 * bytes. Queued per stored element, so every element of a pair (or any wider store) is covered.
		 * Regular / pair / exclusive stores write Rt / Rt2 directly (reg 31 = XZR, stores 0); atomics
		 * write a computed value keyed on Rs, so recompute it with kaddr. */
		if (mi.is_store) {
			if (mi.is_atomic) {
				void* uea = kaddr2uaddr((void*)mi.effective_address);
				uint64_t fixed;
				if (rn != AARCH64_SP_REG && atomic_kaddr_store_value(inst, rn, (uint64_t)restore_val, uea,
				                                         (uint32_t)mi.data_size, state, &fixed))
					log_mem_fixup(uea, fixed, (uint32_t)mi.data_size);
			} else {
				if (mi.target_register == rn && rn != AARCH64_SP_REG)
					log_mem_fixup(kaddr2uaddr((void*)mi.effective_address),
					              (uint64_t)restore_val, (uint32_t)mi.data_size);
				if (mi.is_pair && mi.rt2_register == rn && rn != AARCH64_SP_REG)
					log_mem_fixup(kaddr2uaddr((void*)(mi.effective_address + mi.data_size)),
					              (uint64_t)restore_val, (uint32_t)mi.data_size);
			}
		}
	}
}

void* handle_ret_hook(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;
	uint32_t insn = *(uint32_t*)sim_state->cpu_state.pc;
	branch_type_t bt = classify_branch(insn);

	// A call pushes its architectural return address (the instruction right after it) onto the call
	// stack, then branches to the callee natively — so return NULL and let the native BL/BLR run.
	if(BRANCH_BL == bt || BRANCH_BLR == bt) {
		call_stack_push(sim_state->cpu_state.pc + 4);
		return NULL;
	}

	// A return pops the matching call's address and continues there. With no active call (empty stack)
	// it falls back to the simulation's top-level return address — the harness continuation — which
	// mirrors the kernel seeding LR to the test-case exit.
	if(0xd65f03c0 == insn) { // RET
		uintptr_t target;
		if(!call_stack_pop(&target)) {
			target = simulation.return_address;
		}
		return (void*)target;
	}

	// SP-based memory accesses are function-frame spills (the generator never emits sp as a data base).
	// Emulate them against the sandbox stack rather than running them natively (which would use the CE's
	// host stack): move X30 to/from sandbox memory and adjust the architectural SP. The load/store is
	// still recorded by the trace hook, so the spill appears in the contract footprint exactly as the
	// kernel's does on hardware; then advance past the instruction (return pc+4) so it is not executed
	// natively. cpu_state.sp is checkpointed with the window, so a speculative spill rolls back and SP
	// never drifts off the stack.
	if(is_memory_access(insn) && !is_literal_pc_relative(insn) &&
	   AARCH64_SP_REG == get_rn(insn)) {
		trace_cpu_state_t tcs = { 0 };
		for (uint32_t r = 0; r < 31; ++r) { tcs.gpr[r] = cpu_state_read_base_reg(&sim_state->cpu_state, r); }
		tcs.sp = sim_state->cpu_state.sp; tcs.pc = sim_state->cpu_state.pc;
		mem_access_info_t mi = parse_memory_access_instruction(insn, &tcs);
		void* uea = kaddr2uaddr((void*)(uintptr_t)mi.effective_address);
		if (mi.is_store) {
			uint64_t v = cpu_state_read_base_reg(&sim_state->cpu_state, mi.target_register);
			memcpy(uea, &v, (size_t)mi.data_size);
		} else {
			uint64_t v = 0;
			memcpy(&v, uea, (size_t)mi.data_size);
			cpu_state_write_base_reg(&sim_state->cpu_state, mi.target_register, v);
		}
		// Writeback: pre- and post-indexed frame ops both move SP by their signed imm9 (STR X30,[SP,#-16]!
		// and LDR X30,[SP],#16). A plain signed-offset form has no writeback.
		if (is_pre_index(insn) || is_post_index(insn)) {
			int64_t imm9 = (int64_t)((insn >> 12) & 0x1FF);
			if (imm9 & 0x100) { imm9 |= ~(int64_t)0x1FF; }
			sim_state->cpu_state.sp += (uintptr_t)imm9;
		}
		return (void*)(sim_state->cpu_state.pc + 4);
	}
	return NULL;
}

