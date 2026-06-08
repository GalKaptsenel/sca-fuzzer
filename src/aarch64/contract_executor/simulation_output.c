#include "simulation.h"
#include "simulation_output.h"
#include "simulation_hook.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>

static contract_trace_t* trace_log = NULL;
static size_t current_log_index = 0;
static size_t max_log_index = 0;

static inline contract_trace_t* alloc_contract_trace(size_t num_entries) {
	size_t total_alloc_size = sizeof(contract_trace_t) + num_entries * sizeof(instr_trace_entry_t);
	contract_trace_t* trace = (contract_trace_t*)malloc(total_alloc_size);
	if (NULL == trace) return NULL;
	memset(trace, 0, total_alloc_size);
	return trace;
}

static inline void free_contract_trace(contract_trace_t* trace) {
	if (NULL == trace) return;
	free(trace);
}

static inline uint64_t read_reg(const trace_cpu_state_t *s, unsigned r) {
	if (r == 31) return s->sp;
	return s->gpr[r];
}

static uint64_t decode_imm9(uint32_t inst) {
	int64_t imm9 = (inst >> 12) & 0x1FF;
	if (imm9 & 0x100) imm9 |= ~0x1FF;
	return (uint64_t)imm9;
}

static uint64_t decode_imm7(uint32_t inst) {
	int64_t imm7 = (inst >> 15) & 0x7F;
	if (imm7 & 0x40) imm7 |= ~0x7F;
	return (uint64_t)imm7;
}

static uint64_t decode_imm19(uint32_t inst) {
	int64_t imm19 = (inst >> 5) & 0x7FFFF;
	if (imm19 & 0x40000) imm19 |= ~0x7FFFF;
	return (uint64_t)imm19;
}

static inline int64_t get_offset(uint32_t inst) {
	if(is_pre_index(inst) || is_post_index(inst)) {
		return (int64_t)decode_imm9(inst);
	} else if(is_unsigned_offset(inst)) {
		uint64_t pimm = (inst >> 10) & 0xFFF;
		return pimm * access_size(inst);
	}
	__builtin_trap();
}

static int64_t signextend(size_t orig_len, size_t dest_len, int64_t value) {
	(void)dest_len; /* always extends to 64 bits (int64_t) */
	if (orig_len == 0 || orig_len >= 64) { return value; }
	int shift = 64 - (int)orig_len;
	return (int64_t)((uint64_t)value << shift) >> shift;
}

void* kaddr2uaddr(void* kaddr) {
	void* uaddr = kaddr;
	if(CONFIG_FLAG_REQ_MEM_BASE_VIRT & simulation.sim_input.hdr.config.flags) {
		uintptr_t kbase  = simulation.sim_input.hdr.config.requested_mem_base_virt;
		uintptr_t kaddr_ = (uintptr_t)kaddr;
		/* No bounds check: pre-indexed instructions use a base register that is
		 * offset-adjusted (e.g. STR X4, [X5, #0x82]! where X5 = kbase-0x82).
		 * The arithmetic translation is correct regardless; truly OOB accesses
		 * will fault on the resulting user pointer and get caught by SIGSEGV. */
		uaddr = (char*)simulation.simulation_memory + (kaddr_ - kbase);
	}
	return uaddr;
}

void* uaddr2kaddr(void* uaddr) {
	void* kaddr = uaddr;
	if(CONFIG_FLAG_REQ_MEM_BASE_VIRT & simulation.sim_input.hdr.config.flags) {
		kaddr = (char*)simulation.sim_input.hdr.config.requested_mem_base_virt + ((uintptr_t)uaddr - (uintptr_t)simulation.simulation_memory);
	}
	return kaddr;
}


static int parse_memory_access_instruction(
	uint32_t inst,
	const trace_cpu_state_t *state,
	uintptr_t* effective_address,
	uint32_t* target_register,
	uint32_t* base_register,
	uint32_t* index_register,
	uint32_t* rt2_register,
	bool* is_load_inst,
	bool* is_store_inst,
	uint64_t* data_size) {

	if(effective_address) *effective_address = (uintptr_t)-1;
	if(target_register) *target_register = (uint32_t)-1;
	if(base_register) *base_register = (uint32_t)-1;
	if(index_register) *index_register = (uint32_t)-1;
	if(is_load_inst) *is_load_inst = false;
	if(is_store_inst) *is_store_inst = false;
	if(rt2_register) *rt2_register = (uint32_t)-1;
	if(data_size) *data_size = 0;

	if(!is_memory_access(inst)) return 0;

	if(target_register) *target_register = get_rt(inst);

	uintptr_t effective_address_computed = (uintptr_t)-1;
	if (is_regular_load_store(inst)) {
		if(is_load_inst) *is_load_inst = is_load(inst);
		if(is_store_inst) *is_store_inst = is_store(inst);

		uint32_t Rn   = get_rn(inst);
		uint64_t base = read_reg(state, Rn);
		if(base_register) *base_register = Rn;

		if(data_size) *data_size = access_size(inst);

		if(is_reg_offset(inst)) {

			size_t shift_amount = decode_amount_log2(inst);		
			uint64_t Rm = get_rm(inst);
			if(NULL != index_register) {
				*index_register = Rm;
			}

			bool is_signed = false;
			size_t src_len = 0;
			switch(decode_extend(inst)) {
				case EXT_UXTB: 
					src_len = 8;
					is_signed = false;
					break;
				case EXT_UXTH: 
					src_len = 16;
					is_signed = false;
					break;
				case EXT_UXTW: 
					src_len = 32;
					is_signed = false;
					break;
				case EXT_UXTX: 
					src_len = 64;
					is_signed = false;
					break;
				case EXT_SXTB: 
					src_len = 8;
					is_signed = true;
					break;
				case EXT_SXTH: 
					src_len = 16;
					is_signed = true;
					break;
				case EXT_SXTW: 
					src_len = 32;
					is_signed = true;
					break;
				case EXT_SXTX:
					src_len = 64;
					is_signed = true;
					break;
				default:
					__builtin_unreachable();
			}
			uint64_t rm_val = read_reg(state, Rm);
			if(is_32bit_rm(inst)) {
				rm_val &= ((1ull << 32) - 1);
			}

			/* ARM ExtendReg: mask to src_len bits, sign/zero-extend to 64, then shift */
			uint64_t mask_src = (src_len < 64) ? ((1ull << src_len) - 1) : (uint64_t)-1;
			uint64_t rm_narrow = rm_val & mask_src;
			if (is_signed) {
				int64_t sext_val = signextend(src_len, 64, (int64_t)rm_narrow);
				effective_address_computed = (uint64_t)((int64_t)base + (int64_t)((uint64_t)sext_val << shift_amount));
			} else {
				effective_address_computed = base + (rm_narrow << shift_amount);
			}

		} else if(is_immediate_offset(inst)){
			if(is_unsigned_offset(inst)) {
				uint64_t offset = (uint64_t)get_offset(inst);
				effective_address_computed = base + offset;
			} else if (is_pre_index(inst)){
				int64_t offset = get_offset(inst);
				effective_address_computed = (uint64_t)((int64_t)base + offset);
			} else {
				effective_address_computed = base;
			}
		} else {
			__builtin_unreachable();
		}
	} else if(is_pair_load_store(inst)) {
		uint32_t Rn   = get_rn(inst);
		uint64_t base = read_reg(state, Rn);
		if(base_register) *base_register = Rn;

		if(is_load_inst) *is_load_inst = is_load(inst);
		if(is_store_inst) *is_store_inst = is_store(inst);
		if(data_size) *data_size = pair_element_access_size(inst);

		uint32_t Rt2 = get_rt2(inst);
		if(rt2_register) *rt2_register = Rt2;

		if(is_pair_pre_index(inst) || is_pair_signed_offset(inst)) {
			int64_t offset = decode_imm7(inst) * pair_element_access_size(inst);
			effective_address_computed = (uint64_t)((int64_t)base + offset);
		} else if(is_pair_post_index(inst)) {
			effective_address_computed = base;
		} else {
			__builtin_unreachable();
		}
	} else if(is_literal_pc_relative(inst)) {
		uint64_t sz = literal_pc_relative_access_size(inst);
		if (0 == sz) { return 0; } // PRFM: prefetch hint, not a real memory load
		if (is_store_inst) { *is_store_inst = false; }
		if (is_load_inst) { *is_load_inst = 1; }
		if (data_size) { *data_size = sz; }
		int64_t offset = decode_imm19(inst) * 4;
		effective_address_computed = (uint64_t)((int64_t)state->pc + offset);
	}

	if(effective_address) *effective_address = effective_address_computed;

	return 1;
}

static instr_trace_entry_t* log_sim_state(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;
	if(out_of_simulation(&sim_state->cpu_state)) return NULL;

	if(NULL == trace_log) init_trace_log(simulation.sim_input.hdr.code_size);

	size_t current_index = current_log_index;
	if(max_log_index <= current_index) {
		fprintf(stderr, "Unable to log more instructions, reached maximum log size!\n");
		return NULL;
	}

	instr_trace_entry_t* entry = (instr_trace_entry_t*)(trace_log + 1) + current_index;

	++current_log_index;
	trace_log->entry_count = current_log_index; 
	entry->metadata.instr_index = current_index;
	entry->cpu.gpr[0] = sim_state->cpu_state.gprs.x0;
	entry->cpu.gpr[1] = sim_state->cpu_state.gprs.x1;
	entry->cpu.gpr[2] = sim_state->cpu_state.gprs.x2;
	entry->cpu.gpr[3] = sim_state->cpu_state.gprs.x3;
	entry->cpu.gpr[4] = sim_state->cpu_state.gprs.x4;
	entry->cpu.gpr[5] = sim_state->cpu_state.gprs.x5;
	entry->cpu.gpr[6] = sim_state->cpu_state.gprs.x6;
	entry->cpu.gpr[7] = sim_state->cpu_state.gprs.x7;
	entry->cpu.gpr[8] = sim_state->cpu_state.gprs.x8;
	entry->cpu.gpr[9] = sim_state->cpu_state.gprs.x9;
	entry->cpu.gpr[10] = sim_state->cpu_state.gprs.x10;
	entry->cpu.gpr[11] = sim_state->cpu_state.gprs.x11;
	entry->cpu.gpr[12] = sim_state->cpu_state.gprs.x12;
	entry->cpu.gpr[13] = sim_state->cpu_state.gprs.x13;
	entry->cpu.gpr[14] = sim_state->cpu_state.gprs.x14;
	entry->cpu.gpr[15] = sim_state->cpu_state.gprs.x15;
	entry->cpu.gpr[16] = sim_state->cpu_state.gprs.x16;
	entry->cpu.gpr[17] = sim_state->cpu_state.gprs.x17;
	entry->cpu.gpr[18] = sim_state->cpu_state.gprs.x18;
	entry->cpu.gpr[19] = sim_state->cpu_state.gprs.x19;
	entry->cpu.gpr[20] = sim_state->cpu_state.gprs.x20;
	entry->cpu.gpr[21] = sim_state->cpu_state.gprs.x21;
	entry->cpu.gpr[22] = sim_state->cpu_state.gprs.x22;
	entry->cpu.gpr[23] = sim_state->cpu_state.gprs.x23;
	entry->cpu.gpr[24] = sim_state->cpu_state.gprs.x24;
	entry->cpu.gpr[25] = sim_state->cpu_state.gprs.x25;
	entry->cpu.gpr[26] = sim_state->cpu_state.gprs.x26;
	entry->cpu.gpr[27] = sim_state->cpu_state.gprs.x27;
	entry->cpu.gpr[28] = sim_state->cpu_state.gprs.x28;
	entry->cpu.gpr[29] = sim_state->cpu_state.gprs.x29;
	entry->cpu.gpr[30] = sim_state->cpu_state.lr;
	entry->cpu.sp = sim_state->cpu_state.sp;
	entry->cpu.pc = sim_state->cpu_state.pc;
	entry->cpu.nzcv = sim_state->cpu_state.nzcv;
	/* Read from sim_input.code (original, never patched during hook execution)
	 * rather than from the live sim_code.code, which pac_sign_hook / auth_verify_hook
	 * may have already NOP'd.  sim_input.code is not updated until after all hooks
	 * return (base_hook_c copies sim_code → sim_input after the hook loop). */
	uintptr_t pc_offset = sim_state->cpu_state.pc - (uintptr_t)simulation.sim_code.code;
	entry->cpu.encoding = *(uint32_t*)(simulation.sim_input.code + pc_offset);
	entry->cpu.extra_data_size = 0;
	// no need to allocate space for extra data, because it is of size 0 for now

	uint32_t target_register = (uint32_t)-1;
	uint32_t target_register2 = (uint32_t)-1;
	int has_mem_access = parse_memory_access_instruction(entry->cpu.encoding, &entry->cpu,
			&entry->metadata.memory_access.effective_address,
			&target_register,
			NULL,
			NULL,
			&target_register2,
			NULL,
			(bool*)&entry->metadata.memory_access.is_write,
			&entry->metadata.memory_access.element_size
	);

	entry->metadata.has_memory_access = has_mem_access ? 1 : 0;

	if(has_mem_access) {
		assert((uintptr_t)-1 != entry->metadata.memory_access.effective_address && "Effective address should have been set by parse_memory_access_instruction function");

		void* uaddr = kaddr2uaddr((void*)entry->metadata.memory_access.effective_address);

		uint64_t elem_sz = entry->metadata.memory_access.element_size;

		/* Read exactly elem_sz bytes. The sandbox allocation includes a full page of
		 * padding after mem_size (see main.c) so boundary overflows are safe. */
		uint64_t value_64bit = 0;
		memcpy(&value_64bit, uaddr, (size_t)elem_sz);
		uint64_t write_mask = 0;
		switch(entry->metadata.memory_access.element_size) {
			case 1: write_mask = 0xFF; break;
			case 2: write_mask = 0xFFFF; break;
			case 4: write_mask = 0xFFFFFFFF; break;
			case 8: write_mask = 0xFFFFFFFFFFFFFFFF; break;
			default: {

				fprintf(stderr, "[C] log_instr_hook: got to an unreachable!!!\n");
					 __builtin_unreachable();
				 }
		}

		entry->metadata.memory_access.before = value_64bit;

		entry->metadata.memory_access.after = value_64bit;
		if(entry->metadata.memory_access.is_write) {
			/* Merge written bytes with unchanged upper bytes so 'after' reflects actual memory state. */
			/* TODO: STP writes target_register2 to the adjacent word; only target_register is handled here. */
			entry->metadata.memory_access.after =
				(value_64bit & ~write_mask) |
				(entry->cpu.gpr[target_register] & write_mask);
		}
	}

	return entry;
}

void* log_instr_with_speculation_nesting(struct simulation_state* sim_state, uint64_t speculation_nesting) {
	instr_trace_entry_t* entry = log_sim_state(sim_state);
	if(NULL != entry) {
		entry->metadata.speculation_nesting = speculation_nesting;
	}
	return NULL;
}

void* log_instr_hook(struct simulation_state* sim_state) {
	log_sim_state(sim_state);
	return NULL;
}

void init_trace_log(size_t test_size) {
	current_log_index = 0;
	max_log_index = (test_size / 4) * 128;  // In aarch64, each instruction is 4 bytes. The 128 is done for taking into considiration speculative contracts which may execute an instruction more then once for different flows
	trace_log = alloc_contract_trace(max_log_index);
}

static bool safe_file_write(FILE* f, const void* buff, size_t size) {
	size_t result = 0;
	while(true) {
		result = fwrite(buff, size, 1, f);
		if(1 == result) {
			fflush(f);
		       	return true;
		}
		if(ferror(f) && EINTR == errno) {
			clearerr(f);
			continue;
		}
		return false;
	}
}

static int transmit_payload_to_file(FILE* f, const void* payload, size_t payload_length) {
	if(NULL == f || NULL == payload) return -1;

	struct header hdr = { 0 };
	hdr.length = payload_length;
	hdr.type = 2;

	if(!safe_file_write(f, &hdr, sizeof(hdr))) {
		return -1;
	}

	if(0 < hdr.length) {
		if(!safe_file_write(f, payload, hdr.length)) {
			return -2;
		}
	}

	return hdr.length;
}

void destroy_trace_log() {
	if(NULL == trace_log) return;
	transmit_payload_to_file(stdout, trace_log, current_log_index * sizeof(instr_trace_entry_t) + sizeof(contract_trace_t));
	free_contract_trace(trace_log);
	trace_log = NULL;
	current_log_index = 0;
	max_log_index  = 0;
}
