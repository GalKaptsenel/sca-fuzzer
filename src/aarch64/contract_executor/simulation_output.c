#include "simulation.h"
#include "simulation_output.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static contract_trace_t* trace_log = NULL;
static size_t current_log_index = 0;
static size_t max_log_index = 0;

static inline contract_trace_t* alloc_contract_trace(size_t num_entries) {
	fprintf(stderr, "[C] alloc_contract_trace: start!\n");
	size_t total_alloc_size = sizeof(contract_trace_t) + num_entries * sizeof(instr_trace_entry_t);
	contract_trace_t* trace = (contract_trace_t*)malloc(total_alloc_size);
	fprintf(stderr, "[C] alloc_contract_trace: check if successfull malloc!\n");
	if (NULL == trace) return NULL;
	fprintf(stderr, "[C] alloc_contract_trace: \t True!\n");
	memset(trace, 0, total_alloc_size);
	fprintf(stderr, "[C] alloc_contract_trace: end!\n");
	return trace;
}

static inline void free_contract_trace(contract_trace_t* trace) {
	if (NULL == trace) return;
	free(trace);
}

static inline uint64_t read_reg(const cpu_state_t *s, unsigned r) {
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
	__builtin_unreachable();
}

// TODO: Verify corrctness!
static int64_t signextend(size_t orig_len, size_t dest_len, int64_t value) {
	fprintf(stderr, "signextend(%ld, %ld, %lx): before extension: %lx\n", orig_len, dest_len, value, value);
	uint8_t sign = (1ull << (orig_len - 1)) & value;
	uint64_t extension = 0;
	if(sign) {
		extension = ((1ull << (dest_len - orig_len)) - 1) << orig_len;
	}

	fprintf(stderr, "signextend: after extension with %lx: %lx\n", extension, extension | value);
	return extension | value;
}

void* kaddr2uaddr(void* kaddr) {
	void* uaddr = kaddr;
	if(CONFIG_FLAG_REQ_MEM_BASE_VIRT | simulation.sim_input.hdr.config.flags) {
		uaddr = (char*)simulation.simulation_memory + ((uintptr_t)kaddr - simulation.sim_input.hdr.config.requested_mem_base_virt);
	}
	fprintf(stderr, "translation from kaddr to uaddr: %p --> %p\n", kaddr, uaddr);
	return uaddr;
}

void* uaddr2kaddr(void* uaddr) {
	void* kaddr = uaddr;
	if(CONFIG_FLAG_REQ_MEM_BASE_VIRT | simulation.sim_input.hdr.config.flags) {
		kaddr = (char*)simulation.sim_input.hdr.config.requested_mem_base_virt + ((uintptr_t)uaddr - (uintptr_t)simulation.simulation_memory);
	}
	fprintf(stderr, "translation from uaddr to kaddr: %p --> %p\n", uaddr, kaddr);
	return kaddr;
}


// TODO: Verify corrctness!
static uint64_t zeroextend(size_t orig_len, size_t dest_len, uint64_t value) {
	fprintf(stderr, "zeroextension(%ld, %ld, %lx): before extension: %lx\n", orig_len, dest_len, value, value);
	uint64_t extension = ~(((1ull << (dest_len - orig_len)) - 1) << orig_len);
	fprintf(stderr, "zeroextension: after extension with %lx: %lx\n", extension, extension & value);
	return extension & value;
}

static int parse_memory_access_instruction(
	uint32_t inst,
	const cpu_state_t *state,
	uintptr_t* effective_address,
	uint32_t* target_register,
	uint32_t* base_register,
	uint32_t* index_register,
	uint32_t* rt2_register,
	bool* is_load_inst,
	bool* is_store_inst,
	uint64_t* data_size) {

	fprintf(stderr, "[C] parse_memory_access_instruction: start\n");
	fprintf(stderr, "[C] parse_memory_access_instruction: running with instruction %x\n", inst);
	if(effective_address) *effective_address = (uintptr_t)-1;
	if(target_register) *target_register = (uint32_t)-1;
	if(base_register) *base_register = (uint32_t)-1;
	if(index_register) *index_register = (uint32_t)-1;
	if(is_load_inst) *is_load_inst = false;
	if(is_store_inst) *is_store_inst = false;
	if(rt2_register) *rt2_register = (uint32_t)-1;
	if(data_size) *data_size = 0;

	fprintf(stderr, "[C] parse_memory_access_instruction: check if memory access\n");
	if(!is_memory_access(inst)) return 0;

	fprintf(stderr, "[C] parse_memory_access_instruction: is memory access\n");

	if(target_register) *target_register = get_rt(inst);

	uintptr_t effective_address_computed = (uintptr_t)-1;
	if (is_regular_load_store(inst)) {
		fprintf(stderr, "[C] parse_memory_access_instruction: regular load/store\n");
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
			fprintf(stderr, "[C] parse_memory_access_instruction: base(%d): %lx, reg_offset(%ld): %lx\n", Rn, base, Rm, rm_val);
			if(is_32bit_rm(inst)) {
				rm_val &= ((1ull << 32) - 1);
				fprintf(stderr, "[C] parse_memory_access_instruction: it is 32bit -> base(%d): %lx, reg_offset(%ld): %lx\n", Rn, base, Rm, rm_val);
			}

			size_t orig_len = (src_len < 64 - shift_amount) ? src_len : 64 - shift_amount;
			uint64_t mask_offset_val= -1;
			if(64 != orig_len) mask_offset_val = ((1ull << orig_len) - 1);
			uint64_t offset_val= (rm_val & mask_offset_val) << shift_amount;
			fprintf(stderr, "[C] parse_memory_access_instruction: rm_val: %lx, orig_len: %ld, shift_amount: %ld, offset_val: %lx\n", rm_val, orig_len, shift_amount, offset_val);

			if(is_signed) {
				fprintf(stderr, "[C] parse_memory_access_instruction: signed extension\n");
				int64_t extended_val = signextend(orig_len, 64, offset_val);
				effective_address_computed = (uint64_t)((int64_t)base + extended_val);
			} else {
				fprintf(stderr, "[C] parse_memory_access_instruction: zero extension\n");
				uint64_t extended_val = zeroextend(orig_len, 64, offset_val);
				effective_address_computed = base + extended_val;
			}

		} else if(is_immidiate_offset(inst)){
			fprintf(stderr, "[C] parse_memory_access_instruction: it is an imidiate offset!\n");
			if(is_unsigned_offset(inst)) {
				fprintf(stderr, "[C] parse_memory_access_instruction: unsigned offset\n");
				uint64_t offset = (uint64_t)get_offset(inst);
				fprintf(stderr, "[C] parse_memory_access_instruction: base(%d): %lx, offset: %lx\n", Rn, base, offset);
				effective_address_computed = base + offset;
			} else if (is_pre_index(inst)){
				fprintf(stderr, "[C] parse_memory_access_instruction: pre index\n");
				int64_t offset = get_offset(inst);
				fprintf(stderr, "[C] parse_memory_access_instruction: base(%d): %lx, offset: %lx\n", Rn, base, offset);
				effective_address_computed = (uint64_t)((int64_t)base + offset);
			} else {
				fprintf(stderr, "[C] parse_memory_access_instruction: post index\n");
				fprintf(stderr, "[C] parse_memory_access_instruction: base(%d): %lx\n", Rn, base);
				effective_address_computed = base;
			}
		} else {
			__builtin_unreachable();
		}
	} else if(is_pair_load_store(inst)) {
		fprintf(stderr, "[C] parse_memory_access_instruction: pair load/store\n");
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
		fprintf(stderr, "[C] parse_memory_access_instruction: literal pc relative load\n");
		if (is_store_inst) *is_store_inst = false;
		if(is_load_inst) *is_load_inst = 1; // Only loads exist PC-relative
		if(data_size) *data_size = literal_pc_relative_access_size(inst);
		int64_t offset = decode_imm19(inst) * 4;
		effective_address_computed = (uint64_t)((int64_t)state->pc + offset);
	}

	if(effective_address) *effective_address = effective_address_computed;

	fprintf(stderr, "[C] parse_memory_access_instruction: end\n");
	return 1;
}

void* log_instr_hook(struct simulation_state* sim_state) {
	fprintf(stderr, "[C] log_instr_hook: start!\n");
	if(NULL == sim_state) return NULL;
	fprintf(stderr, "[C] log_instr_hook: check if out of smulation!\n");
	if(out_of_simulation(&sim_state->cpu_state)) return NULL;
	fprintf(stderr, "[C] log_instr_hook: \t inside simualtion, log all!\n");

	if(NULL == trace_log) init_trace_log(simulation.sim_input.hdr.code_size);

	size_t current_index = current_log_index;
	if(max_log_index <= current_index) {
		fprintf(stderr, "Unable to log more instructions, reached maximum log size!\n");
		return NULL;
	}

	fprintf(stderr, "[C] log_instr_hook: trace_log: %p, start of dynamic array at: %p\n", trace_log, trace_log+1);
	instr_trace_entry_t* entry = (instr_trace_entry_t*)(trace_log + 1) + current_index;
	fprintf(stderr, "[C] log_instr_hook:\tentry %ld at: %p\n",current_index, entry);

	trace_log->entry_count = current_index; 
	++current_log_index;
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
	entry->cpu.encoding = *(uint32_t*)(sim_state->cpu_state.pc);
	entry->cpu.extra_data_size = 0;
	// no need to allocate space for extra data, because it is of size 0 for now

	fprintf(stderr, "[C] log_instr_hook:call parse memory access\n");
	uint32_t target_register = (uint32_t)-1;
	uint32_t target_register2 = (uint32_t)-1;
	int is_memory_access = parse_memory_access_instruction(entry->cpu.encoding, &entry->cpu,
			&entry->metadata.memory_access.effective_address,
			&target_register,
			NULL,
			NULL,
			&target_register2,
			NULL,
			(bool*)&entry->metadata.memory_access.is_write,
			&entry->metadata.memory_access.element_size
	);

	fprintf(stderr, "[C] log_instr_hook:parse memory access finished\n");
	fprintf(stderr, "[C] log_instr_hook:check if memory access\n");
	entry->metadata.has_memory_access = is_memory_access ? 1 : 0;

	if(is_memory_access) {
		fprintf(stderr, "[C] log_instr_hook:it is memory access\n");
		fprintf(stderr, "[C] log_instr_hook:parse memory access: target register %d, is_write: %ld, element_size: %ld, ea: %p\n", target_register, entry->metadata.memory_access.is_write, entry->metadata.memory_access.element_size, (void*)entry->metadata.memory_access.effective_address);
		fprintf(stderr, "[C] log_instr_hook:check assert\n");
		assert((uintptr_t)-1 != entry->metadata.memory_access.effective_address && "Effective address should have been set by parse_memory_access_instruction function");
		fprintf(stderr, "[C] log_instr_hook:passed assert\n");

		void* kaddr = (void*)entry->metadata.memory_access.effective_address;
		fprintf(stderr, "[C] log_instr_hook: load value of effective address %p\n", kaddr);
		void* uaddr = kaddr2uaddr(kaddr);
		uint64_t value_64bit = *(uint64_t*)uaddr;
		fprintf(stderr, "[C] log_instr_hook: loaded value of effective address [kernel: %p, user: %p]: %p \n", kaddr, uaddr, (void*)value_64bit);
		uint64_t mask = 0;
		switch(entry->metadata.memory_access.element_size) {
			case 1: mask = 0xFF; break;
			case 2: mask = 0xFFFF; break;
			case 4: mask = 0xFFFFFFFF; break;
			case 8: mask = 0xFFFFFFFFFFFFFFFF; break;
			default: {

				fprintf(stderr, "[C] log_instr_hook: got to an unreachable!!!\n");
					 __builtin_unreachable();
				 }
		}
		fprintf(stderr, "[C] log_instr_hook: got mask of: %lx\n", mask);
		mask = 0xFFFFFFFFFFFFFFFF; // For now at least, force to log the entire 64 bits at this address
		fprintf(stderr, "[C] log_instr_hook: modified to use mask of: %lx\n", mask);

		entry->metadata.memory_access.before = value_64bit & mask;

		entry->metadata.memory_access.after = entry->metadata.memory_access.before;
		if(entry->metadata.memory_access.is_write) {
			fprintf(stderr, "[C] log_instr_hook: write verification of register %d, it has value of %lx\n", target_register, entry->cpu.gpr[target_register]);
			entry->metadata.memory_access.after = entry->cpu.gpr[target_register] & mask; // TODO: Notice that is for example this was a pair store instruction (STP), we log only the first operand written!
																		    // We assume, wrongly but okay for now, that only the target register is written, but also target_register2 is written
		}
		fprintf(stderr, "[C] log_instr_hook: finished parseing memory access\n");
	}

	fprintf(stderr, "[C] log_instr_hook: end!\n");
	return NULL;
}

void init_trace_log(size_t test_size) {
	fprintf(stderr, "[C] init_trace_log: start!\n");
	current_log_index = 0;
	max_log_index = (test_size / 4) * 128;  // In aarch64, each instruction is 4 bytes. The 128 is done for taking into considiration speculative contracts which may execute an instruction more then once for different flows
	fprintf(stderr, "[C] init_trace_log: allocate contrace trace!\n");
	trace_log = alloc_contract_trace(max_log_index);
	fprintf(stderr, "[C] init_trace_log: end!\n");
}

void destroy_trace_log(struct shm_region* shm) {
	if(NULL == trace_log) return;
	fprintf(stderr, "[C] sending respose trace_log");
	ring_send(shm, &shm->resp, 2, (uint8_t*)trace_log, current_log_index * sizeof(instr_trace_entry_t) + sizeof(contract_trace_t));
	free_contract_trace(trace_log);
	trace_log = NULL;
	current_log_index = 0;
	max_log_index  = 0;
}
