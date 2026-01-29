#include "simulation_output.h"

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

static inline uint64_t read_reg(const cpu_state_t *s, unsigned r) {
	if (r == 31) return s->sp;
	return s->gpr[r];
}

static inline int is_memory_access(uint32_t inst) {
	return ((inst >> 25) & 0x7) == 0b100;
}
static inline int is_regular_load_store(uint32_t inst) {
	return ((inst >> 27) & 0x7) == 0b111;
}
static inline int is_pair_load_store(uint32_t inst) {
	return ((inst >> 27) & 0x7) == 0b101;
}

static inline int is_literal_pc_relative(uint32_t inst) {
	return ((inst >> 27) & 0x7) == 0b011;
}

static inline uint64_t access_size(uint32_t inst) {
	uint64_t size = ((inst >> 30) & 0x3);
	return 1u << size;
}
static inline uint64_t pair_element_access_size(uint32_t inst) {
	uint64_t size = ((inst >> 30) & 0x3);
	if(0 == size) return 4;
	if(2 == size) return 8;
	__builtin_unreachable();
}
static inline uint64_t literal_pc_relative_access_size(uint32_t inst) {
	uint64_t size = ((inst >> 30) & 0x3);
	if(0 == size) return 4;
	if(1 == size) return 8;
	if(2 == size) return 8;
	if(3 == size) return 0; // PRFM
	__builtin_unreachable();
}
static inline int is_load(uint32_t inst) {
	return (inst >> 22) & 0x1;
}

static inline int is_store(uint32_t inst) {
	return !is_load(inst);
}

static inline int is_pre_index(uint32_t inst) {
	return ((inst >> 24) & 0x3) == 0x0 && ((inst >> 10) & 0x3) == 0x3;
}

static inline int is_post_index(uint32_t inst) {
	return ((inst >> 24) & 0x3) == 0x0 && ((inst >> 10) & 0x3) == 0x2;
}

static inline int is_reg_offset(uint32_t inst) {
	return ((inst >> 24) & 0x3) == 0x0 && ((inst >> 10) & 0x3) == 0x1;
}

static inline int is_64bit_rm(uint32_t inst) {
	if(!is_reg_offset(inst)) return 0;
	return ((inst >> 13) & 0x1);
}

static inline int is_32bit_rm(uint32_t inst) {
	if(!is_reg_offset(inst)) return 0;
	return !is_64bit_rm(inst);
}

static inline int is_unsigned_offset(uint32_t inst) {
	return ((inst >> 24) & 0x3) == 0x1;
}

static inline int is_pair_pre_index(uint32_t inst) {
	return ((inst >> 23) & 0x7) == 0b011;
}
static inline int is_pair_post_index(uint32_t inst) {
	return ((inst >> 23) & 0x7) == 0b001;
}
static inline int is_pair_signed_offset(uint32_t inst) {
	return ((inst >> 23) & 0x7) == 0b010;
}

static inline int is_immidiate_offset(uint32_t inst) {
	return is_pre_index(inst) || is_post_index(inst) || is_unsigned_offset(inst);
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
static uint32_t get_rm(uint32_t inst) {
	return (inst >> 16) & 0x1F;
}
static uint32_t get_rn(uint32_t inst) {
	return (inst >> 5) & 0x1F;
}
static uint32_t get_rt(uint32_t inst) {
	return (inst) & 0x1F;
}
static uint32_t get_rt2(uint32_t inst) {
	return (inst >> 10) & 0x1F;
}

typedef enum {
	EXT_UXTB, EXT_UXTH, EXT_UXTW, EXT_UXTX,
	EXT_SXTB, EXT_SXTH, EXT_SXTW, EXT_SXTX
} extend_t;


static extend_t decode_extend(uint32_t inst) {
	switch(((inst >> 13) & 0x7)) {
		case 0b000: return EXT_UXTB;
		case 0b001: return EXT_UXTH;
		case 0b010: return EXT_UXTW;
		case 0b011: return EXT_UXTX;
		case 0b100: return EXT_SXTB;
		case 0b101: return EXT_SXTH;
		case 0b110: return EXT_SXTW;
		case 0b111: return EXT_SXTX;
	}
	__builtin_unreachable();
}

static size_t decode_amount_log2(uint32_t inst) {
	if(((inst >> 12) & 0x1)) {
		size_t size = ((inst >> 30) & 0x3);
		return size;
	}
	return 0;
}

static uint64_t is_signextend(uint32_t inst) {
	switch (decode_extend(inst)) {
		case EXT_UXTB: 
		case EXT_UXTH: 
		case EXT_UXTW: 
		case EXT_UXTX: return 0;
		case EXT_SXTB: 
		case EXT_SXTH: 
		case EXT_SXTW: 
		case EXT_SXTX: return 1;

	}
	__builtin_unreachable();
}
// TODO: Verify corrctness!
static int64_t signextend(size_t orig_len, size_t dest_len, int64_t value) {
	uint8_t sign = (1ull << (orig_len - 1)) & value;
	uint64_t extension = 0;
	if(sign) {
		extension = ((1ull << (dest_len - orig_len)) - 1) << orig_len;
	}
	return extension | value;
}

// TODO: Verify corrctness!
static uint64_t zeroextend(size_t orig_len, size_t dest_len, uint64_t value) {
	uint64_t extension = ~(((1ull << (dest_len - orig_len)) - 1) << orig_len);
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

			size_t orig_len = (src_len < 64 - shift_amount) ? src_len : 64 - shift_amount;
			uint64_t offset_val= (rm_val & ((1ull << orig_len) - 1)) << shift_amount;

			if(is_signed) {
				int64_t extended_val = signextend(orig_len, 64, offset_val);
				effective_address_computed = (uint64_t)((int64_t)base + extended_val);
			} else {
				uint64_t extended_val = zeroextend(orig_len, 64, offset_val);
				effective_address_computed = base + extended_val;
			}

		} else if(is_immidiate_offset(inst)){
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
		if (is_store_inst) *is_store_inst = false;
		if(is_load_inst) *is_load_inst = 1; // Only loads exist PC-relative
		if(data_size) *data_size = literal_pc_relative_access_size(inst);
		int64_t offset = decode_imm19(inst) * 4;
		effective_address_computed = (uint64_t)((int64_t)state->pc + offset);
	}

	if(effective_address) *effective_address = effective_address_computed;

	return 1;
}

void* log_instr_hook(struct simulation_state* sim_state) {
	if(NULL == sim_state) return NULL;
	size_t current_index = current_log_index;
	if(max_log_index <= current_index) {
		fprintf(stderr, "Unable to log more instructions, reached maximum log size!\n");
		return NULL;
	}
	++current_log_index;
	trace_log->entries[current_index].meta.instr_index = current_index;
	trace_log->entries[current_index].cpu.gpr[0] = sim_state->cpu_state.gprs.x0;
	trace_log->entries[current_index].cpu.gpr[1] = sim_state->cpu_state.gprs.x1;
	trace_log->entries[current_index].cpu.gpr[2] = sim_state->cpu_state.gprs.x2;
	trace_log->entries[current_index].cpu.gpr[3] = sim_state->cpu_state.gprs.x3;
	trace_log->entries[current_index].cpu.gpr[4] = sim_state->cpu_state.gprs.x4;
	trace_log->entries[current_index].cpu.gpr[5] = sim_state->cpu_state.gprs.x5;
	trace_log->entries[current_index].cpu.gpr[6] = sim_state->cpu_state.gprs.x6;
	trace_log->entries[current_index].cpu.gpr[7] = sim_state->cpu_state.gprs.x7;
	trace_log->entries[current_index].cpu.gpr[8] = sim_state->cpu_state.gprs.x8;
	trace_log->entries[current_index].cpu.gpr[9] = sim_state->cpu_state.gprs.x9;
	trace_log->entries[current_index].cpu.gpr[10] = sim_state->cpu_state.gprs.x10;
	trace_log->entries[current_index].cpu.gpr[11] = sim_state->cpu_state.gprs.x11;
	trace_log->entries[current_index].cpu.gpr[12] = sim_state->cpu_state.gprs.x12;
	trace_log->entries[current_index].cpu.gpr[13] = sim_state->cpu_state.gprs.x13;
	trace_log->entries[current_index].cpu.gpr[14] = sim_state->cpu_state.gprs.x14;
	trace_log->entries[current_index].cpu.gpr[15] = sim_state->cpu_state.gprs.x15;
	trace_log->entries[current_index].cpu.gpr[16] = sim_state->cpu_state.gprs.x16;
	trace_log->entries[current_index].cpu.gpr[17] = sim_state->cpu_state.gprs.x17;
	trace_log->entries[current_index].cpu.gpr[18] = sim_state->cpu_state.gprs.x18;
	trace_log->entries[current_index].cpu.gpr[19] = sim_state->cpu_state.gprs.x19;
	trace_log->entries[current_index].cpu.gpr[20] = sim_state->cpu_state.gprs.x20;
	trace_log->entries[current_index].cpu.gpr[21] = sim_state->cpu_state.gprs.x21;
	trace_log->entries[current_index].cpu.gpr[22] = sim_state->cpu_state.gprs.x22;
	trace_log->entries[current_index].cpu.gpr[23] = sim_state->cpu_state.gprs.x23;
	trace_log->entries[current_index].cpu.gpr[24] = sim_state->cpu_state.gprs.x24;
	trace_log->entries[current_index].cpu.gpr[25] = sim_state->cpu_state.gprs.x25;
	trace_log->entries[current_index].cpu.gpr[26] = sim_state->cpu_state.gprs.x26;
	trace_log->entries[current_index].cpu.gpr[27] = sim_state->cpu_state.gprs.x27;
	trace_log->entries[current_index].cpu.gpr[28] = sim_state->cpu_state.gprs.x28;
	trace_log->entries[current_index].cpu.gpr[29] = sim_state->cpu_state.gprs.x29;
	trace_log->entries[current_index].cpu.gpr[30] = sim_state->cpu_state.lr;
	trace_log->entries[current_index].cpu.sp = sim_state->cpu_state.sp;
	trace_log->entries[current_index].cpu.pc = sim_state->cpu_state.pc;
	trace_log->entries[current_index].cpu.nzcv = sim_state->cpu_state.nzcv;
	trace_log->entries[current_index].cpu.encoding = *(uint32_t*)sim_state->cpu_state.pc;
	trace_log->entries[current_index].cpu.extra_data_size = 0;
	trace_log->entries[current_index].cpu.extra_data = NULL;

	uint32_t target_register = (uint32_t)-1;
	uint32_t target_register2 = (uint32_t)-1;
	int is_memory_access = parse_memory_access_instruction(trace_log->entries[current_index].cpu.encoding, &trace_log->entries[current_index].cpu,
			&trace_log->entries[current_index].cpu.mem_access.effective_address,
			&target_register,
			NULL,
			NULL,
			&target_register2,
			NULL,
			(bool*)&trace_log->entries[current_index].cpu.mem_access.is_write,
			&trace_log->entries[current_index].cpu.mem_access.element_size
	);
	if(is_memory_access) {
		assert((uintptr_t)-1 != trace_log->entries[current_index].cpu.mem_access.effective_address && "Effective address should have been set by parse_memory_access_instruction function");

		uint64_t value_64bit = *(uint64_t*)trace_log->entries[current_index].cpu.mem_access.effective_address;
		uint64_t mask = 0;
		switch(trace_log->entries[current_index].cpu.mem_access.element_size) {
			case 1: mask = 0xFF; break;
			case 2: mask = 0xFFFF; break;
			case 4: mask = 0xFFFFFFFF; break;
			case 8: mask = 0xFFFFFFFFFFFFFFFF; break;
			default: __builtin_unreachable();
		}
		mask = 0xFFFFFFFFFFFFFFFF; // For now at least, force to log the entire 64 bits at this address

		trace_log->entries[current_index].cpu.mem_access.before = value_64bit & mask;
		if(trace_log->entries[current_index].cpu.mem_access.is_write) {
			trace_log->entries[current_index].cpu.mem_access.after = trace_log->entries[current_index].cpu.gpr[target_register] & mask; // TODO: Notice that is for example this was a pair store instruction (STP), we log only the first operand written!
																		    // We assume, wrongly but okay for now, that only the target register is written, but also target_register2 is written
		}
	}
	return NULL;

}

void init_trace_log(size_t test_size) {
	current_log_index = 0;
	max_log_index = test_size / 4;  // In aarch64, each instruction is 4 bytes
	trace_log = alloc_contract_trace(max_log_index);
}

void destroy_trace_log() {
	free_contract_trace(trace_log);
}
