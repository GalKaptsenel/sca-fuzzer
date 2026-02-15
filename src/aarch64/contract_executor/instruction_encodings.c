#include "instruction_encodings.h"

size_t emit_mov64(uintptr_t buff, int reg, uint64_t imm) {
	size_t ctr = 0;
	uint32_t* p = (uint32_t*)buff;

	// MOVZ Xd, imm16, LSL #0
	*p++ = 0xD2800000 | ((imm & 0xFFFF) << 5) | reg;
	++ctr;

	// MOVK Xd, imm16, LSL #16
	if ((imm >> 16) & 0xFFFF) {
		*p++ = 0xF2A00000 | (((imm >> 16) & 0xFFFF) << 5) | reg;
		++ctr;
	}

	// MOVK Xd, imm16, LSL #32
	if ((imm >> 32) & 0xFFFF) {
		*p++ = 0xF2C00000 | (((imm >> 32) & 0xFFFF) << 5) | reg;
		++ctr;
	}

	// MOVK Xd, imm16, LSL #48
	if ((imm >> 48) & 0xFFFF) {
		*p++ = 0xF2E00000 | (((imm >> 48) & 0xFFFF) << 5) | reg;
		++ctr;
	}

	return ctr;
}

uint32_t encode_b(uintptr_t from, uintptr_t to) {
    int64_t diff = (int64_t)to - (int64_t)from;
    int64_t imm26 = diff >> 2;

    if (imm26 < -(1LL << 25) || imm26 >= (1LL << 25)) {
        return 0;
    }

    return 0x14000000 | (imm26 & 0x03FFFFFF);
}

uint32_t encode_bl(uintptr_t from, uintptr_t to) {
    int64_t diff = (int64_t)to - (int64_t)from;
    int64_t imm26 = diff >> 2;

    if (imm26 < -(1LL << 25) || imm26 >= (1LL << 25)) {
        return 0;
    }

    return 0x94000000 | (imm26 & 0x03FFFFFF);
}

branch_type_t classify_branch(uint32_t instr) {
    if ((instr & 0xFF000010) == 0x54000000) return BRANCH_B_COND;
    if ((instr & 0xFC000000) == 0x14000000) return BRANCH_B;
    if ((instr & 0xFC000000) == 0x94000000) return BRANCH_BL;
    if ((instr & 0xFFFFFC1F) == 0xD63F0000) return BRANCH_BLR;
    if ((instr & 0x7F000000) == 0x34000000) return BRANCH_CBZ;
    if ((instr & 0x7F000000) == 0x35000000) return BRANCH_CBNZ;
    if ((instr & 0x7F000000) == 0x36000000) return BRANCH_TBZ;
    if ((instr & 0x7F000000) == 0x37000000) return BRANCH_TBNZ;
    return BRANCH_NONE;
}

uintptr_t evaluate_cond_target(uintptr_t pc, uint32_t insn) {
	branch_type_t btype = classify_branch(insn);

	if(BRANCH_B_COND == btype) {
		int32_t imm19 = (insn >> 5) & 0x7FFFF;
		if (imm19 & 0x40000) imm19 |= 0xFFF80000; // Sign Extend
		return pc + (imm19 << 2);
	}

	if (BRANCH_CBZ == btype || BRANCH_CBNZ == btype) {
		int32_t imm19 = (insn >> 5) & 0x7FFFF;
		if (imm19 & 0x40000) imm19 |= 0xFFF80000;  // Sign Extend
		return pc + (imm19 << 2);
	}

	if (BRANCH_TBZ == btype || BRANCH_TBNZ == btype) {
		int32_t imm14 = (insn >> 5) & 0x3FFF;
		if (imm14 & 0x2000) imm14 |= 0xFFFFC000; // sign extend 14-bit
		return pc + (imm14 << 2);
	}

	// Not a conditional branch we recognize
	return 0;
}

inline int is_memory_access(uint32_t inst) { return ((inst >> 25) & 0x7) == 0b100; }
inline int is_regular_load_store(uint32_t inst) { return ((inst >> 27) & 0x7) == 0b111; }
inline int is_pair_load_store(uint32_t inst) { return ((inst >> 27) & 0x7) == 0b101; }
inline int is_literal_pc_relative(uint32_t inst) { return ((inst >> 27) & 0x7) == 0b011; }
inline uint64_t access_size(uint32_t inst) {
	uint64_t size = ((inst >> 30) & 0x3);
	return 1u << size;
}
inline uint64_t pair_element_access_size(uint32_t inst) {
	uint64_t size = ((inst >> 30) & 0x3);
	if(0 == size) return 4;
	if(2 == size) return 8;
	__builtin_unreachable();
}
inline uint64_t literal_pc_relative_access_size(uint32_t inst) {
	uint64_t size = ((inst >> 30) & 0x3);
	if(0 == size) return 4;
	if(1 == size) return 8;
	if(2 == size) return 8;
	if(3 == size) return 0; // PRFM
	__builtin_unreachable();
}
inline int is_load(uint32_t inst) { return (inst >> 22) & 0x1; }
inline int is_store(uint32_t inst) { return !is_load(inst); }
inline int is_pre_index(uint32_t inst) { return ((inst >> 24) & 0x3) == 0x0 && ((inst >> 10) & 0x3) == 0x3; }
inline int is_post_index(uint32_t inst) { return ((inst >> 24) & 0x3) == 0x0 && ((inst >> 10) & 0x3) == 0x1; }
inline int is_reg_offset(uint32_t inst) { return ((inst >> 24) & 0x3) == 0x0 && ((inst >> 10) & 0x3) == 0x2; }
inline int is_64bit_rm(uint32_t inst) {
	if(!is_reg_offset(inst)) return 0;
	return ((inst >> 13) & 0x1);
}
inline int is_32bit_rm(uint32_t inst) {
	if(!is_reg_offset(inst)) return 0;
	return !is_64bit_rm(inst);
}

inline int is_unsigned_offset(uint32_t inst) { return ((inst >> 24) & 0x3) == 0x1; }
inline int is_pair_pre_index(uint32_t inst) { return ((inst >> 23) & 0x7) == 0b011; }
inline int is_pair_post_index(uint32_t inst) { return ((inst >> 23) & 0x7) == 0b001; }
inline int is_pair_signed_offset(uint32_t inst) { return ((inst >> 23) & 0x7) == 0b010; }
inline int is_immidiate_offset(uint32_t inst) { return is_pre_index(inst) || is_post_index(inst) || is_unsigned_offset(inst); }
inline uint32_t get_rm(uint32_t inst) { return (inst >> 16) & 0x1F; }
inline uint32_t get_rn(uint32_t inst) { return (inst >> 5) & 0x1F; }
inline uint32_t get_rt(uint32_t inst) { return (inst) & 0x1F; }
inline uint32_t get_rt2(uint32_t inst) { return (inst >> 10) & 0x1F; }
inline uint32_t set_rn(uint32_t inst, uint8_t rn) {
	return (inst & ~(0x1F << 5)) | rn << 5;
}

extend_t decode_extend(uint32_t inst) {
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

size_t decode_amount_log2(uint32_t inst) {
	if(((inst >> 12) & 0x1)) {
		size_t size = ((inst >> 30) & 0x3);
		return size;
	}
	return 0;
}

