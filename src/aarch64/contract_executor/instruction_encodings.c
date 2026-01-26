#include "instruction_encodings.h"

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
