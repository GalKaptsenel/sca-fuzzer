#ifndef SIMULATION_INSTRUCTION_ENCODINGS_H
#define SIMULATION_INSTRUCTION_ENCODINGS_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    BRANCH_NONE,
    BRANCH_B_COND,
    BRANCH_B,
    BRANCH_BL,
    BRANCH_BLR,
    BRANCH_CBZ,
    BRANCH_CBNZ,
    BRANCH_TBZ,
    BRANCH_TBNZ
} branch_type_t;

branch_type_t classify_branch(uint32_t instr);
uintptr_t evaluate_cond_target(uintptr_t pc, uint32_t insn);
uint32_t encode_bl(uintptr_t from, uintptr_t to);
uint32_t encode_b(uintptr_t from, uintptr_t to);
size_t emit_mov64(uintptr_t buff, int reg, uint64_t imm);
int is_memory_access(uint32_t inst);
int is_regular_load_store(uint32_t inst);
int is_pair_load_store(uint32_t inst);
int is_literal_pc_relative(uint32_t inst);
uint64_t access_size(uint32_t inst);
uint64_t pair_element_access_size(uint32_t inst);
uint64_t literal_pc_relative_access_size(uint32_t inst);
int is_load(uint32_t inst);
int is_store(uint32_t inst);
int is_pre_index(uint32_t inst);
int is_post_index(uint32_t inst);
int is_reg_offset(uint32_t inst);
int is_64bit_rm(uint32_t inst);
int is_32bit_rm(uint32_t inst);
int is_unsigned_offset(uint32_t inst);
int is_pair_pre_index(uint32_t inst);
int is_pair_post_index(uint32_t inst);
int is_pair_signed_offset(uint32_t inst);
int is_immidiate_offset(uint32_t inst);
uint32_t get_rm(uint32_t inst);
uint32_t get_rn(uint32_t inst);
uint32_t get_rt(uint32_t inst);
uint32_t get_rt2(uint32_t inst);
uint32_t set_rn(uint32_t inst, uint8_t rn);

typedef enum {
	EXT_UXTB, EXT_UXTH, EXT_UXTW, EXT_UXTX,
	EXT_SXTB, EXT_SXTH, EXT_SXTW, EXT_SXTX
} extend_t;
extend_t decode_extend(uint32_t inst);

size_t decode_amount_log2(uint32_t inst);

#endif // SIMULATION_INSTRUCTION_ENCODINGS_H
