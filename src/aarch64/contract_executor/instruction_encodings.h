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


#endif // SIMULATION_INSTRUCTION_ENCODINGS_H
