#ifndef SIMULATION_INSTRUCTION_ENCODINGS_H
#define SIMULATION_INSTRUCTION_ENCODINGS_H

#include <stdint.h>

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


uint32_t encode_bl(uintptr_t from, uintptr_t to);

#endif // SIMULATION_INSTRUCTION_ENCODINGS_H
