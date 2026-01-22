#include "main.h"

uint32_t encode_bl(uintptr_t from, uintptr_t to) {
    int64_t diff = (int64_t)to - (int64_t)from;
    int64_t imm26 = diff >> 2;

    if (imm26 < -(1LL << 25) || imm26 >= (1LL << 25)) {
        return 0;
    }

    return 0x94000000 | (imm26 & 0x03FFFFFF);
}

