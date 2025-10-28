#ifndef ARM64_MTE_H
#define ARM64_MTE_H

#include "main.h"

void mte_randomly_tag_region(const void* ptr, uint64_t length);
void stg(const void* ptr);
uint8_t enable_TCMA1_bit(void);
uint8_t disable_TCMA1_bit(void);
void enable_mte_tag_checking(void);
uint8_t enable_TCO_bit(void);
uint8_t disable_TCO_bit(void);

#endif // ARM64_MTE_H
