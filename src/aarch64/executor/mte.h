#ifndef ARM64_MTE_H
#define ARM64_MTE_H

#include "main.h"

// The default allocation tag the sandbox is uniformly loaded with. TBD: generalize to randomized
// per-granule initial tags.
#define MTE_INITIAL_TAG		(6)

void *mte_alloc_tagged_region(size_t size);
void mte_free_tagged_region(void *ptr, size_t size);
bool mte_region_is_tagged(const void *ptr, size_t size);
void mte_randomly_tag_region(const void* ptr, uint64_t length);
void mte_init_sandbox_tags(const void* base, uint64_t length, uint8_t tag);
void mte_apply_sandbox_tags(const void* base, const uint8_t* tags, uint64_t n_granules);
uint8_t enable_TCMA1_bit(void);
uint8_t disable_TCMA1_bit(void);
void enable_mte_tag_checking(void);
uint8_t enable_TCO_bit(void);
uint8_t disable_TCO_bit(void);

#endif // ARM64_MTE_H
