#ifndef ARM64_MTE_H
#define ARM64_MTE_H

#include "main.h"

// The default allocation tag the sandbox is uniformly loaded with. TBD: generalize to randomized
// per-granule initial tags.
#define MTE_INITIAL_DEFAULT_TAG		(0xF)

// Per-CPU MTE control registers, saved before the module reprograms them and restored on unload.
struct mte_control_state {
	uint64_t sctlr_el1;
	uint64_t tcr_el1;
};

void* mte_canonical_ptr(const void* p);

void *mte_alloc_tagged_region(size_t size);
void mte_free_tagged_region(void *ptr, size_t size);
bool mte_region_is_tagged(const void *ptr, size_t size);
void mte_init_sandbox_tags(const void* base, uint64_t length, uint8_t tag);
void mte_apply_sandbox_tags(const void* base, const uint8_t* tags, uint64_t n_granules);
void mte_save_control(struct mte_control_state* state);
void mte_restore_control(const struct mte_control_state* state);
void mte_set_sync(void);
// Force GCR_EL1.Exclude=0 for the test run (returns the prior GCR_EL1 to restore). The CE and the seal
// model ADDG/IRG tag arithmetic with no exclusion; the kernel (KASAN_HW_TAGS) leaves tags reserved in
// Exclude, which makes a sealed ADDG retag land on a different tag on HW -> architectural tag fault.
uint64_t mte_gcr_clear_exclude(void);
void mte_gcr_restore(uint64_t saved_gcr);
uint8_t enable_TCMA1_bit(void);
uint8_t disable_TCMA1_bit(void);
uint8_t enable_TCO_bit(void);
uint8_t disable_TCO_bit(void);

#endif // ARM64_MTE_H
