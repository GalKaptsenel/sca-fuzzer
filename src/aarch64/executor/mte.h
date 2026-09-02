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
// GCR_EL1 management for the test run. The CE and the seal model ADDG/IRG tag arithmetic with no
// exclusion; the kernel (KASAN_HW_TAGS) leaves tags reserved in GCR_EL1.Exclude, which makes a sealed
// ADDG retag land on a different tag on HW -> architectural tag fault. mte_gcr_read() captures the
// pre-test-case value; mte_gcr_clear_exclude() zeroes Exclude right before a test executes;
// mte_gcr_restore() puts the captured pre-test-case value back (between inputs and at the TC boundary).
uint64_t mte_gcr_read(void);
uint64_t mte_gcr_clear_exclude(void);
void mte_gcr_restore(uint64_t saved_gcr);
// Read + clear TFSR_EL1 (async tag-fault status): non-zero => a tag-check fault happened in the window.
uint64_t mte_read_clear_tfsr(void);
// Force TCF=SYNC for the run (returns prior SCTLR_EL1); mte_restore_sctlr puts it back.
uint64_t mte_force_sync(void);
void mte_restore_sctlr(uint64_t saved);
uint8_t enable_TCMA1_bit(void);
uint8_t disable_TCMA1_bit(void);
uint8_t enable_TCO_bit(void);
uint8_t disable_TCO_bit(void);

#endif // ARM64_MTE_H
