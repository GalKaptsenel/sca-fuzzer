#include "main.h"

/* android14-5.15 GKI names the EL1 tag-check field SCTLR_ELx_TCF_* and exposes no EL1 ATA (bit 43)
 * macro; newer kernels use SCTLR_EL1_TCF_*. */
#ifndef SCTLR_EL1_TCF_MASK
#define SCTLR_EL1_TCF_MASK SCTLR_ELx_TCF_MASK
#endif
#ifndef SCTLR_EL1_TCF_SYNC
#define SCTLR_EL1_TCF_SYNC SCTLR_ELx_TCF_SYNC
#endif
#ifndef SCTLR_EL1_ATA
#define SCTLR_EL1_ATA (UL(1) << 43)
#endif

/* Allocate a physically-contiguous region from the linear map (Normal-Tagged on
 * CONFIG_ARM64_MTE), so STG tagging and tag checks apply to it. get_order returns a 2^order
 * block aligned to its own size, so a caller whose size >= its alignment need (e.g. the sandbox
 * vs L1D_SIZE) is correctly aligned. Generic (page allocation), so defined for MTE and non-MTE. */
void *mte_alloc_tagged_region(size_t size) {
	return (void *)__get_free_pages(GFP_KERNEL | __GFP_ZERO, get_order(size));
}

void mte_free_tagged_region(void *ptr, size_t size) {
	if (NULL != ptr) {
		free_pages((unsigned long)ptr, get_order(size));
	}
}

#if CONFIG_ARM64_MTE_HW	// Real MTE hardware implementation

static inline void stg(const void* ptr) {
	asm volatile("stg %[address], [%[address]]"
			:
			: [address]"r"(ptr)
			: "memory");
}

static inline void *tag_ptr(void *p, u8 tag) {
    // The allocation tag is bits [59:56]; mask to 4 bits so a wider value cannot bleed into the
    // rest of the top byte.
    return (void *)(((u64)p & 0x00FFFFFFFFFFFFFFULL) | ((u64)(tag & 0xF) << 56));
}


void mte_init_sandbox_tags(const void* base, uint64_t length, uint8_t tag) {
	uint64_t loc = 0;
	for (; loc < length; loc += MTE_GRANULE_SIZE) {
		uintptr_t current_ptr = (uintptr_t)base + loc;
		const void* tagged_ptr = tag_ptr((void*)current_ptr, tag);
		stg(tagged_ptr);
	}
}

void mte_apply_sandbox_tags(const void* base, const uint8_t* tags, uint64_t n_granules) {
	for (uint64_t i = 0; i < n_granules; ++i) {
		uintptr_t current_ptr = (uintptr_t)base + i * MTE_GRANULE_SIZE;
		const void* tagged_ptr = tag_ptr((void*)current_ptr, tags[i]);
		stg(tagged_ptr);
	}
}

// MTE system register bit accessors
DEFINE_FULL_MSR_BIT_ACCESSORS(TCO, TCO, 25)
DEFINE_FULL_MSR_BIT_ACCESSORS(TCR_EL1, TCMA1, 58)

uint8_t enable_TCMA1_bit(void) {
	return set_TCMA1_bit(1);
}

uint8_t disable_TCMA1_bit(void) {
	return set_TCMA1_bit(0);
}

uint8_t enable_TCO_bit(void) {
	return set_TCO_bit(1);
}

uint8_t disable_TCO_bit(void) {
	return set_TCO_bit(0);
}

void mte_set_sync(void) {
	// ATA (bit 43) enables allocation-tag access; TCF (bits [41:40]) = 0b01 = synchronous faults.
	sysreg_clear_set(sctlr_el1, SCTLR_EL1_TCF_MASK, SCTLR_EL1_ATA | SCTLR_EL1_TCF_SYNC);
	isb();
}

void mte_save_control(struct mte_control_state* state) {
	state->sctlr_el1 = read_sysreg(sctlr_el1);
	state->tcr_el1 = read_TCR_EL1();
}

void mte_restore_control(const struct mte_control_state* state) {
	write_sysreg(state->sctlr_el1, sctlr_el1);
	write_TCR_EL1(state->tcr_el1);
	isb();
}

bool mte_region_is_tagged(const void *ptr, size_t size) {
	return pte_region_attr_is((void *)ptr, size, MT_NORMAL_TAGGED);
}

#else	// Non-MTE hardware: all stubs

static inline void stg(const void* ptr)				{ (void)ptr; }

void mte_init_sandbox_tags(const void* base, uint64_t length, uint8_t tag) { (void)base; (void)length; (void)tag; }

void mte_apply_sandbox_tags(const void* base, const uint8_t* tags, uint64_t n_granules) { (void)base; (void)tags; (void)n_granules; }

uint8_t enable_TCMA1_bit(void)					{ return 0; }

uint8_t disable_TCMA1_bit(void)					{ return 0; }

uint8_t enable_TCO_bit(void)					{ return 0; }

uint8_t disable_TCO_bit(void)					{ return 0; }

bool mte_region_is_tagged(const void *ptr, size_t size)		{ (void)ptr; (void)size; return true; }

#endif

