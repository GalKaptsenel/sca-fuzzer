#include "main.h"
#if CONFIG_ARM64_MTE_HW	// Real MTE hardware implementation

static inline void stg(const void* ptr) {
	asm volatile("stg %[address], [%[address]]"
			:
			: [address]"r"(ptr)
			: "memory");
}

static inline void *tag_ptr(void *p, u8 tag) {
    return (void *)(((u64)p & 0x00FFFFFFFFFFFFFFULL) | ((u64)tag << 56));
}


void mte_randomly_tag_region(const void* ptr, uint64_t length) {
	uint64_t loc = 0;

	for (; loc < length; loc += MTE_GRANULE_SIZE) {
		uintptr_t current_ptr = (uintptr_t)ptr + loc;
		uint8_t tag = 6; // mte_get_random_tag();
		const void* tagged_ptr = tag_ptr((void*)current_ptr, tag);
		stg(tagged_ptr);
	}
}

void mte_init_sandbox_tags(const void* base, uint64_t length, uint8_t tag) {
	uint64_t loc = 0;
	for (; loc < length; loc += MTE_GRANULE_SIZE) {
		uintptr_t current_ptr = (uintptr_t)base + loc;
		const void* tagged_ptr = tag_ptr((void*)current_ptr, tag);
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

static inline void mte_set_sync(void) {
    u64 sctlr;

    asm volatile("mrs %0, SCTLR_EL1" : "=r"(sctlr));

    sctlr |= (1ULL << 18);        // ATA = 1
    sctlr &= ~(3ULL << 19);       // clear TCF
    sctlr |= (1ULL << 19);        // TCF = 01 (sync)

    asm volatile("msr SCTLR_EL1, %0" :: "r"(sctlr));
    asm volatile("isb");
}
static inline void mte_set_sync_callback(void* a) {
	(void)a;
	mte_set_sync();
}

void enable_mte_tag_checking(void) {
	disable_TCO_bit();
	unsigned long sctlr = read_sysreg(sctlr_el1);

	if (!(sctlr & SCTLR_EL1_TCF_SYNC)) {
		sysreg_clear_set(sctlr_el1, SCTLR_EL1_TCF_MASK, SCTLR_EL1_TCF_SYNC);
		isb();
	}

	for (int i = 0; i < nr_cpu_ids; ++i) {
		execute_on_pinned_cpu(i, mte_set_sync_callback, NULL);
	}
}

#else	// Non-MTE hardware: all stubs

static inline void stg(const void* ptr)				{ (void)ptr; }

void mte_randomly_tag_region(const void* ptr, uint64_t length)	{ (void)ptr; (void)length; }

void mte_init_sandbox_tags(const void* base, uint64_t length, uint8_t tag) { (void)base; (void)length; (void)tag; }

uint8_t enable_TCMA1_bit(void)					{ return 0; }

uint8_t disable_TCMA1_bit(void)					{ return 0; }

uint8_t enable_TCO_bit(void)					{ return 0; }

uint8_t disable_TCO_bit(void)					{ return 0; }

void enable_mte_tag_checking(void)				{ }

#endif

