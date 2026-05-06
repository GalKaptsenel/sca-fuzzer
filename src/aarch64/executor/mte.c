#include "main.h"
#if CONFIG_ARM64_MTE_HW	// Real MTE hardware implementation

inline void stg(const void* ptr) {
	asm volatile("stg %[address], [%[address]]"
			:
			: [address]"r"(ptr)
			: "memory");
}
EXPORT_SYMBOL(stg);

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
EXPORT_SYMBOL(mte_randomly_tag_region);

// MTE system register bit accessors
DEFINE_FULL_MSR_BIT_ACCESSORS(TCO, TCO, 25)
DEFINE_FULL_MSR_BIT_ACCESSORS(TCR_EL1, TCMA1, 58)
DEFINE_READ_MSR(CLIDR_EL1);

uint8_t enable_TCMA1_bit(void) {
	return set_TCMA1_bit(1);
}
EXPORT_SYMBOL(enable_TCMA1_bit);

uint8_t disable_TCMA1_bit(void) {
	return set_TCMA1_bit(0);
}
EXPORT_SYMBOL(disable_TCMA1_bit);

uint8_t enable_TCO_bit(void) {
	return set_TCO_bit(1);
}
EXPORT_SYMBOL(enable_TCO_bit);

uint8_t disable_TCO_bit(void) {
	return set_TCO_bit(0);
}
EXPORT_SYMBOL(disable_TCO_bit);

static uint64_t g_value = 0;

static void read_CLIDR_EL1_void(void* arg) {
	(void)arg;
	g_value = read_CLIDR_EL1();
}

void dump_mte_status(void) {
    u64 sctlr, tcr;

    asm volatile("mrs %0, SCTLR_EL1" : "=r"(sctlr));
    asm volatile("mrs %0, TCR_EL1"   : "=r"(tcr));

    u64 ata = (sctlr >> 18) & 1;
    u64 tcf = (sctlr >> 19) & 0x3;

    pr_info("MTE status:\n");
    pr_info("  ATA = %llu\n", ata);
    pr_info("  TCF = %llu\n", tcf);
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
//	module_err("On cpu %d", (int)a);
	mte_set_sync();
}

void enable_mte_tag_checking(void) {
//	dump_mte_status();
	disable_TCO_bit();
	unsigned long sctlr = read_sysreg(sctlr_el1);

	if (!(sctlr & SCTLR_EL1_TCF_SYNC)) {
//		module_err("Resetting MTE to SYNC mode");
		sysreg_clear_set(sctlr_el1, SCTLR_EL1_TCF_MASK, SCTLR_EL1_TCF_SYNC);
		isb();
	}

	uint64_t val = read_CLIDR_EL1();
	for (int i = 0; i < nr_cpu_ids; ++i) {
		execute_on_pinned_cpu(i, read_CLIDR_EL1_void, NULL);
		execute_on_pinned_cpu(i, mte_set_sync_callback, (char*)i);
//		module_err("CLIDR_EL1 on cpu %d: 0x%llx\n", i, g_value);
	}
//	module_err("CLIDR_EL1: 0x%llx\n", val);

//	dump_mte_status();
}
EXPORT_SYMBOL(enable_mte_tag_checking);

#else	// Non-MTE hardware: all stubs

inline void stg(const void* ptr)				{ (void)ptr; }
EXPORT_SYMBOL(stg);

void mte_randomly_tag_region(const void* ptr, uint64_t length)	{ (void)ptr; (void)length; }
EXPORT_SYMBOL(mte_randomly_tag_region);

uint8_t enable_TCMA1_bit(void)					{ return 0; }
EXPORT_SYMBOL(enable_TCMA1_bit);

uint8_t disable_TCMA1_bit(void)					{ return 0; }
EXPORT_SYMBOL(disable_TCMA1_bit);

uint8_t enable_TCO_bit(void)					{ return 0; }
EXPORT_SYMBOL(enable_TCO_bit);

uint8_t disable_TCO_bit(void)					{ return 0; }
EXPORT_SYMBOL(disable_TCO_bit);

void enable_mte_tag_checking(void)				{ }
EXPORT_SYMBOL(enable_mte_tag_checking);

#endif

static inline unsigned long read_id_aa64pfr1_el1(void) {
	unsigned long val = 0;
	asm volatile("mrs %0, ID_AA64PFR1_EL1" : "=r"(val));
	return val;
}

int mte_ext(void) {
	unsigned long val = read_id_aa64pfr1_el1();
	unsigned long mte = (val >> 8) & 0xF;
	return mte;
}


