#include "main.h"

inline void stg(const void* ptr) {
	asm volatile ("stg %[address], [%[address]]" : : [address]"r"(ptr) : "memory");
}

void mte_randomly_tag_region(const void* ptr, uint64_t length) {
	uint64_t loc = 0;

	for(; loc < length; loc += MTE_GRANULE_SIZE) {
		size_t current_ptr = (size_t)ptr + loc;
		module_err("tagging address %p", current_ptr);
	        uint8_t tag = 5;//mte_get_random_tag();
		const void* tagged_ptr = __tag_set((void*)current_ptr, tag);
		stg(tagged_ptr);
	}
}

static inline uint64_t read_tcr_el1(void) {
	uint64_t tcr_el1 = 0;
	asm volatile ("mrs %[tcr], TCR_EL1" : [tcr]"=r"(tcr_el1));
	return tcr_el1;
}

static inline void write_tcr_el1(uint64_t new_tcr_el1) {
	asm volatile ("msr TCR_EL1, %[tcr]" :: [tcr]"r"(new_tcr_el1));
	asm volatile ("isb" ::: "memory");
}

static uint8_t set_TCMA1_bit(uint8_t value) {
	uint64_t tcr_el1 = read_tcr_el1();
	uint8_t bit = (tcr_el1 & (1UL << TCMA1_BIT_SHIFT)) != 0;

	if(value) {
		tcr_el1 |= (1UL << TCMA1_BIT_SHIFT);
	} else {
		tcr_el1 &= ~(1UL << TCMA1_BIT_SHIFT);
	}

	write_tcr_el1(tcr_el1);
	return bit;

}

uint8_t enable_TCMA1_bit(void) {
	return set_TCMA1_bit(1);
}

uint8_t disable_TCMA1_bit(void) {
	return set_TCMA1_bit(0);
}

