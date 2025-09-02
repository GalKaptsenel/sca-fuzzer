#include "main.h"
//
//inline void stg(const void* ptr) {
//#if defined(CONFIG_ARM64_MTE)
//	asm volatile ("stg %[address], [%[address]]" : : [address]"r"(ptr) : "memory");
//#endif
//}
//EXPORT_SYMBOL(stg);
//
//void mte_randomly_tag_region(const void* ptr, uint64_t length) {
//#ifdef CONFIG_ARM64_MTE
//	uint64_t loc = 0;
//
//	for(; loc < length; loc += MTE_GRANULE_SIZE) {
//		uintptr_t current_ptr = (uintptr_t)ptr + loc;
//		module_err("tagging address %p", (void*)current_ptr);
//	        uint8_t tag = 6;//mte_get_random_tag();
//		const void* tagged_ptr = __tag_set((void*)current_ptr, tag);
//		stg(tagged_ptr);
//	}
//#endif
//}
//EXPORT_SYMBOL(mte_randomly_tag_region);
//
//// Defines set_TCMA1_bit and set_TCO_bit functions automatically
//DEFINE_FULL_MSR_BIT_ACCESSORS(TCO, TCO, 25)
//DEFINE_FULL_MSR_BIT_ACCESSORS(TCR_EL1, TCMA1, 58)
//DEFINE_READ_MSR(CLIDR_EL1);
//
//uint8_t enable_TCMA1_bit(void) {
//#ifdef CONFIG_ARM64_MTE
//	return set_TCMA1_bit(1);
//#endif
//}
//EXPORT_SYMBOL(enable_TCMA1_bit);
//
//uint8_t disable_TCMA1_bit(void) {
//#ifdef CONFIG_ARM64_MTE
//	return set_TCMA1_bit(0);
//#endif
//}
//EXPORT_SYMBOL(disable_TCMA1_bit);
//
//uint8_t enable_TCO_bit(void) {
//#ifdef CONFIG_ARM64_MTE
//	return set_TCO_bit(1);
//#endif
//}
//EXPORT_SYMBOL(enable_TCO_bit);
//
//uint8_t disable_TCO_bit(void) {
//#ifdef CONFIG_ARM64_MTE
//	return set_TCO_bit(0);
//#endif
//}
//EXPORT_SYMBOL(disable_TCO_bit);
//
//static uint64_t g_value = 0;
//static void read_CLIDR_EL1_void(void* arg) {
//#ifdef CONFIG_ARM64_MTE
//	g_value = read_CLIDR_EL1();
//#endif
//}
//
//void enable_mte_tag_checking(void) {
//#ifdef CONFIG_ARM64_MTE
//	disable_TCO_bit();
//	unsigned long sctlr = read_sysreg(sctlr_el1);
//
//	if(!(sctlr & SCTLR_EL1_TCF_SYNC)) {
//		module_err("Resetting MTE to SYNC mode");
//		sysreg_clear_set(sctlr_el1, SCTLR_EL1_TCF_MASK, SCTLR_EL1_TCF_SYNC);
//		isb();
//	}
//
//	uint64_t val = read_CLIDR_EL1();
//
//	for(int i = 0; i < 10; ++i) {
//
//		execute_on_pinned_cpu(i, read_CLIDR_EL1_void, NULL);
//		module_err("CLIDR_EL1 on cpu %d: 0x%llx\n",i , g_value); 
//
//	}
//	module_err("CLIDR_EL1: 0x%llx\n", val); 
//#endif
//}
//EXPORT_SYMBOL(enable_mte_tag_checking);

