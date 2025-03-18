#include "main.h"

static struct mm_struct *init_mm_ptr	=	NULL;

static void __nocfi load_init_mm(void) {
	load_global_symbol(kallsyms_lookup_name_fn, struct mm_struct *, init_mm_ptr, init_mm);
}

static void decode_entry(uint64_t entry, const char *level) {
	uint64_t present_bit = 0;
	uint64_t entry_type = 0;
	uint64_t accessed_flag = 0;
	uint64_t access_permissions = 0;
	uint64_t attr_index = 0;
	uint64_t sharability = 0;
	uint64_t dirty_bit_modifier = 0;
	uint64_t uxn_bit = 0;
	uint64_t pxn_bit = 0;
	uint64_t software_defined = 0;

	module_info("Decoding %s entry: 0x%016llx\n", level, entry);

	// Present Bit
	present_bit = entry & 0x1;
	if (present_bit) {
		module_info("%s: Present\n", level);
	}
	else {
		module_info("%s: Not Present\n", level);
		return;
	}

	// Page Type
	entry_type = (entry >> 1) & 0x1;
	if (entry_type) {
		module_info("%s: Table descriptor(levels 0, 1 and 2)/Table entry(levels 1 and 2)\n", level);
	}
	else {
		module_info("%s: Block entry(levels 1 and 2)\n", level);
	}

	// Accessed Flag
	accessed_flag = (entry >> 10) & 0x1;
	if (accessed_flag) {
		module_info("%s: Has already been used [AF=1]\n", level);
	}
	else {
		module_info("%s: Has not yet been used [AF=0]\n", level);
	}

	// Access Permissions
	access_permissions = (entry >> 6) & 0x3;
	switch(access_permissions) {

		case 0:
			module_info("%s: No access for EL0 and RW for EL1/2/3", level);
			break;
		case 1:
			module_info("%s: RW for EL0 and RW for EL1/2/3", level);
			break;
		case 2:
			module_info("%s: No access for EL0 and Read-Only for EL1/2/3", level);
			break;
		case 3:
			module_info("%s: Read-Only for EL0 and Read-Only for EL1/2/3", level);
			break;
	}

	// Attributes
	attr_index = (entry >> 2) & 0x7;
	module_info("%s: Memory Attributes Index: %llu\n", level, attr_index);

	// Shareability
	sharability = (entry >> 8) & 0x3;
	switch (sharability) {

		case 0:
			module_info("%s: Non-shareable\n", level);
			break;
		case 1:
			module_info("%s: Unpredictable shareability\n", level);
			break;
		case 2:
			module_info("%s: Outer shareable\n", level);
			break;
		case 3:
			module_info("%s: Inner shareable\n", level);
			break;
	}

	// Dirty Bit
	dirty_bit_modifier = (entry >> 51) & 0x1;
	if (dirty_bit_modifier) {
		module_info("%s: Dirty bit modifier\n", level);
	}

	// UXN Bit
	uxn_bit = (entry >> 54) & 0x1;
	if (uxn_bit) {
		module_info("%s: EL0 cannot execute code [UXN=1]\n", level);
	}
	else {
		module_info("%s: EL0 can execute code [UXN=0]\n", level);
	}

	// PXN Bit
	pxn_bit = (entry >> 53) & 0x1;
	if (pxn_bit) {
		module_info("%s: EL1/2/3 cannot execute code [PXN=1]\n", level);
	}
	else {
		module_info("%s: EL1/2/3 can execute code [PXN=0]\n", level);
	}

	// Software bits (optional, custom implementation-defined)
	software_defined = (entry >> 55);
	module_info("%s: Software-defined bits: 0x%llx\n", level, software_defined);

	module_info("| P\t|Type\t|AF\t|AP/Dirty\t|AttrIndx\t|SH\t|PXN\t|UXN\t| D Modifier\t|SW-defined\t");
	module_info("  %d\t  %d\t  %d\t  %d\t  %d\t\t %d\t  %d\t  %d\t  %d\t  %d\t",
	 present_bit, entry_type, accessed_flag, access_permissions,
	  attr_index,sharability, pxn_bit, uxn_bit, dirty_bit_modifier, software_defined);
}

// Functions to decode entries
static void decode_pgd(pgd_t pgd) {
    decode_entry(pgd_val(pgd), "PGD");
}

static void decode_p4d(p4d_t p4d) {
    decode_entry(p4d_val(p4d), "P4D");
}

static void decode_pud(pud_t pud) {
    decode_entry(pud_val(pud), "PUD");
}

static void decode_pmd(pmd_t pmd) {
    decode_entry(pmd_val(pmd), "PMD");
}

static void decode_pte(pte_t pte) {
    decode_entry(pte_val(pte), "PTE");
}

void page_walk_explorer(void *addr) {
	pgd_t *pgd = NULL;
	p4d_t *p4d = NULL;
	pud_t *pud = NULL;
	pmd_t *pmd = NULL;
	pte_t *pte = NULL;
	size_t vaddr = (size_t)addr;

	if(NULL == init_mm_ptr) {
		load_init_mm();
	}

	pgd = pgd_offset(init_mm_ptr, vaddr);
	if (pgd_none(*pgd) || pgd_bad(*pgd)) {
		module_err("Invalid PGD\n");
		return;
	}
	decode_pgd(*pgd);


	p4d = p4d_offset(pgd, vaddr);
	if (p4d_none(*p4d) || p4d_bad(*p4d)) {
		module_err("Invalid P4D\n");
		return;
	}
	decode_p4d(*p4d);

	pud = pud_offset(p4d, vaddr);
	if (pud_none(*pud) || pud_bad(*pud)) {
		module_err("Invalid PUD\n");
		return;
	}
	decode_pud(*pud);

	pmd = pmd_offset(pud, vaddr);
	if (pmd_none(*pmd) || pmd_bad(*pmd)) {
		module_err("Invalid PMD\n");
		return;
	}
	decode_pmd(*pmd);

	if (!(pmd_val(*pmd) & PTE_VALID)) {
		module_err("PMD not present\n");
		return;
	}

	pte = pte_offset_kernel(pmd, vaddr);
	if (!pte) {
		module_err("Invalid PTE\n");
		return;
	}
	decode_pte(*pte);
}

static pte_t* get_pte(void *addr) {
	pgd_t *pgd = NULL;
	p4d_t *p4d = NULL;
	pud_t *pud = NULL;
	pmd_t *pmd = NULL;
	pte_t *pte = NULL;
	size_t vaddr = (size_t)addr;


	pgd = pgd_offset(init_mm_ptr, vaddr);
	if (pgd_none(*pgd) || pgd_bad(*pgd)) {
		module_err("Invalid PGD\n");
		return NULL;
	}


	p4d = p4d_offset(pgd, vaddr);
	if (p4d_none(*p4d) || p4d_bad(*p4d)) {
		module_err("Invalid P4D\n");
		return NULL;
	}

	pud = pud_offset(p4d, vaddr);
	if (pud_none(*pud) || pud_bad(*pud)) {
		module_err("Invalid PUD\n");
		return NULL;
	}

	pmd = pmd_offset(pud, vaddr);
	if (pmd_none(*pmd) || pmd_bad(*pmd)) {
		module_err("Invalid PMD\n");
		return NULL;
	}

	if (!(pmd_val(*pmd) & PTE_VALID)) {
		module_err("PMD not present\n");
		return NULL;
	}

	pte = pte_offset_kernel(pmd, vaddr);
	if (!pte) {
		module_err("Invalid PTE\n");
		return NULL;
	}

	return pte;
}

static inline pte_t pte_clear_flags_custom(pte_t pte, unsigned long flags) {
	return __pte(pte_val(pte) & ~flags);
}

void disable_mte_for_region(void* start, size_t size) {
	size_t end = (size_t)start + size;
	size_t addr = (size_t)start;
	pte_t *pte = NULL;

	if(NULL == init_mm_ptr) {
		load_init_mm();
	}
	
	for (; addr < end; addr += PAGE_SIZE) {

		pte = get_pte((void*)addr);

		if (NULL == pte) {
			continue;
		}
	
		module_err("value before: %d", pte_val(*pte) & PTE_ATTRINDX_MASK);
		// Remove the MTE bit (bit 54) from the PTE
		*pte = pte_clear_flags_custom(*pte, PTE_ATTRINDX_MASK);
	
		module_err("value after: %d", pte_val(*pte) & PTE_ATTRINDX_MASK);
		// Ensure the update takes effect
		flush_tlb_kernel_range(addr, addr + PAGE_SIZE);
	}
}

