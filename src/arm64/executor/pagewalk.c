#include "main.h"

static struct mm_struct *init_mm_ptr	=	NULL;

static void __nocfi load_init_mm(void) { 
	load_global_symbol(kallsyms_lookup_name_fn, struct mm_struct *, init_mm_ptr, init_mm);
}

static void decode_entry(u64 entry, const char *level) {
	u64 present_bit = 0;
	u64 entry_type = 0;
	u64 accessed_flag = 0;
	u64 access_permissions = 0;
	u64 attr_index = 0;
	u64 sharability = 0;
	u64 dirty_bit = 0;
	u64 uxn_bit = 0;
	u64 pxn_bit = 0;
	u64 software_defined = 0;

	module_err("Decoding %s entry: 0x%016llx\n", level, entry);

	// Present Bit
	present_bit = entry & 0x1;
	if (present_bit) {
		module_err("%s: Present\n", level);
	}
	else {
		module_err("%s: Not Present\n", level);
		return;
	}

	// Page Type
	entry_type = (entry >> 1) & 0x1;
	if (entry_type) {
		module_err("%s: Table descriptor(levels 0, 1 and 2)/Table entry(levels 1 and 2)\n", level);
	}
	else {
		module_err("%s: Block entry(levels 1 and 2)\n", level);
	}

	// Accessed Flag
	accessed_flag = (entry >> 10) & 0x1;
	if (accessed_flag) { 
		module_err("%s: Has already been used [AF=1]\n", level);
	}
	else {
		module_err("%s: Has not yet been used [AF=0]\n", level);
	}

	// Access Permissions
	access_permissions = (entry >> 6) & 0x3;
	switch(access_permissions) {

		case 0:
			module_err("%s: No access for EL0 and RW for EL1/2/3", level);
			break;
		case 1:
			module_err("%s: RW for EL0 and RW for EL1/2/3", level);
			break;
		case 2:
			module_err("%s: No access for EL0 and Read-Only for EL1/2/3", level);
			break;
		case 3:
			module_err("%s: Read-Only for EL0 and Read-Only for EL1/2/3", level);
			break;
	}
	
	// Attributes
	attr_index = (entry >> 2) & 0x7;
	module_err("%s: Memory Attributes Index: %llu\n", level, attr_index);

	// Shareability
	sharability = (entry >> 8) & 0x3;
	switch (sharability) {

		case 0:
			module_err("%s: Non-shareable\n", level);
			break;
		case 1:
			module_err("%s: Unpredictable shareability\n", level);
			break;
		case 2:
			module_err("%s: Outer shareable\n", level);
			break;
		case 3:
			module_err("%s: Inner shareable\n", level);
			break;
	}

	// Dirty Bit
	dirty_bit = (entry >> 55) & 0x1;
	if (dirty_bit) { 
		module_err("%s: Dirty\n", level);
	}
	
	// UXN Bit
	uxn_bit = (entry >> 54) & 0x1;
	if (uxn_bit) { 
		module_err("%s: EL0 cannot execute code [UXN=1]\n", level);
	}
	else {
		module_err("%s: EL0 can execute code [UXN=0]\n", level);
	}

	// PXN Bit
	pxn_bit = (entry >> 53) & 0x1;
	if (pxn_bit) { 
		module_err("%s: EL1/2/3 cannot execute code [PXN=1]\n", level);
	}
	else {
		module_err("%s: EL1/2/3 can execute code [PXN=0]\n", level);
	}

	// Software bits (optional, custom implementation-defined)
	software_defined = (entry >> 55);
	module_err("%s: Software-defined bits: 0x%llx\n", level, software_defined);

	module_err("| P\t|Type\t|AF\t|AP\t|AttrIndx\t|SH\t|PXN\t|UXN\t| D\t|SW-defined\t");
	module_err("  %d\t  %d\t  %d\t  %d\t  %d\t\t %d\t  %d\t  %d\t  %d\t  %d\t", present_bit, entry_type, accessed_flag, access_permissions, attr_index,sharability, pxn_bit, uxn_bit, dirty_bit, software_defined);
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

void page_walk_decoder(void *addr) {
	
	//struct mm_struct *mm = current->mm; // Current process's memory descriptor
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

