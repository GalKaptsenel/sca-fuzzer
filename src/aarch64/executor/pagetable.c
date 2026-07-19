#include "main.h"

static struct mm_struct *init_mm_ptr	=	NULL;

static void __nocfi load_init_mm(void) {
	load_global_symbol(kallsyms_lookup_name_fn, struct mm_struct *, init_mm_ptr, init_mm);
}

static bool ensure_init_mm(void) {
	if (NULL == init_mm_ptr) {
		load_init_mm();
	}
	return NULL != init_mm_ptr;
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
	module_info("  %llx\t  %llx\t  %llx\t  %llx\t  %llx\t\t %llx\t  %llx\t  %llx\t  %llx\t  %llx\t",
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

	if (!ensure_init_mm()) {
		module_err("init_mm unavailable (kallsyms lookup failed)\n");
		return;
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
	if (NULL == pte) {
		module_err("Invalid PTE\n");
		return;
	}
	decode_pte(*pte);
}

/* Leaf descriptor mapping `addr`, at whatever level: a PUD/PMD block or a PTE. Block and page
 * descriptors share the PTE_ATTRINDX layout, so this is for read-only attribute inspection only. */
static pte_t* get_leaf_entry(void *addr) {
	size_t vaddr = (size_t)addr;
	pgd_t *pgd; p4d_t *p4d; pud_t *pud; pmd_t *pmd;

	if (NULL == init_mm_ptr) { return NULL; }
	pgd = pgd_offset(init_mm_ptr, vaddr);
	if (pgd_none(*pgd) || pgd_bad(*pgd)) { return NULL; }
	p4d = p4d_offset(pgd, vaddr);
	if (p4d_none(*p4d) || p4d_bad(*p4d)) { return NULL; }
	pud = pud_offset(p4d, vaddr);
	if (pud_none(*pud)) { return NULL; }
	if (pud_sect(*pud)) { return (pte_t*)pud; }
	if (pud_bad(*pud)) { return NULL; }
	pmd = pmd_offset(pud, vaddr);
	if (pmd_none(*pmd)) { return NULL; }
	if (pmd_sect(*pmd)) { return (pte_t*)pmd; }
	if (pmd_bad(*pmd)) { return NULL; }
	return pte_offset_kernel(pmd, vaddr);
}

// Whether every page in [start, start+size) is mapped with MAIR attribute index `attrindx`.
bool pte_region_attr_is(void* start, size_t size, unsigned int attrindx) {
	size_t addr = (size_t)start;
	size_t end = (size_t)start + size;

	if (!ensure_init_mm()) {
		module_err("init_mm unavailable (kallsyms lookup failed)\n");
		return false;
	}

	for (; addr < end; addr += PAGE_SIZE) {
		pte_t *pte = get_leaf_entry((void*)addr);
		if (NULL == pte || (pte_val(*pte) & PTE_ATTRINDX_MASK) != PTE_ATTRINDX(attrindx)) {
			return false;
		}
	}
	return true;
}

/* Raw leaf descriptor for `va` (0 if unmapped) — read-only attribute inspection. */
uint64_t leaf_pte_val(void *va) {
	pte_t *pte;
	if (!ensure_init_mm()) { return 0; }
	pte = get_leaf_entry(va);
	return NULL == pte ? 0 : (uint64_t)pte_val(*pte);
}

/* True only when `va` is mapped at 4K PTE granularity (never a block) — a guard so a PTE write can
 * never accidentally touch a 2MB/1GB block covering unrelated memory. */
static bool is_4k_pte_leaf(void *va) {
	size_t v = (size_t)va;
	pgd_t *pgd; p4d_t *p4d; pud_t *pud; pmd_t *pmd;
	if (NULL == init_mm_ptr) { return false; }
	pgd = pgd_offset(init_mm_ptr, v); if (pgd_none(*pgd) || pgd_bad(*pgd)) { return false; }
	p4d = p4d_offset(pgd, v);         if (p4d_none(*p4d) || p4d_bad(*p4d)) { return false; }
	pud = pud_offset(p4d, v);         if (pud_none(*pud) || pud_sect(*pud) || pud_bad(*pud)) { return false; }
	pmd = pmd_offset(pud, v);         if (pmd_none(*pmd) || pmd_sect(*pmd) || pmd_bad(*pmd)) { return false; }
	return true;
}

/* Overwrite the 4K leaf PTE for `va` with `newval`, invalidating the TLB for that page. Refuses (and
 * returns false) unless `va` is 4K-mapped. On success `*old_out` holds the prior descriptor so the
 * caller can restore it. Inner-shareable TLBI so all CPUs see the change. */
bool write_leaf_pte(void *va, uint64_t newval, uint64_t *old_out) {
	pte_t *pte;
	if (!ensure_init_mm() || !is_4k_pte_leaf(va)) { return false; }
	pte = get_leaf_entry(va);
	if (NULL == pte) { return false; }
	*old_out = (uint64_t)pte_val(*pte);
	WRITE_ONCE(*pte, __pte(newval));
	asm volatile("dsb ishst" ::: "memory");
	asm volatile("tlbi vaae1is, %0" :: "r"(((uint64_t)va) >> 12) : "memory");
	asm volatile("dsb ish\n isb" ::: "memory");
	return true;
}

