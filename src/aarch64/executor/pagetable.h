#ifndef ARM64_EXECUTOR_PAGETABLE_H
#define ARM64_EXECUTOR_PAGETABLE_H

void page_walk_explorer(void *addr);
bool pte_region_attr_is(void* start, size_t size, unsigned int attrindx);
uint64_t leaf_pte_val(void *va);
bool write_leaf_pte(void *va, uint64_t newval, uint64_t *old_out);

#endif // ARM64_EXECUTOR_PAGETABLE_H
