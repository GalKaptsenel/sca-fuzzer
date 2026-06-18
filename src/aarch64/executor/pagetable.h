#ifndef ARM64_EXECUTOR_PAGETABLE_H
#define ARM64_EXECUTOR_PAGETABLE_H

void page_walk_explorer(void *addr);
void disable_mte_for_region(void* start, size_t size);
bool pte_region_attr_is(void* start, size_t size, unsigned int attrindx);

#endif // ARM64_EXECUTOR_PAGETABLE_H
