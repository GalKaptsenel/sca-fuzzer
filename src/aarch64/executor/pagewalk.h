#ifndef ARM64_EXECUTOR_PAGEWALK_H
#define ARM64_EXECUTOR_PAGEWALK_H

void page_walk_explorer(void *addr);
void disable_mte_for_region(void* start, size_t size);

#endif // ARM64_EXECUTOR_PAGEWALK_H
