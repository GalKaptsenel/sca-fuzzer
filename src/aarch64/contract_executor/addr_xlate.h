#ifndef CE_ADDR_XLATE_H
#define CE_ADDR_XLATE_H

/* Pure kaddr<->uaddr translation arithmetic (the simulation_output.c wrappers pass the live globals).
 * kaddr = the sandbox address the test computes (based at kbase, a TTBR1 VA); uaddr = the matching
 * offset into the CE's simulation_memory. kaddr2uaddr masks the top byte (TBI tag), so
 * uaddr2kaddr(kaddr2uaddr(k)) restores k only when k's top byte already equals kbase's. */

#define CE_TBI_ADDR_MASK 0x00FFFFFFFFFFFFFFull

static inline uintptr_t kaddr2uaddr_calc(uintptr_t kaddr, uintptr_t kbase, uintptr_t sim_mem) {
	return sim_mem + ((kaddr & CE_TBI_ADDR_MASK) - (kbase & CE_TBI_ADDR_MASK));
}

static inline uintptr_t uaddr2kaddr_calc(uintptr_t uaddr, uintptr_t kbase, uintptr_t sim_mem) {
	return kbase + (uaddr - sim_mem);
}

#endif /* CE_ADDR_XLATE_H */
