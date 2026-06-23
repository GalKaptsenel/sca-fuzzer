#include "main.h"
#include "pmu.h"

/* PMCR_EL0.N event counters, or 0 when PMUv3 is not implemented. PMCR_EL0 (and
 * PMEVTYPER<n>/PMEVCNTR<n> for n >= N) is UNDEFINED without PMUv3, so read it only
 * once PMUv3 is confirmed -- a VM may virtualize fewer counters than the host. */
unsigned int pmu_event_counters(void) {
	uint64_t dfr0 = 0, pmcr = 0;

	asm volatile("mrs %0, id_aa64dfr0_el1" : "=r" (dfr0));
	if (pmu_has_pmuv3(dfr0)) {
		asm volatile("mrs %0, pmcr_el0" : "=r" (pmcr));
	}
	return pmu_event_counters_decode(dfr0, pmcr);
}

bool pmu_measurement_supported(void) {
	return pmu_counters_sufficient(pmu_event_counters());
}

/* Clears and reconfigures the programmable performance counters. -ENODEV when the
 * PMU has too few counters (touching PMEVTYPER<n> for n >= PMCR_EL0.N would fault). */
int config_pfc(void) {
	uint64_t val = 0;
	uint64_t filter_events = (1 << 30) | (1 << 27) | (1 << 26);

	if (!pmu_measurement_supported()) {
		module_err("PMU exposes %u event counters; measurement needs %d\n",
		           pmu_event_counters(), REQUIRED_PMU_COUNTERS);
		return -ENODEV;
	}

	// disable PMU user-mode access (not necessary?)
	asm volatile("msr pmuserenr_el0, %0" :: "r" (0x1));
	asm volatile("isb\n");

	// disable PMU counters before selecting the event we want
	val = 0;
	asm volatile("mrs %0, pmcr_el0" : "=r" (val));
	asm volatile("msr pmcr_el0, %0" :: "r" ((uint64_t)0x0));
	asm volatile("isb\n");
	asm volatile("msr pmcntenclr_el0, %0" :: "r" ((uint64_t)0b1111));
	asm volatile("isb\n");

	// select events:
	// 1. L1D cache refills (0x3)
	asm volatile("msr pmevtyper0_el0, %0" :: "r" ((uint64_t)(filter_events | 0x03)));
	asm volatile("isb\n");

	// 2. Instructions retired (0x08)
	asm volatile("msr pmevtyper1_el0, %0" :: "r" ((uint64_t)(filter_events | 0x08)));
	asm volatile("isb\n");

	// 3. Instruction speculatively executed (0x1b)
	asm volatile("msr pmevtyper2_el0, %0" :: "r" ((uint64_t)(filter_events | 0x1b)));
	asm volatile("isb\n");

	// 4. Branch instruction architecturally executed, mispredicted immediate (0x8111)
	asm volatile("msr pmevtyper3_el0, %0" :: "r" ((uint64_t)(filter_events | 0x8111)));
	asm volatile("isb\n");

	// enable counting
	asm volatile("msr pmcntenset_el0, %0" :: "r" (((uint64_t)0b1111) | (1ULL << 31)));
	asm volatile("isb\n");

	// enable PMU counters and reset the counters (using 3 bits)
	val = 0;
	asm volatile("mrs %0, pmcr_el0" : "=r" (val));
	asm volatile("msr pmcr_el0, %0" :: "r" (val | 0b111));
	asm volatile("isb\n");

	return 0;
}
