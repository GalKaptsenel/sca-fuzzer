#ifndef ARM64_EXECUTOR_PMU_LOGIC_H
#define ARM64_EXECUTOR_PMU_LOGIC_H

/* Pure PMU-capability decoding, free of kernel headers so it can be host-compiled
 * and unit-tested (see test_pmu.c). The register reads themselves live in pmu.c. */

#define REQUIRED_PMU_COUNTERS	(4)   /* config_pfc programs event counters 0..3 */

/* ID_AA64DFR0_EL1.PMUVer (bits [11:8]): 0 = not implemented, 0xf = IMPDEF (not PMUv3). */
static inline int pmu_has_pmuv3(unsigned long long id_aa64dfr0) {
	unsigned int pmuver = (unsigned int)((id_aa64dfr0 >> 8) & 0xf);
	return 0u != pmuver && 0xfu != pmuver;
}

/* Implemented event counters: PMCR_EL0.N (bits [15:11]) when PMUv3 is present, else 0. */
static inline unsigned int pmu_event_counters_decode(unsigned long long id_aa64dfr0,
                                                     unsigned long long pmcr) {
	return pmu_has_pmuv3(id_aa64dfr0) ? (unsigned int)((pmcr >> 11) & 0x1f) : 0u;
}

static inline int pmu_counters_sufficient(unsigned int n) {
	return n >= REQUIRED_PMU_COUNTERS;
}

#endif // ARM64_EXECUTOR_PMU_LOGIC_H
