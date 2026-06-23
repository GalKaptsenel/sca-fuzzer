/* Host-compiled unit test for the pure PMU-capability logic (pmu_logic.h).
 * Oracle: the ARM ARM (ID_AA64DFR0_EL1.PMUVer bits [11:8]; PMCR_EL0.N bits [15:11])
 * and Revizor's REQUIRED_PMU_COUNTERS. No kernel, no hardware. */
#include <stdio.h>
#include "pmu_logic.h"

static int failed = 0;
#define CHECK(cond) do { \
	if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, #cond); ++failed; } \
} while (0)

#define DFR0_PMUVER(v)  ((unsigned long long)(v) << 8)   /* ID_AA64DFR0_EL1.PMUVer */
#define PMCR_N(n)       ((unsigned long long)(n) << 11)   /* PMCR_EL0.N */

int main(void) {
	/* No PMUv3 (PMUVer 0) or IMPDEF non-PMUv3 (0xf): no usable counters, whatever PMCR reads. */
	CHECK(0 == pmu_event_counters_decode(DFR0_PMUVER(0x0), PMCR_N(8)));
	CHECK(0 == pmu_event_counters_decode(DFR0_PMUVER(0xf), PMCR_N(8)));
	CHECK(!pmu_has_pmuv3(DFR0_PMUVER(0x0)));
	CHECK(!pmu_has_pmuv3(DFR0_PMUVER(0xf)));
	CHECK(pmu_has_pmuv3(DFR0_PMUVER(0x1)));
	CHECK(pmu_has_pmuv3(DFR0_PMUVER(0x6)));

	/* PMUv3 present: counters == PMCR_EL0.N. */
	CHECK(0 == pmu_event_counters_decode(DFR0_PMUVER(0x1), PMCR_N(0)));
	CHECK(1 == pmu_event_counters_decode(DFR0_PMUVER(0x1), PMCR_N(1)));
	CHECK(4 == pmu_event_counters_decode(DFR0_PMUVER(0x1), PMCR_N(4)));
	CHECK(31 == pmu_event_counters_decode(DFR0_PMUVER(0x6), PMCR_N(31)));

	/* Decode reads only PMCR.N: unrelated high/low bits must not bleed in. */
	CHECK(4 == pmu_event_counters_decode(DFR0_PMUVER(0x1), PMCR_N(4) | 0x7ff | (0xffffull << 32)));

	/* Sufficiency is `>= REQUIRED_PMU_COUNTERS`: check the boundary either side so a
	 * wrong operator (>, <=, ==) is caught, while the required count stays single-sourced. */
	CHECK(!pmu_counters_sufficient(0));
	CHECK(!pmu_counters_sufficient(REQUIRED_PMU_COUNTERS - 1));
	CHECK(pmu_counters_sufficient(REQUIRED_PMU_COUNTERS));
	CHECK(pmu_counters_sufficient(REQUIRED_PMU_COUNTERS + 1));

	/* The VM case: PMUv3 present but 0 programmable counters -> measurement unsupported. */
	CHECK(!pmu_counters_sufficient(pmu_event_counters_decode(DFR0_PMUVER(0x1), PMCR_N(0))));

	if (failed) {
		printf("\n%d pmu_logic test(s) FAILED\n", failed);
		return 1;
	}
	printf("all pmu_logic tests passed\n");
	return 0;
}
