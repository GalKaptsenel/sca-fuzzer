#ifndef ARM64_EXECUTOR_PMU_H
#define ARM64_EXECUTOR_PMU_H
#include <linux/types.h>
#include "pmu_logic.h"

unsigned int pmu_event_counters(void);        /* PMCR_EL0.N, or 0 when PMUv3 is absent */
bool         pmu_measurement_supported(void); /* >= REQUIRED_PMU_COUNTERS available */
int          config_pfc(void);                /* program the event counters; -ENODEV if too few */

#endif // ARM64_EXECUTOR_PMU_H
