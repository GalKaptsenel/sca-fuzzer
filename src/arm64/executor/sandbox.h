#ifndef ARM64_EXECUTOR_SANDBOX_H
#define ARM64_EXECUTOR_SANDBOX_H

#include "main.h"

// Configuration 
#define WORKING_MEMORY_SIZE		MB
#define MAIN_REGION_SIZE		PAGESIZE
#define FAULTY_REGION_SIZE		PAGESIZE
#define MEMORY_INPUT_SIZE		(MAIN_REGION_SIZE + FAULTY_REGION_SIZE)
#define OVERFLOW_REGION_SIZE		PAGESIZE
#define REG_INITIALIZATION_REGION_SIZE	64
#define EVICT_REGION_SIZE		L1D_SIZE // TODO What is it? why is it like this?

typedef struct sandbox {
    char eviction_region[EVICT_REGION_SIZE];   // region used in Prime+Probe for priming
    char lower_overflow[OVERFLOW_REGION_SIZE]; // zero-initialized region for accidental overflows
    char main_region[MAIN_REGION_SIZE];        // first input page. does not cause faults
    char faulty_region[FAULTY_REGION_SIZE];    // second input. causes a (configurable) fault
    char upper_overflow[OVERFLOW_REGION_SIZE]; // zero-initialized region for accidental overflows
    uint64_t stored_rsp;
    measurement_t latest_measurement; // measurement results
} sandbox_t;

#endif // ARM64_EXECUTOR_SANDBOX_H
