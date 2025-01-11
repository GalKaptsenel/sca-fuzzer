#ifndef ARM64_EXECUTOR_MEASUREMENT_H
#define ARM64_EXECUTOR_MEASUREMENT_H

#define HTRACE_WIDTH	(1)
#define NUM_PFC		(3)

typedef struct measurement {
    uint64_t htrace[HTRACE_WIDTH];
    uint64_t pfc[NUM_PFC];
} measurement_t;

int execute(void);
void initialize_measurement(measurement_t*);

#endif // ARM64_EXECUTOR_MEASUREMENT_H
