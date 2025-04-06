#ifndef ARM64_EXECUTOR_MEASUREMENT_H
#define ARM64_EXECUTOR_MEASUREMENT_H

#define HTRACE_WIDTH	(1)
#define NUM_PFC		    (3)
#define WIDTH_MEMORY_IDS_BITS    (128)
#define WIDTH_MEMORY_IDS        (WIDTH_MEMORY_IDS_BITS / (sizeof(uint64_t) * 8))

typedef struct measurement {
    uint64_t htrace[HTRACE_WIDTH];
    uint64_t pfc[NUM_PFC];
    uint64_t memory_ids_bitmap[WIDTH_MEMORY_IDS];
} measurement_t;

int execute(void);
void initialize_measurement(measurement_t*);

#endif // ARM64_EXECUTOR_MEASUREMENT_H
