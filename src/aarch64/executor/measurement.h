#ifndef ARM64_EXECUTOR_MEASUREMENT_H
#define ARM64_EXECUTOR_MEASUREMENT_H
#include "main.h"

#define HTRACE_WIDTH	(1)
#define NUM_PFC		    (3)
#define WIDTH_MEMORY_IDS_BITMAP_BITS	(128)
#define WIDTH_MEMORY_IDS        (WIDTH_MEMORY_IDS_BITMAP_BITS / (8 * sizeof(uint64_t)))

typedef struct measurement {
	uint64_t htrace[HTRACE_WIDTH];
	uint64_t pfc[NUM_PFC];
	uint64_t memory_ids_bitmap[WIDTH_MEMORY_IDS];
	struct aux_buffer_t* aux_buffer;
} measurement_t;

int execute(void);
int64_t initialize_measurement(measurement_t*);
void free_measurement(measurement_t*);

#endif // ARM64_EXECUTOR_MEASUREMENT_H
