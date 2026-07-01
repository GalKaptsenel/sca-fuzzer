#ifndef ARM64_EXECUTOR_MEASUREMENT_H
#define ARM64_EXECUTOR_MEASUREMENT_H
#include "main.h"
#include "userapi/executor_user_api.h"

typedef struct measurement {
	uint64_t htrace[HTRACE_WIDTH];
	uint64_t pfc[NUM_PFC];
} measurement_t;

int execute(void);
int64_t initialize_measurement(measurement_t*);
void free_measurement(measurement_t*);

#endif // ARM64_EXECUTOR_MEASUREMENT_H
