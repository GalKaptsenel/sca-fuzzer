#ifndef EXECUTOR_USERAPI_H
#define EXECUTOR_USERAPI_H

#ifdef __KERNEL__
	#include <linux/types.h>
#else
	#include <stdint.h>
#endif

#define UAPI_KB					(1024)
#define UAPI_PAGESIZE				(4 * UAPI_KB)

// Configuration
#define UAPI_MAIN_REGION_SIZE		        UAPI_PAGESIZE
#define UAPI_FAULTY_REGION_SIZE		        UAPI_PAGESIZE

#define HTRACE_WIDTH	(1)
#define NUM_PFC		    (3)

typedef struct user_measurement {
	uint64_t htrace[HTRACE_WIDTH];
	uint64_t pfc[NUM_PFC];
} user_measurement_t;

typedef struct user_registers {
	uint64_t x0;
	uint64_t x1;
	uint64_t x2;
	uint64_t x3;
	uint64_t x4;
	uint64_t x5;
	uint64_t flags;
	uint64_t sp;
} user_registers_t;

#define USER_CONTROLLED_INPUT_LENGTH	(UAPI_MAIN_REGION_SIZE + UAPI_FAULTY_REGION_SIZE + sizeof(user_registers_t))

#endif // EXECUTOR_USERAPI_H
