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
#define UAPI_OVERFLOW_REGION_SIZE	        UAPI_PAGESIZE

/* htrace channels: [0]=L1D, [1]=L2 (reserved, not yet populated), [2]=BTB (branch-target P+P) */
#define HTRACE_WIDTH	(3)
#define HTRACE_L1D	(0)
#define HTRACE_L2	(1)
#define HTRACE_BTB	(2)
#define NUM_PFC		    (3)

#define REVISOR_EXECUTOR_ABI_VERSION	(2)

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
