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
#define UAPI_MEMORY_INPUT_SIZE		        (MAIN_REGION_SIZE + FAULTY_REGION_SIZE)

#define UAPI_REG_INITIALIZATION_REGION_SIZE_ALIGNED			(4 * UAPI_KB)

#define HTRACE_WIDTH	(1)
#define NUM_PFC		    (3)
#define WIDTH_MEMORY_IDS_BITMAP_BITS	(128) // size of sve vector
#define WIDTH_MEMORY_IDS        (WIDTH_MEMORY_IDS_BITMAP_BITS / (8 * sizeof(uint64_t)))

typedef struct user_measurement {
	uint64_t htrace[HTRACE_WIDTH];
	uint64_t pfc[NUM_PFC];
	uint64_t memory_ids_bitmap[WIDTH_MEMORY_IDS];
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

typedef struct user_input {

	char main_region[UAPI_MAIN_REGION_SIZE];
	char faulty_region[UAPI_FAULTY_REGION_SIZE];
	union {
		char regs_region[UAPI_REG_INITIALIZATION_REGION_SIZE_ALIGNED];
		user_registers_t regs;
	};
} user_input_t;

#define USER_CONTROLLED_INPUT_LENGTH	(MAIN_REGION_SIZE + FAULTY_REGION_SIZE + sizeof(user_registers_t))

#endif // EXECUTOR_USERAPI_H
