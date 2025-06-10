#ifndef EXECUTOR_USERLAND_UTILS_H
#define EXECUTOR_USERLAND_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <sys/stat.h>
#include <time.h>
#include <ctype.h>

// Configuration
#define KB					(1024)
#define PAGESIZEW				(4 * KB)
#define MAIN_REGION_SIZE		        PAGESIZEW
#define FAULTY_REGION_SIZE		        PAGESIZEW
#define MEMORY_INPUT_SIZE		        (MAIN_REGION_SIZE + FAULTY_REGION_SIZE)

#define REG_INITIALIZATION_REGION_SIZE_ALIGNED			(4 * KB)

#define HTRACE_WIDTH	(1)
#define NUM_PFC		    (3)
#define WIDTH_MEMORY_IDS_BITMAP_BITS	(128) // size of sve vector
#define WIDTH_MEMORY_IDS        (WIDTH_MEMORY_IDS_BITMAP_BITS / (8 * sizeof(uint64_t)))

typedef struct measurement {
	uint64_t htrace[HTRACE_WIDTH];
	uint64_t pfc[NUM_PFC];
	uint64_t memory_ids_bitmap[WIDTH_MEMORY_IDS];
} measurement_t;

typedef struct registers {
	uint64_t x0;
	uint64_t x1;
	uint64_t x2;
	uint64_t x3;
	uint64_t x4;
	uint64_t x5;
	uint64_t flags;
	uint64_t sp;
} registers_t;

typedef struct Input {

	char main_region[MAIN_REGION_SIZE];
	char faulty_region[FAULTY_REGION_SIZE];
	union {
		char regs_region[REG_INITIALIZATION_REGION_SIZE_ALIGNED];
		registers_t regs;
	};
} input_t;

struct input_and_id_pair {
    input_t input;
    int64_t id;
};

struct input_batch {
    uint64_t size;
    struct input_and_id_pair array[];
};


#define USER_CONTROLLED_INPUT_LENGTH	(MAIN_REGION_SIZE + FAULTY_REGION_SIZE + sizeof(registers_t))

#endif // EXECUTOR_USERLAND_UTILS_H
