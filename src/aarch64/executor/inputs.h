#ifndef ARM64_EXECUTOR_INPUTS_H
#define ARM64_EXECUTOR_INPUTS_H

#include "main.h"

#define REG_INITIALIZATION_REGION_SIZE_ALIGNED			(4 * KB)

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

struct input_node {
	int id;
	input_t input;
	measurement_t measurement;
	struct rb_node node;
};

struct input_and_id_pair {
    input_t* input;
    ssize_t id;
}

struct input_batch {
    uint64_t size;
    input_and_id_pair array[];
};

#define USER_CONTROLLED_INPUT_LENGTH	(MAIN_REGION_SIZE + FAULTY_REGION_SIZE + sizeof(registers_t))

void initialize_inputs_db(void);
int allocate_input(void);
measurement_t* get_measurement(int id);
input_t* get_input(int id);
void remove_input(int id);
void destroy_inputs_db(void);

#endif // ARM64_EXECUTOR_INPUTS_H
