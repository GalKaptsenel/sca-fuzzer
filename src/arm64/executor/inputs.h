#ifndef ARM64_EXECUTOR_INPUTS_H
#define ARM64_EXECUTOR_INPUTS_H

#include "main.h"

#define REG_INITIALIZATION_REGION_SIZE_ALIGNED			(4 * KB)

typedef struct registers {
	u64 x0;
	u64 x1;
	u64 x2;
	u64 x3;
	u64 x4;
	u64 x5;
	u64 flags;
	u64 sp;
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

int initialize_inputs_db();
void free_inputs_db();
int insert_input(input_t *input);
input_t* find_input(int id);
void remove_input(int id);
void destroy_inputs_db(void);

#endif // ARM64_EXECUTOR_INPUTS_H
