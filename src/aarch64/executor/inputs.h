#ifndef ARM64_EXECUTOR_INPUTS_H
#define ARM64_EXECUTOR_INPUTS_H

#include "main.h"
#include "pac.h"
#include "userapi/executor_input_format.h"

#define REG_INITIALIZATION_REGION_SIZE_ALIGNED			(4 * KB)

/* One MTE allocation tag per granule of the main|faulty span (see executor_input_format.h). */
#define INPUT_MTE_TAG_COUNT					(MEMORY_INPUT_SIZE / MTE_GRANULE_SIZE)

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

/*
 * Per-input execution-environment state, parsed from the wire format
 * (userapi/executor_input_format.h). main/faulty/regs are the classic initial
 * memory + registers; mte_tags / pac_keys are optional per-input initial state,
 * each guarded by its *_present flag.
 */
typedef struct Input {
	char main_region[MAIN_REGION_SIZE];
	char faulty_region[FAULTY_REGION_SIZE];
	union {
		char regs_region[REG_INITIALIZATION_REGION_SIZE_ALIGNED];
		registers_t regs;
	};
	uint8_t mte_tags[INPUT_MTE_TAG_COUNT];   /* one 4-bit tag per 16B granule, unpacked */
	bool mte_tags_present;
	struct pac_keys pac_keys;                /* per-input PAC key override */
	bool pac_keys_present;
	/* Relocations applied to the test case before this input executes; the terminator in
	 * slot 0 means no relocations. */
	struct revisor_code_reloc_entry code_reloc[REVISOR_INPUT_MAX_CODE_RELOCS + 1];
	/* Conditional branches trained to a requested direction before this input executes; the
	 * terminator in slot 0 means no branch training. */
	struct revisor_bpu_train_entry bpu_train[REVISOR_INPUT_MAX_BPU_TRAIN + 1];
} input_t;

struct input_node {
	int64_t id;
	input_t input;
	measurement_t measurement;
	struct rb_node node;
};

void initialize_inputs_db(void);
int64_t allocate_input(void);
measurement_t* get_measurement(int64_t id);
input_t* get_input(int64_t id);
void remove_input(int64_t id);
void destroy_inputs_db(void);

#endif // ARM64_EXECUTOR_INPUTS_H
