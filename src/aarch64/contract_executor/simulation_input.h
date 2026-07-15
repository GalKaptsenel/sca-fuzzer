#ifndef SIMULATION_INPUT_H
#define SIMULATION_INPUT_H 

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <fcntl.h>
#include <unistd.h>
#include "stream_ipc.h"
#include "common_msg_constants.h"

#define MAX_PAYLOAD_SIZE	(1 << 21)

/* Flags */
enum sim_flags {
	RVZR_FLAG_NONE		= 0,
	RVZR_FLAG_HAS_CODE	= 1 << 0,
	RVZR_FLAG_HAS_INPUT	= 1 << 1,   /* the shared input initialization (executor_input_format) follows code */
};

enum config_flags {
	CONFIG_FLAG_NONE		= 0,
	CONFIG_FLAG_REQ_CODE_BASE_PHYS	= 1 << 0,
	CONFIG_FLAG_REQ_CODE_BASE_VIRT	= 1 << 1,
	CONFIG_FLAG_REQ_MEM_BASE_PHYS	= 1 << 2,
	CONFIG_FLAG_REQ_MEM_BASE_VIRT	= 1 << 3,
};

/* Contracts are COMPOSABLE: `configuration.execution_clauses` is a BITMASK of clauses, so
 * several speculation behaviours can be enabled at once (e.g. cond | bpas == cond-bpas).
 * Each bit enables one execution clause (see execution_clauses.c). */
#define EXEC_CLAUSE_COND  (1u << 0)  /* mispredict every conditional branch                     */
#define EXEC_CLAUSE_BPAS  (1u << 1)  /* speculatively bypass stores (read stale memory)         */
#define EXEC_CLAUSE_BPU   (1u << 2)  /* mispredict branches per the selected branch predictor   */
#define EXEC_CLAUSE_BARRIER (1u << 3)/* honor barriers: cut speculation a fencing barrier stops  */
#define EXEC_CLAUSE_SLS   (1u << 4)  /* straight-line speculation: explore pc+4 past a branch    */
/* seq / arch-only == no clauses enabled (execution_clauses == 0). */

/* Branch predictor EXEC_CLAUSE_BPU uses, selected via the input (see branch_predictors.c).
 * Extend by adding an enum value here + a registry entry. */
enum branch_predictor_id {
	BRANCH_PREDICTOR_NONE        = 0,
	BRANCH_PREDICTOR_NEOVERSE_N3 = 1,
};

/* ============================
 * On-disk header format
 * ============================ */

/*
 * This struct is written verbatim to disk.
 * All fields are little-endian.
 */

struct configuration {
	uint64_t flags;
	uint64_t max_misspred_branch_nesting;
	uint64_t max_misspred_instructions; // per-window instruction cap (0 = no cap)
	uint64_t requested_code_base_phys; // NOT SUPPORTED
	uint64_t requested_code_base_virt;
	uint64_t requested_mem_base_phys; // NOT SUPPORTED
	uint64_t requested_mem_base_virt;
	uint64_t execution_clauses;        /* bitmask of EXEC_CLAUSE_*; 0 = seq (no speculation) */
	uint64_t branch_predictor;         /* enum branch_predictor_id; used iff EXEC_CLAUSE_BPU */
};

/* Envelope for one contract-executor message: exec-params (config) + code + a shared input initialization
 * (executor_input_format). All members are u64. The payload after this header is `code` followed by
 * the input initialization; memory/regs/mte-tags/pac-keys are sections inside that input_init. */
struct input_header {
    uint64_t magic;       /* RVZRCE_MAGIC */
    uint64_t version;     /* RVZRCE_VERSION */
    uint64_t arch;

    uint64_t flags;
    struct configuration config;

    uint64_t code_size;   /* bytes of machine code following the header */
    uint64_t input_init_size;   /* bytes of the input initialization following the code */

    uint64_t reserved;    /* must be 0 */
};

/* Wire ABI must match the Python encoder (ContractExecution.encode). */
_Static_assert(sizeof(struct configuration) == 9 * sizeof(uint64_t), "configuration ABI mismatch");
_Static_assert(sizeof(struct input_header) == 16 * sizeof(uint64_t), "input_header ABI mismatch");

/* ============================
 * In-memory representation
 * ============================ */

struct simulation_input {
    struct input_header hdr;

    uint8_t *code;     /* machine code */
    uint8_t *memory;   /* initial memory image (main || faulty from the input initialization) */
    size_t   mem_size;
    uint8_t *regs;     /* register data (gpr section); slot 6 (x6/NZCV) is already in PSTATE format */
    size_t   regs_size;
    uint8_t *mte_tags; /* one tag per granule (unpacked), or NULL */
    size_t   mte_tag_count;
    uint64_t pac_keys[10];
    bool     pac_keys_present;
};

/* ============================
 * API
 * ============================ */

/* Load test case from file path */
int simulation_input_load_path(const char* path, struct simulation_input* sim_input);

/* Load test case from file descriptor (supports stdin) */
int simulation_input_load_fd(int fd, struct simulation_input* sim_input);

/* Load test case from shared memory */
int simulation_input_from_file(FILE* f, struct simulation_input* sim_input);

/* Free all allocated buffers */
void simulation_input_free(struct simulation_input* sim_input);

/* Validate header sanity */
int simulation_input_validate_header(const struct input_header* hdr);

/* Utility: total payload size */
size_t simulation_input_payload_size(const struct input_header* hdr);

#endif // SIMULATION_INPUT_H

