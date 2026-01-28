#ifndef SIMULATION_INPUT_H
#define SIMULATION_INPUT_H 

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include "stream_ipc.h"

/* ============================
 * Constants & magic
 * ============================ */

#define RVZR_MAGIC 0x525A5652u  /* "RVZR" */
#define RVZR_VERSION 1

/* Architectures */
enum sim_arch {
    RVZR_ARCH_X86_64  = 1,
    RVZR_ARCH_AARCH64 = 2,
};

/* Flags */
enum sim_flags {
	RVZR_FLAG_NONE		= 0,
	RVZR_FLAG_HAS_CODE	= 1 << 0,
	RVZR_FLAG_HAS_REGS	= 1 << 1,
	RVZR_FLAG_HAS_MEMORY	= 1 << 2,
};

enum config_flags {
	CONFIG_FLAG_NONE		= 0,
	CONFIG_FLAG_REQ_CODE_BASE_PHYS	= 1 << 0,
	CONFIG_FLAG_REQ_CODE_BASE_VIRT	= 1 << 1,
	CONFIG_FLAG_REQ_MEM_BASE_PHYS	= 1 << 2,
	CONFIG_FLAG_REQ_MEM_BASE_VIRT	= 1 << 3,
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
	uint64_t max_misspred_instructioins; // NOT SUPPPORTED
	uint64_t requested_code_base_phys; // NOT SUPPORTED
	uint64_t requested_code_base_virt;
	uint64_t requested_mem_base_phys; // NOT SUPPORTED
	uint64_t requested_mem_base_virt;
};

struct input_header {
    uint32_t magic;
    uint16_t version;
    uint16_t arch;

    uint64_t flags;
    struct configuration config;

    uint64_t code_size;   /* bytes */
    uint64_t mem_size;    /* bytes */
    uint64_t regs_size;   /* bytes */

    uint64_t reserved;    /* must be 0 */
};

/* ============================
 * In-memory representation
 * ============================ */

struct simulation_input {
    struct input_header hdr;

    uint8_t *code;   /* machine code */
    uint8_t *memory; /* initial memory image */
    uint8_t *regs;   /* architecture-specific register blob */
};

/* ============================
 * API
 * ============================ */

/* Load test case from file path */
int simulation_input_load(const char* path, struct simulation_input* sim_input);

/* Load test case from file descriptor (supports stdin) */
int simulation_input_load_fd(int fd, struct simulation_input* sim_input);

/* Load test case from shared memory */
int simulation_input_load_shm(struct shm_region* shm, struct simulation_input* sim_input);

/* Free all allocated buffers */
void simulation_input_free(struct simulation_input* sim_input);

/* Validate header sanity */
int simulation_input_validate_header(const struct input_header* hdr);

/* Utility: total payload size */
size_t simulation_input_payload_size(const struct input_header* hdr);

#endif // SIMULATION_INPUT_H

