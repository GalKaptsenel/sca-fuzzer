#ifndef SIMULATION_OUTPUT_H
#define SIMULATION_OUTPUT_H 

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <stdbool.h>
#include "simulation_state.h"
#include "stream_ipc.h"
#include "common_msg_constants.h"

#define NUM_GPRS 31

typedef struct {
	uint64_t effective_address;		// effective memory address
	uint64_t before;	// value before access
	uint64_t after;		// value after access
	uint64_t  element_size;		// size in bytes
	uint64_t  is_write;	// 1 = write, 0 = read
	uint64_t  is_atomic;	// 1 = read-modify-write (atomic/exclusive/CAS): reads AND writes the cell
} mem_access_t;

typedef struct {
	uint64_t gpr[NUM_GPRS];
	uint64_t sp;
	uint64_t pc;
	uint64_t nzcv;
	uint64_t encoding;		// Upper bits are 0 for aarch64
	size_t extra_data_size;
	// dynamic array of size extra_data_size
} trace_cpu_state_t;

typedef struct {
	uint64_t instr_index;		// instruction index in trace
	uint64_t has_memory_access;		// boolean: the instruction accesses memory
	uint64_t speculation_nesting;	// Speculation nesting (0 for no speculation)
	uint64_t is_pair;		// boolean: LDP/STP — memory_access2 (element 1) is also valid
	mem_access_t memory_access;	// first (or only) access
	mem_access_t memory_access2;	// second access (pair element 1); valid iff is_pair
} instr_metadata_t;

typedef struct {
    trace_cpu_state_t cpu;
    instr_metadata_t metadata;
} instr_trace_entry_t;

typedef struct {
	size_t entry_count;
	uint64_t truncated;	// 1 = the log hit max_log_index and dropped entries; the trace is incomplete
	// dynamic array of trace entries
} contract_trace_t;

/* extra_data is unused (always size 0), so every entry is a fixed 416-byte stride the Python parser
 * relies on. Lock the layout: a field/padding change must fail the build, not silently desync. */
_Static_assert(sizeof(mem_access_t) == 48, "mem_access_t layout changed");
_Static_assert(sizeof(trace_cpu_state_t) == 288, "trace_cpu_state_t layout changed");
_Static_assert(sizeof(instr_metadata_t) == 128, "instr_metadata_t layout changed");
_Static_assert(sizeof(instr_trace_entry_t) == 416, "instr_trace_entry_t stride changed");
_Static_assert(sizeof(contract_trace_t) == 16, "contract_trace_t header changed");

void init_trace_log(size_t test_size);
void* log_instr_hook(struct simulation_state* sim_state);
void* log_instr_with_speculation_nesting(struct simulation_state* sim_state, uint64_t speculation_nesting);
void destroy_trace_log();

// TODO: TMP
void* kaddr2uaddr(void*);
void* uaddr2kaddr(void*);

/* Decoded memory-access description. is_mem == 0 means the instruction is not a (real) memory
 * access and all other fields are unset; a *_register field of (uint32_t)-1 means "not applicable". */
typedef struct {
	int       is_mem;
	uintptr_t effective_address;
	uint32_t  target_register;   /* Rt */
	uint32_t  base_register;     /* Rn */
	uint32_t  index_register;    /* Rm (register-offset forms) */
	uint32_t  rt2_register;      /* Rt2 (pair forms) */
	int       is_load;
	int       is_store;
	int       is_pair;          /* LDP/STP family: a second element is accessed at EA + data_size */
	int       is_atomic;        /* LSE atomic / CAS / SWP / exclusive / acquire-release (RMW-class) */
	uint64_t  data_size;        /* access / element size in bytes (per element for pairs) */
} mem_access_info_t;

mem_access_info_t parse_memory_access_instruction(uint32_t inst, const trace_cpu_state_t *state);
#endif // SIMULATION_OUTPUT_H
