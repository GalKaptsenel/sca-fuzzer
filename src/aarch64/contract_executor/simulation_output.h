#ifndef SIMULATION_OUTPUT_H
#define SIMULATION_OUTPUT_H 

#include <stdint.h>
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
#define EXTRA_DATA_SIZE 128

typedef struct {
	uint64_t effective_address;		// effective memory address
	uint64_t before;	// value before access
	uint64_t after;		// value after access
	uint64_t  element_size;		// size in bytes
	uint64_t  is_write;	// 1 = write, 0 = read
} mem_access_t;

typedef struct {
	uint64_t gpr[NUM_GPRS];
	uint64_t sp;
	uint64_t pc;
	uint64_t nzcv;
	uint64_t encoding;		// Upper bits are 0 for aarch64
	mem_access_t mem_access;
	size_t extra_data_size;
	uint8_t* extra_data;		// dynamic array of size extra_data_size
} cpu_state_t;

typedef struct {
    uint64_t instr_index;		// instruction index in trace
} instr_metadata_t;

typedef struct {
    cpu_state_t cpu;
    instr_metadata_t meta;
} instr_trace_entry_t;

typedef struct {
    size_t entry_count;
    instr_trace_entry_t *entries;	// dynamic array of trace entries
} contract_trace_t;

void init_trace_log(size_t test_size);
void* log_instr_hook(struct simulation_state* sim_state);
void destroy_trace_log();
#endif // SIMULATION_OUTPUT_H
