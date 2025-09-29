#ifndef TRACE_WRITER_H
#define TRACE_WRITER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <executor_utils.h>
#include <buffer.h>
#include <cJSON.h>

#define KEY_ORDER		"order"
#define KEY_TEST_NAME		"test_name"
#define KEY_INPUT_NAME		"input_name"
#define KEY_IID			"iid"
#define KEY_TYPE		"type"
#define KEY_HTRACES		"htraces"
#define KEY_PFCS		"pfcs"
#define KEY_MEMORY_IDS_BITMAP	"memory_ids_bitmap"
#define KEY_AUX_BUFFER          "aux_buffer"
#define KEY_AUX_BUFFER_SIZE     "size"
#define KEY_AUX_BUFFER_DATA     "data_b64"

struct trace_json {
	uint64_t order;
	const char* test_name;
	const char* input_name;
	int iid;
	char** hwtraces;
	uint64_t* pfcs;
	char** memory_ids_bitmap;
	void* arrays[HTRACE_WIDTH + NUM_PFC + WIDTH_MEMORY_IDS];
	struct buffer_t* aux_buffer;
};

typedef enum {
    LOG_TYPE_UNKNOWN = 0,
    LOG_TYPE_TRACE= 1,
} log_type_t;

cJSON* build_trace_json(const struct trace_json* trj);
int dump_jsons(const char* filename, const cJSON* array[], size_t count);
void init_trace_json(struct trace_json* trj);
void release_trace_json(struct trace_json* trj);

#endif // TRACE_WRITER_H
