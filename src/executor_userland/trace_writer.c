#include "trace_writer.h"

static cJSON* initialize_uint64_array(const uint64_t* data, size_t count) {

	if(NULL == data) return NULL;

	cJSON* arr = cJSON_CreateArray();
	if(NULL == arr) return NULL;

	for(size_t i = 0; i < count; ++i) {
		cJSON* num = cJSON_CreateNumber(data[i]);
		if(!num) {
			cJSON_Delete(arr);
			return NULL;
		}
		if(!cJSON_AddItemToArray(arr, num)) {
			cJSON_Delete(num);
			cJSON_Delete(arr);
			return NULL;
		}
	}

	return arr;
}

static cJSON* initialize_string_array(const char** data, size_t count) {

	if(NULL == data) return NULL;

	cJSON* arr = cJSON_CreateArray();
	if(NULL == arr) return NULL;

	for(size_t i = 0; i < count; ++i) {
		cJSON* item = (data[i] == NULL) ? cJSON_CreateNull() : cJSON_CreateString(data[i]);

		if(!item) {
			cJSON_Delete(arr);
			return NULL;
		}

		if (!cJSON_AddItemToArray(arr, item)) {
			cJSON_Delete(item);
			cJSON_Delete(arr);
			return NULL;
		}
	}

	return arr;
}

cJSON* build_trace_json(const struct trace_json* trj) {
	if(NULL == trj) return NULL;

	cJSON *root = cJSON_CreateObject();
	if (NULL == root) return NULL;
	
	if (!cJSON_AddNumberToObject(root, KEY_ORDER, trj->order)) goto error;
	if (!cJSON_AddStringToObject(root, KEY_TEST_NAME, trj->test_name)) goto error;
	if (!cJSON_AddStringToObject(root, KEY_INPUT_NAME, trj->input_name)) goto error;
	if (!cJSON_AddNumberToObject(root, KEY_IID, trj->iid)) goto error;
	if (!cJSON_AddNumberToObject(root, KEY_TYPE, LOG_TYPE_TRACE)) goto error;
	
	cJSON *hwtrace_array = initialize_string_array((const char**)trj->hwtraces, HTRACE_WIDTH);
	if (!hwtrace_array) goto error;
	cJSON_AddItemToObject(root, KEY_HTRACES, hwtrace_array);
	
	cJSON *pfc_array = initialize_uint64_array(trj->pfcs, NUM_PFC);
	if (!pfc_array) goto error;
	cJSON_AddItemToObject(root, KEY_PFCS, pfc_array);

	cJSON *memory_ids_bitmap_array = initialize_string_array((const char**)trj->memory_ids_bitmap, WIDTH_MEMORY_IDS);
	if (!memory_ids_bitmap_array) goto error;
	cJSON_AddItemToObject(root, KEY_MEMORY_IDS_BITMAP, memory_ids_bitmap_array);

	return root;

error:
	cJSON_Delete(root);
	return NULL;
}

int dump_jsons(const char* filename, const cJSON* array[], size_t count) {

	if(NULL == filename || NULL == array) return -1;

	FILE* file = fopen(filename, "w");

	if(NULL == file) return -2;

	char *json_str = NULL;
	int result = 0;

	for(size_t i = 0; i < count; ++i) {

    		json_str = cJSON_PrintUnformatted(array[i]);
		if(NULL == json_str) {
			fclose(file);
			return -3;
		}

    		result = fprintf(file, "%s\n", json_str);
    		free(json_str);
		json_str = NULL;
		if(0 >= result) {
			fclose(file);
			return -4;
		}
	}

	fclose(file);
	return 0;
}

void init_trace_json(struct trace_json* trj) {
	if(NULL == trj) return;
	trj->hwtraces = (char**)trj->arrays;
	trj->pfcs = (uint64_t*)(trj->hwtraces + HTRACE_WIDTH);
	trj->memory_ids_bitmap = (char**)(trj->pfcs + NUM_PFC);
	trj->iid = -1;
	trj->input_name = NULL;
	trj->test_name = NULL;
	trj->order = (uint64_t)-1;

	for(size_t i = 0; i < HTRACE_WIDTH; ++i) {
		trj->hwtraces[i] = calloc(65, sizeof(char));
	}

	for(size_t i = 0; i < WIDTH_MEMORY_IDS; ++i) {
		trj->memory_ids_bitmap[i] = calloc(65, sizeof(char));
	}
}

void release_trace_json(struct trace_json* trj) {
	if(NULL == trj) return;

	for(size_t i = 0; i < HTRACE_WIDTH; ++i) {
		if(trj->hwtraces[i]) free(trj->hwtraces[i]);
		trj->hwtraces[i] = NULL;
	}

	for(size_t i = 0; i < WIDTH_MEMORY_IDS; ++i) {
		if(trj->memory_ids_bitmap[i]) free(trj->memory_ids_bitmap[i]);
		trj->memory_ids_bitmap[i] = NULL;
	}
}
