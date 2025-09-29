#include <trace_writer.h>

// Base64 character set
static const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Function to encode data in Base64
char* base64_encode(const unsigned char* data, size_t input_length, size_t* output_length) {
	if (NULL == data || 0 == input_length) {
		if (output_length) *output_length = 0;
		return NULL;
	}

	size_t output_len = 4 * ((input_length + 2) / 3);
	char* encoded_data = (char*)malloc(output_len + 1); // +1 for null terminator
	if (NULL == encoded_data) {
		if (output_length) *output_length = 0;
		return NULL;
	}

	*output_length = output_len;

	for (size_t i = 0, j = 0; i < input_length;) {
		uint32_t octet_a = i < input_length ? data[i++] : 0;
		uint32_t octet_b = i < input_length ? data[i++] : 0;
		uint32_t octet_c = i < input_length ? data[i++] : 0;

		uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

		encoded_data[j++] = base64_chars[(triple >> 3 * 6) & 0x3F];
		encoded_data[j++] = base64_chars[(triple >> 2 * 6) & 0x3F];
		encoded_data[j++] = base64_chars[(triple >> 1 * 6) & 0x3F];
		encoded_data[j++] = base64_chars[(triple >> 0 * 6) & 0x3F];
	}

	// Handle padding
	int padding = (3 - input_length % 3) % 3;

	for (int k = 0; k < padding; k++) {
		encoded_data[*output_length - 1 - k] = '=';
	}

	encoded_data[*output_length] = '\0'; // Null-terminate the string

	return encoded_data;
}

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

	char* b64 = NULL;
	cJSON* aux_obj = NULL;
	if (trj->aux_buffer && trj->aux_buffer->addr && trj->aux_buffer->data_size > 0) {
		size_t b64_size = 0;
		b64 = base64_encode(trj->aux_buffer->addr, trj->aux_buffer->data_size, &b64_size);
		if(!b64) goto error;

		aux_obj = cJSON_CreateObject();
		if(!aux_obj) goto aux_buffer_error;


		if (!cJSON_AddNumberToObject(aux_obj, KEY_AUX_BUFFER_SIZE, trj->aux_buffer->data_size)) goto aux_object_error;
		if (!cJSON_AddStringToObject(aux_obj, KEY_AUX_BUFFER_DATA, b64)) goto aux_object_error;
		free(b64);
		b64 = NULL;
		cJSON_AddItemToObject(root, KEY_AUX_BUFFER, aux_obj);
	}

	return root;

aux_object_error:
	cJSON_Delete(aux_obj);
aux_buffer_error:
	free(b64);
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

	trj->aux_buffer = NULL;
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

	if(trj->aux_buffer) {
		buffer_free(trj->aux_buffer);
		trj->aux_buffer = NULL;
	}
}
