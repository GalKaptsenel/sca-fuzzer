#include <buffer.h>

struct buffer_t* buffer_alloc(size_t size) {
	if (size == 0) return NULL;

	struct buffer_t* b = (struct buffer_t*)malloc(sizeof(struct buffer_t));
	if (NULL == b) return NULL;

	b->addr = calloc(1, size);
	if (NULL == b->addr) {
		free(b);
		return NULL;
	}

	b->buffer_size = size;
	b->data_size = 0;
	return b;
}

void buffer_free(struct buffer_t* b) {
	if (NULL == b) return;
	if (b->addr) free(b->addr);
	free(b);
}

void buffer_init(struct buffer_t* b) {
    if (NULL == b || NULL == b->addr) return;
    memset(b->addr, 0, b->buffer_size);
}

void buffer_dump_range(const struct buffer_t* b, size_t offset, size_t length) {
	if(NULL == b || NULL == b->addr) return;
	if(offset >= b->data_size) return;

	uint8_t* buf = b->addr;
	char line[256] = { 0 };
	char ascii[17] = { 0 };

	if (offset + length > b->data_size) {
		length = b->data_size - offset;
	}

	printf("Auxilary Buffer Dump (offset=%zu, length=%zu):\n", offset, length);

	for(size_t i = 0; i < length; i += 16) {
		size_t line_len = length - i;
		if(16 < line_len) {
			line_len = 16;
		}

		size_t pos = (size_t)snprintf(line, sizeof(line), "%04zx: ", offset + i);

		for(size_t j = 0; j < line_len && pos < sizeof(line); ++j) {
			pos += snprintf(line + pos, sizeof(line) - pos, "%02x ", buf[offset + i + j]);
			ascii[j] = (buf[offset + i + j] >= 32 && buf[offset + i + j] <= 126) ? buf[offset + i + j] : '.';
		}

		for (size_t j = line_len; j < 16; ++j) {
			pos += snprintf(line + pos, sizeof(line) - pos, "   ");
			ascii[j] = ' ';
		}

		ascii[16] = '\0';
		snprintf(line + pos, sizeof(line) - pos, " |%s|", ascii);

		line[sizeof(line) - 1] = '\0';

		printf("%s\n", line);
	}
}

void buffer_dump(const struct buffer_t* b) {
	if (NULL == b) return;
	buffer_dump_range(b, 0, b->data_size);
}

