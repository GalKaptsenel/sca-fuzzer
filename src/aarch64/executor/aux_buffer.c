#include "main.h"

void aux_buffer_dump_range(const struct aux_buffer_t* auxb, size_t offset, size_t length) {
	if(NULL == auxb || NULL == auxb->addr) return;
	if(offset >= auxb->size) return;

	uint8_t* buf = auxb->addr;
	char line[256] = { 0 };
	char ascii[17] = { 0 };

	if (offset + length > auxb->size) {
		length = auxb->size - offset;
	}

	module_err("Auxilary Buffer Dump (offset=%zu, length=%zu):", offset, length);

	for(size_t i = 0; i < length; i += 16) {
		size_t line_len = length - i;
		if(16 < line_len) {
			line_len = 16;
		}

		int pos = snprintf(line, sizeof(line), "%04zx: ", offset + i);

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

		module_err("%s", line);
	}
}

void aux_buffer_dump(const struct aux_buffer_t* auxb) {
	if(NULL == auxb || NULL == auxb->addr) return;
	aux_buffer_dump_range(auxb, 0, auxb->size);
}

struct aux_buffer_t* aux_buffer_alloc(size_t size) {
	struct aux_buffer_t* auxb = kzalloc(sizeof(struct aux_buffer_t), GFP_KERNEL);
	if(NULL == auxb) {
		module_err("Unable to allocate auxiliary zone handler!");
		return NULL;
	}

	auxb->addr = kzalloc(size, GFP_KERNEL);
	if(NULL == auxb->addr) {
		 module_err("Unable to allocate auxiliary zone memory (size=%zu)!", size);
		 kfree(auxb);
		 return NULL;
	}

	auxb->size = size;

	aux_buffer_init(auxb);
	return auxb;
}

void aux_buffer_init(struct aux_buffer_t* auxb) {
	if(NULL == auxb || NULL == auxb->addr) return;
	memset(auxb->addr, 0, auxb->size);
}

void aux_buffer_free(struct aux_buffer_t* auxb) {
	if(NULL == auxb) return;
	if(auxb->addr) kfree(auxb->addr);
	kfree(auxb);
}

