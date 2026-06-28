/*
 * Parser/validator for the /dev/executor input wire format. The authoritative
 * structure and section contract live in userapi/executor_input_format.h.
 */
#include "userapi/executor_input_format.h"

int revisor_input_header_valid(const void *input_init, uint64_t total_len) {
	const struct revisor_input_header *h;
	const struct revisor_input_section *tab;

	if (sizeof(struct revisor_input_header) > total_len) {
		return 0;
	}
	h = (const struct revisor_input_header *)input_init;
	if (REVISOR_INPUT_MAGIC != h->magic || REVISOR_INPUT_VERSION != h->version) {
		return 0;
	}
	if (total_len != h->total_len) {
		return 0;
	}
	if (REVISOR_INPUT_MAX_SECTIONS < h->n_sections) {
		return 0;
	}
	if (REVISOR_INPUT_HEADER_LEN(h->n_sections) != h->header_len) {
		return 0;
	}
	if (total_len < h->header_len) {
		return 0;
	}

	tab = (const struct revisor_input_section *)((const char *)input_init +
	      sizeof(struct revisor_input_header));
	for (uint64_t i = 0; i < h->n_sections; ++i) {
		uint64_t off = tab[i].offset;
		uint64_t len = tab[i].length;
		uint64_t end = off + len;
		if (off < h->header_len || total_len < off) {
			return 0;
		}
		if (end < off || total_len < end) {   /* overflow or out of bounds */
			return 0;
		}
	}
	return 1;
}

const struct revisor_input_section *revisor_input_find_section(const void *input_init,
                                                               uint64_t type) {
	const struct revisor_input_header *h = (const struct revisor_input_header *)input_init;
	const struct revisor_input_section *tab =
		(const struct revisor_input_section *)((const char *)input_init +
		sizeof(struct revisor_input_header));

	for (uint64_t i = 0; i < h->n_sections; ++i) {
		if (type == tab[i].type) {
			return &tab[i];
		}
	}
	return NULL;
}
