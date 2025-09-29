#ifndef BUFFER_USERLAND_H
#define BUFFER_USERLAND_H

#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

struct buffer_t {
	void* addr;
	size_t data_size;
	size_t buffer_size;
};

// Allocate a new buffer
struct buffer_t* buffer_alloc(size_t size);

// Free an buffer
void buffer_free(struct buffer_t* b);

// Dump buffer contents to stdout
void buffer_dump_range(const struct buffer_t* b, size_t offset, size_t length);
void buffer_dump(const struct buffer_t* b);

#endif // BUFFER_USERLAND_H
