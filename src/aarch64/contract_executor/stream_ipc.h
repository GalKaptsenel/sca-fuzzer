#ifndef STREAM_IPC_H
#define STREAM_IPC_H

#include <stdint.h>

/* Message framing header: every IPC message on the stdin/stdout pipe is
 * preceded by this 8-byte header (length = payload bytes, type = message type).
 * All fields little-endian. */
struct header {
	uint32_t length;
	uint32_t type;
};

#endif // STREAM_IPC_H
