#ifndef STREAM_IPC_H
#define STREAM_IPC_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <stddef.h>
#include <inttypes.h>

#define DEFAULT_SHM_NAME "/contract_executor_stream_shm"
#define REQ_RING_SIZE  (1 << 21)  // 2 MB
#define RESP_RING_SIZE (1 << 21)  // 2 MB

struct ring {
	_Atomic uint32_t head;   // producer index
	_Atomic uint32_t tail;   // consumer index
	uint32_t size;           // must be power-of-two
	uint64_t offset_from_shm_base;
};

struct shm_region {
	struct ring req;
	struct ring resp;
};

void ring_write(void* shm_base, struct ring *r, const void *src, uint32_t len);
void ring_read(void* shm_base, struct ring *r, void *dst, uint32_t len);
struct shm_region* init_shm();
void destroy_shm(struct shm_region* shm);

// Message format protocol
struct header {
	uint32_t length;
	uint32_t type;
};

void ring_send(void* shm_base, struct ring *r, uint32_t msg_type, const uint8_t *payload, uint32_t payload_len);
void ring_recv(void* shm_base, struct ring *r, uint32_t *msg_type, uint8_t *payload, uint32_t *payload_len);

#endif
