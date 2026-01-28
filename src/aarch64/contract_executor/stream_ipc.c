#include "stream_ipc.h"

static inline int futex_wait(uint32_t* addr, uint32_t val) {
	return syscall(SYS_futex, addr, FUTEX_WAIT, val, NULL, NULL, 0);
}

static inline int futex_wake(uint32_t* addr, int n) {
	return syscall(SYS_futex, addr, FUTEX_WAKE, n, NULL, NULL, 0);
}

static inline uint32_t ring_used(struct ring* r) {
	// ring is full iff head == tail
	uint32_t head = atomic_load_explicit(&r->head, memory_order_acquire);
	uint32_t tail = atomic_load_explicit(&r->tail, memory_order_acquire);
	return head - tail;
}

static inline uint32_t ring_free(struct ring* r) {
	// ring is free iff head - tail == ring.size
	return r->size - ring_used(r);
}

void ring_write(void* shm_base, struct ring* r, const void* src, uint32_t len) {
	while (ring_free(r) < len) {
		uint32_t tail_val = atomic_load_explicit(&r->tail, memory_order_acquire);
		futex_wait((uint32_t*)&r->tail, tail_val);  // sleep until space
	}

	uint32_t head = atomic_load_explicit(&r->head, memory_order_relaxed);
	uint32_t off = head & (r->size - 1);
	uint32_t first = r->size - off;

        fprintf(stderr, "[C] Write at offset: %lx\n", r->offset_from_shm_base + off);
	if (first >= len) {
		memcpy((char*)shm_base + r->offset_from_shm_base + off, src, len);
	} else {
		memcpy((char*)shm_base + r->offset_from_shm_base + off, src, first);
		memcpy((char*)shm_base + r->offset_from_shm_base, (const uint8_t*)src + first, len - first);
	}

	atomic_store_explicit(&r->head, head + len, memory_order_release);
	futex_wake((uint32_t*)&r->head, 1);  // wake consumer
}

void ring_read(void* shm_base, struct ring* r, void* dst, uint32_t len) {
	while (ring_used(r) < len) {
		uint32_t head_val = atomic_load_explicit(&r->head, memory_order_acquire);
		futex_wait((uint32_t*)&r->head, head_val);  // sleep until enough data
	}

	uint32_t tail = atomic_load_explicit(&r->tail, memory_order_relaxed);
	uint32_t off = tail & (r->size - 1);
	uint32_t first = r->size - off;

        fprintf(stderr, "[C] Read at offset: %lx\n", r->offset_from_shm_base + off);

	if (first >= len) {
		memcpy(dst, (char*)shm_base + r->offset_from_shm_base + off, len);
	} else {
		memcpy(dst, (char*)shm_base + r->offset_from_shm_base + off, first);
		memcpy((uint8_t*)dst + first, (char*)shm_base + r->offset_from_shm_base, len - first);
	}

	atomic_store_explicit(&r->tail, tail + len, memory_order_release);
	futex_wake((uint32_t*)&r->tail, 1);  // wake producer
}

void ring_send(void* shm_base, struct ring* r, uint32_t msg_type, const uint8_t* payload, uint32_t payload_len) {
	struct header header = { 0 };

	if (sizeof(header) + payload_len > r->size) {
		fprintf(stderr, "Payload+header too big!\n");
		exit(1);
	}

	header.length = payload_len;
	header.type = msg_type;
	ring_write(shm_base, r, &header, sizeof(header));
	ring_write(shm_base, r, payload, payload_len);
}

void ring_recv(void* shm_base, struct ring* r, uint32_t* msg_type, uint8_t* payload, uint32_t* payload_len) {
	struct header header = { 0 };
	ring_read(shm_base, r, &header, sizeof(header));
	*msg_type = header.type;
	*payload_len = header.length;
	if (0 < *payload_len) {
		ring_read(shm_base, r, payload, *payload_len);
	}
}

static const size_t shm_total_size = sizeof(struct shm_region) + REQ_RING_SIZE + RESP_RING_SIZE;
struct shm_region* init_shm(const char* shm_name) {
	if(NULL == shm_name) shm_name = DEFAULT_SHM_NAME;
	int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
	if (fd < 0) { perror("shm_open"); exit(1); }

	if (ftruncate(fd, shm_total_size) < 0) { perror("ftruncate"); exit(1); }

	struct shm_region* shm = mmap(NULL, shm_total_size,
			PROT_READ | PROT_WRITE,
			MAP_SHARED, fd, 0);
	if (shm == MAP_FAILED) { perror("mmap"); exit(1); }

	shm->req.size = REQ_RING_SIZE;
	atomic_store(&shm->req.head, 0);
	atomic_store(&shm->req.tail, 0);

	shm->resp.size = RESP_RING_SIZE;
	atomic_store(&shm->resp.head, 0);
	atomic_store(&shm->resp.tail, 0);

	uint8_t* base = (uint8_t*)(shm + 1); // skip the shared memory header
	shm->req.offset_from_shm_base = base - (uint8_t*)shm;
	shm->resp.offset_from_shm_base = base + REQ_RING_SIZE - (uint8_t*)shm;

	return shm;
}

void destroy_shm(struct shm_region* shm) {
	if(shm) {
		if(-1 == munmap(shm, shm_total_size)) {
			perror("munmap");
		}
	}
}

//int main() {
//    struct shm_region* shm = init_shm(); // Already sets up rings
//
//    uint32_t* buffer = (uint32_t*)malloc(1024*1024*4);
//
//    while (1) {
//        uint32_t type = 0;
//        uint32_t len = 0;
//        ring_recv(shm, &shm->req, &type, (uint8_t*)buffer, &len);
//
//        printf("[C] Type: %d, Received: %.*s\n", type, (int)len, (char*)buffer);
//
//        // Echo back
//        ring_send(shm, &shm->resp, type, (uint8_t*)buffer, len);
//        printf("[C] Sent back\n");
//    }
//    free(buffer);
//    return 0;
//}
