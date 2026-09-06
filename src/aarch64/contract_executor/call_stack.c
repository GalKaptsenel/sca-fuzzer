#include "call_stack.h"
#include <string.h>

/* Depth cap. The architectural call graph is an acyclic forward DAG, so its depth is bounded by the
 * function count; speculative paths add only a bounded window's worth on top. This is generous; an
 * overflow means a genuinely unexpected shape, so we fail loudly rather than silently wrap. */
#define CALL_STACK_MAX 256

struct call_stack {
	uintptr_t entries[CALL_STACK_MAX];
	size_t    top;
};

static struct call_stack g_call_stack;

void call_stack_reset(void) {
	g_call_stack.top = 0;
}

void call_stack_push(uintptr_t return_addr) {
	if (g_call_stack.top >= CALL_STACK_MAX) {
		__builtin_trap();   /* call depth exceeded CALL_STACK_MAX */
	}
	g_call_stack.entries[g_call_stack.top++] = return_addr;
}

int call_stack_pop(uintptr_t *out) {
	if (0 == g_call_stack.top) {
		return 0;
	}
	*out = g_call_stack.entries[--g_call_stack.top];
	return 1;
}

/* A snapshot copies only the live prefix (top + used entries), so a checkpoint stays small. */
size_t call_stack_snapshot_bytes(void) {
	return sizeof(size_t) + g_call_stack.top * sizeof(uintptr_t);
}

void call_stack_snapshot(void *buf) {
	unsigned char *p = (unsigned char *)buf;
	memcpy(p, &g_call_stack.top, sizeof(size_t));
	memcpy(p + sizeof(size_t), g_call_stack.entries, g_call_stack.top * sizeof(uintptr_t));
}

void call_stack_restore(const void *buf) {
	const unsigned char *p = (const unsigned char *)buf;
	size_t top;
	memcpy(&top, p, sizeof(size_t));
	if (top > CALL_STACK_MAX) {
		__builtin_trap();   /* corrupt snapshot */
	}
	g_call_stack.top = top;
	memcpy(g_call_stack.entries, p + sizeof(size_t), top * sizeof(uintptr_t));
}
