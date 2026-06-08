#ifndef ARM64_EXECUTOR_TEMPLATES_JIT_H
#define ARM64_EXECUTOR_TEMPLATES_JIT_H

enum Templates {
	UNSET_TEMPLATE,
	PRIME_AND_PROBE_TEMPLATE,
	FLUSH_AND_RELOAD_TEMPLATE,
	NUM_TEMPLATES,
};

int load_jit_template(size_t tc_size);
void refresh_tc_insert_offsets(void);
size_t current_tc_insert_offset_bytes(void);

#endif // ARM64_EXECUTOR_TEMPLATES_JIT_H
