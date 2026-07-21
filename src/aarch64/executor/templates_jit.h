#ifndef ARM64_EXECUTOR_TEMPLATES_JIT_H
#define ARM64_EXECUTOR_TEMPLATES_JIT_H

enum Templates {
	UNSET_TEMPLATE,
	PRIME_AND_PROBE_TEMPLATE,
	FLUSH_AND_RELOAD_TEMPLATE,
	PRIME_AND_RELOAD_TEMPLATE,
	NUM_TEMPLATES,
};

int load_jit_template(size_t tc_size);
void refresh_tc_insert_offsets(void);
size_t current_tc_insert_offset_bytes(void);

/* JIT-harness memoization. load_jit_template() is a pure function of
 * (test_case bytes, tc_size, measurement_template); trace() calls it on every
 * repetition, and each build performs MAX_MEASUREMENT_VIEWS+1 icache flushes,
 * each an SMP-wide IPI broadcast (kick_all_cpus_sync). The sealed/NI fuzzing
 * path re-traces byte-identical harnesses n_reps times, so the rebuild+flush is
 * pure overhead on reps 2..n. invalidate_jit_cache() must be called at every
 * point that changes the harness inputs (TC load/unload, template change). */
void invalidate_jit_cache(void);

/* Runtime toggle (sysfs jit_memoize) + counters (sysfs jit_stats), for A/B. */
extern bool jit_memoize_enabled;
extern uint64_t jit_build_calls;  /* load_jit_template() invocations */
extern uint64_t jit_build_done;   /* actual rebuilds (cache misses) */
extern uint64_t jit_build_ns;     /* cumulative ns spent in the rebuild body */

#endif // ARM64_EXECUTOR_TEMPLATES_JIT_H
