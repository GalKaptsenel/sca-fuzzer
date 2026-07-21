#include "main.h"
#include "jit.h"
#include <linux/ktime.h>

/* JIT-harness memoization state (see templates_jit.h). jit_cache_valid is the
 * single source of truth: it is only true while the currently-built harness
 * still matches (test_case bytes, tc_size, template). Every mutation of those
 * inputs calls invalidate_jit_cache(), so a valid cache guarantees identity. */
static bool   jit_cache_valid = false;
static size_t jit_cached_size = 0;

bool     jit_memoize_enabled = true;
uint64_t jit_build_calls = 0;
uint64_t jit_build_done  = 0;
uint64_t jit_build_ns    = 0;

void invalidate_jit_cache(void) {
	jit_cache_valid = false;
}

// Note on registers.
// Some of the registers are reserved for a specific purpose and should never be overwritten.
// These include:
//   * X15 - hardware trace
//   * X20 - performance counter 1
//   * X21 - performance counter 2
//   * X22 - performance counter 3
//   * X29 - address of sandbox_t

extern int (*set_memory_rw_fn)(unsigned long, int);

#define NINDEXES (64)

#ifndef PRIME_REPS
#define PRIME_REPS 4
#endif




/* Byte offset of the test case within the JIT'd harness, indexed by measurement
 * template. */
static size_t tc_insert_offset_bytes_by_template[NUM_TEMPLATES] = { 0 };

enum template_regions {
	BASE_REGION,
	EVICTION_REGION,
	LOWER_OVERFLOW_REGION,
	MAIN_REGION,
	FAULTY_REGION,
	UPPER_OVERFLOW_REGION,
	STORED_RSP_REGION,
	LATEST_MEASUREMENT_REGION,
};

static int64_t sandbox_region_to_offset(enum template_regions region) {
	switch(region) {
		case BASE_REGION:
		case EVICTION_REGION:
			return offsetof(sandbox_t, eviction_region);
		case LOWER_OVERFLOW_REGION:
			return offsetof(sandbox_t, lower_overflow);
		case MAIN_REGION:
			return offsetof(sandbox_t, main_region);
		case FAULTY_REGION:
			return offsetof(sandbox_t, faulty_region);
		case UPPER_OVERFLOW_REGION:
			return offsetof(sandbox_t, upper_overflow);
		case STORED_RSP_REGION:
			return offsetof(sandbox_t, stored_rsp);
		case LATEST_MEASUREMENT_REGION:
			return offsetof(sandbox_t, latest_measurement);
		default:
			return -EINVAL;
	}
}

static int adjust_reg(jit_t* jit, int reg, int tmp, enum template_regions src, enum template_regions dest) {
	int64_t src_off  = sandbox_region_to_offset(src);
	int64_t dest_off = sandbox_region_to_offset(dest);
	if (0 > src_off || 0 > dest_off) { return -EINVAL; }
	jit_li64(jit, tmp, dest_off - src_off);
	jit_addr64(jit, reg, reg, tmp);
	return 0;
}

static int set_registers_from_input(jit_t* jit, int base_reg, int tmp_reg) {
	const int sp_reg = 31;
	adjust_reg(jit, base_reg, 16, BASE_REGION, LOWER_OVERFLOW_REGION);
	jit_ldp64_post_index(jit, 0, 1, base_reg, 16);
	jit_ldp64_post_index(jit, 2, 3, base_reg, 16);
	jit_ldp64_post_index(jit, 4, 5, base_reg, 16);
	jit_ldp64_post_index(jit, 6, 7, base_reg, 16);
	jit_msr_nzcv(jit, 6); // nzcv <= x6
	jit_add64(jit, sp_reg, 7, 0);
	jit_sub64(jit, base_reg, base_reg, 4 * 16);
	adjust_reg(jit, base_reg, tmp_reg, LOWER_OVERFLOW_REGION, BASE_REGION);
	return 0;
}

static int prologue(jit_t* jit, int base_reg, int tmp_reg) {
	const int sp_reg = 31;
	jit_stp64_pre_index(jit, 16, 17, sp_reg, -16);
	jit_stp64_pre_index(jit, 18, 19, sp_reg, -16);
	jit_stp64_pre_index(jit, 20, 21, sp_reg, -16);
	jit_stp64_pre_index(jit, 22, 23, sp_reg, -16);
	jit_stp64_pre_index(jit, 24, 25, sp_reg, -16);
	jit_stp64_pre_index(jit, 26, 27, sp_reg, -16);
	jit_stp64_pre_index(jit, 28, 29, sp_reg, -16);
	jit_stp64_pre_index(jit, 30, 31, sp_reg, -16);
	jit_movr64(jit, base_reg, 0);
	adjust_reg(jit, base_reg, tmp_reg, BASE_REGION, STORED_RSP_REGION);
	jit_add64(jit, 0, sp_reg, 0);
	jit_str64(jit, 0, base_reg);
	adjust_reg(jit, base_reg, tmp_reg, STORED_RSP_REGION, BASE_REGION);
	return 0;
}

static int epilogue(jit_t* jit, int base_reg, int hw_trace_reg, int pmu1_reg, int pmu2_reg, int pmu3_reg, int tmp_reg) {
	const int sp_reg = 31;
	adjust_reg(jit, base_reg, tmp_reg, BASE_REGION, LATEST_MEASUREMENT_REGION);
	jit_stp64_post_index(jit, hw_trace_reg, pmu1_reg, base_reg, 16);
	jit_stp64_post_index(jit, pmu2_reg, pmu3_reg, base_reg, 16);
	jit_sub64(jit, base_reg, base_reg, 2 * 16);
	adjust_reg(jit, base_reg, tmp_reg, LATEST_MEASUREMENT_REGION, STORED_RSP_REGION);
	jit_ldr64(jit, 0, base_reg);
	jit_add64(jit, sp_reg, 0, 0);
	adjust_reg(jit, base_reg, 16, STORED_RSP_REGION, BASE_REGION);
	jit_ldp64_post_index(jit, 30, 31, sp_reg, 16);
	jit_ldp64_post_index(jit, 28, 29, sp_reg, 16);
	jit_ldp64_post_index(jit, 26, 27, sp_reg, 16);
	jit_ldp64_post_index(jit, 24, 25, sp_reg, 16);
	jit_ldp64_post_index(jit, 22, 23, sp_reg, 16);
	jit_ldp64_post_index(jit, 20, 21, sp_reg, 16);
	jit_ldp64_post_index(jit, 18, 19, sp_reg, 16);
	jit_ldp64_post_index(jit, 16, 17, sp_reg, 16);
	return 0;
}

static int probe_pfcs_start(jit_t* jit, int pmu1_reg, int pmu2_reg, int pmu3_reg) {
	jit_eor64(jit, pmu1_reg, pmu1_reg, pmu1_reg);
	jit_eor64(jit, pmu2_reg, pmu2_reg, pmu2_reg);
	jit_eor64(jit, pmu3_reg, pmu3_reg, pmu3_reg);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_read64_pmu(jit, 1, pmu1_reg); // should be pmu1
	jit_read64_pmu(jit, 2, pmu2_reg); // should be pmu2
	jit_read64_pmu(jit, 3, pmu3_reg); // should be pmu3
	return 0;
}

static int probe_pfcs_end(jit_t* jit, int pmu1_reg, int pmu2_reg, int pmu3_reg, int tmp) {
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_read64_pmu(jit, 1, tmp);
	jit_subr64(jit, pmu1_reg, tmp, pmu1_reg);
	jit_read64_pmu(jit, 2, tmp);
	jit_subr64(jit, pmu2_reg, tmp, pmu2_reg);
	jit_read64_pmu(jit, 3, tmp);
	jit_subr64(jit, pmu3_reg, tmp, pmu3_reg);
	return 0;
}

/* Prime: fill the whole eviction region with stores, `reps` passes. Linear sweep --
 * priming fills every set, so no anti-prefetch scrambling (unlike probe/reload). */
static int prime(jit_t* jit, int base_reg, int off_reg, int tmp_reg, int counter_reg, int reps) {
	const struct cache_geometry g = cache_geometry_on_cpu(executor.config.pinned_cpu_id, 1, CACHE_TYPE_DATA);
	if (!g.valid) {
		module_err("L1D geometry probe failed on CPU %d\n", executor.config.pinned_cpu_id);
		return -EINVAL;
	}
	const int nlines = g.size / 64;

	adjust_reg(jit, base_reg, tmp_reg, BASE_REGION, EVICTION_REGION);
	jit_isb(jit);
	jit_dsb_sy(jit);

	jit_mov64(jit, off_reg, reps);          /* outer pass counter */
	uint8_t* outer = jit_get_cur(jit);
	jit_movr64(jit, tmp_reg, base_reg);     /* reset walking pointer to eviction base each pass */
	jit_mov64(jit, counter_reg, nlines);

	uint8_t* loop = jit_get_cur(jit);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_str64(jit, counter_reg, tmp_reg);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_add64(jit, tmp_reg, tmp_reg, 64);
	jit_sub64(jit, counter_reg, counter_reg, 1);
	jit_cbnz64(jit, counter_reg, loop);

	jit_sub64(jit, off_reg, off_reg, 1);
	jit_cbnz64(jit, off_reg, outer);

	jit_isb(jit);
	jit_dsb_sy(jit);
	adjust_reg(jit, base_reg, tmp_reg, EVICTION_REGION, BASE_REGION);
	return 0;
}

/* Flush main + faulty regions + the single overflow line past faulty out of L1D. */
/* DC CIVAC every one of NINDEXES lines of the region base_reg currently points at. */
static void flush_one_region(jit_t* jit, int base_reg, int cnt, int addr, int cmp) {
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_mov64(jit, cnt, 0);
	uint8_t* loop = jit_get_cur(jit);
	jit_lsl64(jit, addr, cnt, 6);
	jit_addr64(jit, addr, base_reg, addr);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_dc(jit, addr, DC_CIVAC);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_add64(jit, cnt, cnt, 1);
	jit_sub64(jit, cmp, cnt, NINDEXES);
	jit_cbnz64(jit, cmp, loop);
}

static int flush(jit_t* jit, int base_reg, int tmp1, int tmp2, int tmp3) {
	const int cnt = tmp2, addr = tmp3;

	/* TEMP(lmfu): evict all four sandbox pages -- lower_overflow, main, faulty, upper_overflow. */
	adjust_reg(jit, base_reg, tmp1, BASE_REGION, LOWER_OVERFLOW_REGION);
	flush_one_region(jit, base_reg, cnt, addr, tmp1);
	adjust_reg(jit, base_reg, tmp1, LOWER_OVERFLOW_REGION, MAIN_REGION);
	flush_one_region(jit, base_reg, cnt, addr, tmp1);
	adjust_reg(jit, base_reg, tmp1, MAIN_REGION, FAULTY_REGION);
	flush_one_region(jit, base_reg, cnt, addr, tmp1);
	adjust_reg(jit, base_reg, tmp1, FAULTY_REGION, UPPER_OVERFLOW_REGION);
	flush_one_region(jit, base_reg, cnt, addr, tmp1);
	adjust_reg(jit, base_reg, tmp1, UPPER_OVERFLOW_REGION, BASE_REGION);

	jit_isb(jit);
	jit_dsb_sy(jit);
	return 0;
}

/* Probe the eviction region; set bit `set` iff evicted by the test (timed load slow).
 * Loop walks sets in rbit (bitrev6) order -- no constant stride for the prefetcher. */
static int probe(jit_t* jit, int base_reg, int tmp1, int tmp2, int tmp3, int index_reg, int destination_reg) {
	const int zero_reg = 31;
	const struct cache_geometry g = cache_geometry_on_cpu(executor.config.pinned_cpu_id, 1, CACHE_TYPE_DATA);
	if (!g.valid) {
		module_err("L1D geometry probe failed on CPU %d\n", executor.config.pinned_cpu_id);
		return -EINVAL;
	}
	const int aggregate_chunks = g.size / 4096;
	const int cnt = tmp3, idx = index_reg, off = tmp1, t_pre = tmp2, t_post = tmp1;

	jit_isb(jit);
	jit_dsb_sy(jit);
	adjust_reg(jit, base_reg, tmp1, BASE_REGION, EVICTION_REGION);
	jit_eor64(jit, destination_reg, destination_reg, destination_reg);

	jit_mov64(jit, cnt, 0);
	uint8_t* loop = jit_get_cur(jit);
	jit_rbit64(jit, idx, cnt);            /* idx = rbit(cnt) ... */
	jit_lsr64(jit, idx, idx, 58);         /* ... >> 58 = bitrev6(cnt) in [0,63] */
	jit_lsl64(jit, off, idx, 6);          /* byte offset of that set */

	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_read64_pmu(jit, 0, t_pre);
	jit_isb(jit);
	jit_dsb_sy(jit);
	for (int k = 0; k < aggregate_chunks; ++k) {   /* one line per 4K page mapping to this set */
		jit_ldr64shift0(jit, zero_reg, base_reg, off);
		jit_add64(jit, off, off, 4096);
	}
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_read64_pmu(jit, 0, t_post);       /* t_post aliases tmp1: off is dead after the loads */
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_subr64(jit, t_post, t_post, t_pre);           /* cycle delta */
	jit_cbz32(jit, t_post, jit_get_cur(jit) + 4 * 4); /* fast (still primed) -> skip the 3-insn bit set */
	jit_mov64(jit, t_pre, 1);             /* t_pre free after the delta */
	jit_lslr64(jit, t_pre, t_pre, idx);
	jit_orr64(jit, destination_reg, destination_reg, t_pre);
	jit_add64(jit, cnt, cnt, 1);
	jit_sub64(jit, tmp1, cnt, NINDEXES);
	jit_cbnz64(jit, tmp1, loop);

	jit_isb(jit);
	jit_dsb_sy(jit);
	adjust_reg(jit, base_reg, tmp1, EVICTION_REGION, BASE_REGION);
	return 0;
}

/* Reload main + faulty (+ overflow line); set bit `set` iff resident (timed load hit).
 * Loop walks sets in rbit (bitrev6) order -- no constant stride for the prefetcher. */
static int reload(jit_t* jit, int base_reg, int tmp1, int tmp2, int tmp3, int index_reg, int destination_reg) {
	const int zero_reg = 31;
	const int cnt = tmp3, idx = index_reg, off = tmp1, t_pre = tmp2, t_post = tmp1;
	/* TEMP(lmfu): walk all four sandbox pages lower_overflow -> main -> faulty -> upper_overflow, so
	 * main and faulty each sit mid-walk with a neighbour on both sides, but record hit bits into the
	 * htrace ONLY for main and faulty. */
	const enum template_regions region_from[4] = { BASE_REGION, LOWER_OVERFLOW_REGION, MAIN_REGION, FAULTY_REGION };
	const enum template_regions region_to[4]   = { LOWER_OVERFLOW_REGION, MAIN_REGION, FAULTY_REGION, UPPER_OVERFLOW_REGION };

	jit_eor64(jit, destination_reg, destination_reg, destination_reg);

	for (int r = 0; r < 4; ++r) {
		bool record = (MAIN_REGION == region_to[r]) || (FAULTY_REGION == region_to[r]);
		adjust_reg(jit, base_reg, tmp1, region_from[r], region_to[r]);
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_mov64(jit, cnt, 0);
		uint8_t* loop = jit_get_cur(jit);
		jit_rbit64(jit, idx, cnt);            /* idx = rbit(cnt) ... */
		jit_lsr64(jit, idx, idx, 58);         /* ... >> 58 = bitrev6(cnt) in [0,63] */
		jit_lsl64(jit, off, idx, 6);          /* byte offset of that set */
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_read64_pmu(jit, 0, t_pre);
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_ldr64shift0(jit, zero_reg, base_reg, off);
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_read64_pmu(jit, 0, t_post);       /* t_post aliases tmp1: off is dead after the load */
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_subr64(jit, t_post, t_post, t_pre);           /* cycle delta */
		if (record) {
			jit_cbnz64(jit, t_post, jit_get_cur(jit) + 4 * 4); /* slow/evicted -> skip the bit set */
			jit_mov64(jit, t_pre, 1);
			jit_lslr64(jit, t_pre, t_pre, idx);
			jit_orr64(jit, destination_reg, destination_reg, t_pre);
		}
		jit_add64(jit, cnt, cnt, 1);
		jit_sub64(jit, tmp1, cnt, NINDEXES);
		jit_cbnz64(jit, tmp1, loop);
	}

	jit_isb(jit);
	jit_dsb_sy(jit);
	adjust_reg(jit, base_reg, tmp1, region_to[3], BASE_REGION);
	return 0;
}

static size_t prime_probe_method(jit_t* jit, uint32_t* tc, size_t tc_size, size_t* tc_off_bytes) {
	if (NULL == jit) {
		return -EINVAL;
	}
	int base_reg = 29;
	int hw_trace_reg = 15;
	int pmu1_reg = 20;
	int pmu2_reg = 21;
	int pmu3_reg = 22;
	int tmp_reg1 = 16;
	int tmp_reg2 = 17;
	int tmp_reg3 = 18;
	int tmp_reg4 = 19;
	
	uintptr_t start = (uintptr_t)jit_get_cur(jit);
	jit_perm_rw(jit);
	jit_bti(jit, 1, 1);
	prologue(jit, base_reg, tmp_reg1);
	set_registers_from_input(jit, base_reg, tmp_reg1);

	prime(jit, base_reg, tmp_reg1, tmp_reg2, tmp_reg3, PRIME_REPS);

	adjust_reg(jit, base_reg, tmp_reg1, BASE_REGION, MAIN_REGION);

	probe_pfcs_start(jit, pmu1_reg, pmu2_reg, pmu3_reg);

	jit_isb(jit);
	jit_dsb_sy(jit);

	if (NULL != tc_off_bytes) {
		*tc_off_bytes = (uintptr_t)jit_get_cur(jit) - start;
	}
	for (int i = 0; i < (tc_size / 4); ++i) {
		jit_emit(jit, tc[i]);
	}

	jit_isb(jit);
	jit_dsb_sy(jit);

	probe_pfcs_end(jit, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);

	adjust_reg(jit, base_reg, tmp_reg1, MAIN_REGION, BASE_REGION);

	probe(jit, base_reg, tmp_reg1, tmp_reg2, tmp_reg3, tmp_reg4, hw_trace_reg);

	epilogue(jit, base_reg, hw_trace_reg, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);
	jit_ret(jit, 30);
	jit_perm_rx(jit);
	return (size_t)((uintptr_t)jit_get_cur(jit) - start);
}

static size_t flush_reload_method(jit_t* jit, uint32_t* tc, size_t tc_size, size_t* tc_off_bytes) {
	if (NULL == jit) {
		return -EINVAL;
	}
	int base_reg = 29;
	int hw_trace_reg = 15;
	int pmu1_reg = 20;
	int pmu2_reg = 21;
	int pmu3_reg = 22;
	int tmp_reg1 = 16;
	int tmp_reg2 = 17;
	int tmp_reg3 = 18;
	int tmp_reg4 = 19;

	uintptr_t start = (uintptr_t)jit_get_cur(jit);
	jit_perm_rw(jit);
	jit_bti(jit, 1, 1);
	prologue(jit, base_reg, tmp_reg1);
	set_registers_from_input(jit, base_reg, tmp_reg1);

	flush(jit, base_reg, tmp_reg1, tmp_reg2, tmp_reg3);

	adjust_reg(jit, base_reg, tmp_reg1, BASE_REGION, MAIN_REGION);

	probe_pfcs_start(jit, pmu1_reg, pmu2_reg, pmu3_reg);

	jit_isb(jit);
	jit_dsb_sy(jit);

	if (NULL != tc_off_bytes) {
		*tc_off_bytes = (uintptr_t)jit_get_cur(jit) - start;
	}
	for (int i = 0; i < (tc_size / 4); ++i) {
		jit_emit(jit, tc[i]);
	}

	jit_isb(jit);
	jit_dsb_sy(jit);

	probe_pfcs_end(jit, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);

	adjust_reg(jit, base_reg, tmp_reg1, MAIN_REGION, BASE_REGION);

	reload(jit, base_reg, tmp_reg1, tmp_reg2, tmp_reg3, tmp_reg4, hw_trace_reg);

	epilogue(jit, base_reg, hw_trace_reg, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);
	jit_ret(jit, 30);
	jit_perm_rx(jit);
	return (size_t)((uintptr_t)jit_get_cur(jit) - start);
}

static size_t prime_reload_method(jit_t* jit, uint32_t* tc, size_t tc_size, size_t* tc_off_bytes) {
	if (NULL == jit) {
		return -EINVAL;
	}
	int base_reg = 29;
	int hw_trace_reg = 15;
	int pmu1_reg = 20;
	int pmu2_reg = 21;
	int pmu3_reg = 22;
	int tmp_reg1 = 16;
	int tmp_reg2 = 17;
	int tmp_reg3 = 18;
	int tmp_reg4 = 19;

	uintptr_t start = (uintptr_t)jit_get_cur(jit);
	jit_perm_rw(jit);
	jit_bti(jit, 1, 1);
	prologue(jit, base_reg, tmp_reg1);
	set_registers_from_input(jit, base_reg, tmp_reg1);

	prime(jit, base_reg, tmp_reg1, tmp_reg2, tmp_reg3, PRIME_REPS);

	adjust_reg(jit, base_reg, tmp_reg1, BASE_REGION, MAIN_REGION);

	probe_pfcs_start(jit, pmu1_reg, pmu2_reg, pmu3_reg);

	jit_isb(jit);
	jit_dsb_sy(jit);

	if (NULL != tc_off_bytes) {
		*tc_off_bytes = (uintptr_t)jit_get_cur(jit) - start;
	}
	for (int i = 0; i < (tc_size / 4); ++i) {
		jit_emit(jit, tc[i]);
	}

	jit_isb(jit);
	jit_dsb_sy(jit);

	probe_pfcs_end(jit, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);

	adjust_reg(jit, base_reg, tmp_reg1, MAIN_REGION, BASE_REGION);

	reload(jit, base_reg, tmp_reg1, tmp_reg2, tmp_reg3, tmp_reg4, hw_trace_reg);

	epilogue(jit, base_reg, hw_trace_reg, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);
	jit_ret(jit, 30);
	jit_perm_rx(jit);
	return (size_t)((uintptr_t)jit_get_cur(jit) - start);
}

typedef size_t (*measurement_template_builder)(jit_t*, uint32_t*, size_t, size_t* tc_off_bytes);
struct measurement_method {
	enum Templates type;
	measurement_template_builder builder;
	const char* name;
};

static struct measurement_method methods[] = {
	{PRIME_AND_PROBE_TEMPLATE,	prime_probe_method,	"prime and probe"},
	{FLUSH_AND_RELOAD_TEMPLATE,	flush_reload_method, 	"flush and reload"},
	{PRIME_AND_RELOAD_TEMPLATE,	prime_reload_method, 	"prime and reload"},
};

static ssize_t build_measurement_code(jit_t* jit, enum Templates t, uint32_t* tc, size_t tc_size) {
	for (int i = 0; i < (sizeof(methods) / sizeof(methods[0])); ++i) {
		if (methods[i].type == t) {
			size_t* store_offset = tc_insert_offset_bytes_by_template[t] ? NULL :
				tc_insert_offset_bytes_by_template + t;
			return (ssize_t)methods[i].builder(jit, tc, tc_size, store_offset);
		}
	}
	return -EINVAL;
}


/* (Re)compute tc_insert_offset_bytes_by_template[] for every template.  Each
 * builder is run once into view[0] with an empty test case; the recorded offset
 * is independent of the test-case bytes. */
void __nocfi refresh_tc_insert_offsets(void) {
	memset(tc_insert_offset_bytes_by_template, 0,
	       sizeof(tc_insert_offset_bytes_by_template));

	for (size_t i = 0; i < ARRAY_SIZE(methods); ++i) {
		enum Templates t = methods[i].type;
		jit_t* jit = jit_init(MAX_MEASUREMENT_CODE_SIZE,
		                      (uint32_t*)executor.measurement_code_views[0]);
		methods[i].builder(jit, NULL, 0, tc_insert_offset_bytes_by_template + t);
		jit_free(jit);
	}

	/* The builders leave view[0] mapped RX (jit_perm_rx); restore RW so later
	 * writes to it (e.g. unload's memset, the next JIT build) don't fault. */
	set_memory_rw_fn((unsigned long)executor.measurement_code_views[0],
	                 MAX_MEASUREMENT_CODE_SIZE >> PAGE_SHIFT);
}

int __nocfi load_jit_template(size_t tc_size) {
	ktime_t build_start;

	if(UNSET_TEMPLATE == executor.config.measurement_template) {
		module_err("Template is not set!\n");
		return -EINVAL;
	}

	++jit_build_calls;

	/* Fast path: the harness for (test_case, tc_size, template) is already built
	 * and mapped. Reuse it and skip the rebuild + MAX_MEASUREMENT_VIEWS+1
	 * IPI-broadcasting icache flushes. jit_cache_valid implies the built harness
	 * still matches, so tc_size is guaranteed equal to what we built. Verify the
	 * first TC word is still present at the insert offset before trusting the
	 * cache: running a stale harness would, on the PAC path, execute an AUT*
	 * sealed for the wrong pointer and FPAC-fault. If it ever mismatches, scream
	 * and fall through to a full rebuild rather than run a wrong harness. */
	if (jit_memoize_enabled && jit_cache_valid && tc_size >= sizeof(uint32_t)) {
		size_t off = current_tc_insert_offset_bytes();
		uint32_t* at = (uint32_t*)((char*)executor.measurement_code_views[0] + off);
		uint32_t* tc = (uint32_t*)executor.test_case;
		if (*at == tc[0]) {
			return jit_cached_size;
		}
		module_err("jit cache self-check failed (view@0x%zx=0x%08x != tc[0]=0x%08x); rebuilding\n",
		           off, *at, tc[0]);
		jit_cache_valid = false;
	} else if (jit_memoize_enabled && jit_cache_valid) {
		/* tc_size < 4: nothing to verify, trust the invalidation contract. */
		return jit_cached_size;
	}

	build_start = ktime_get();

	const uint64_t max_size_after_expantion = MAX_MEASUREMENT_CODE_SIZE;
	uint32_t* destination = (uint32_t*)executor.measurement_code_views[0];
	jit_t* jit = jit_init(max_size_after_expantion, destination);

	ssize_t expanded_size = build_measurement_code(jit, executor.config.measurement_template,
	                                               (uint32_t*)executor.test_case, tc_size);

	bool overflowed = jit->overflow;
	jit_free(jit);

	set_memory_rw_fn((unsigned long)executor.measurement_code_views[0],
	                 MAX_MEASUREMENT_CODE_SIZE >> PAGE_SHIFT);

	if (overflowed) {
		module_err("Measurement harness overflowed the JIT buffer; refusing to run a truncated harness\n");
		return -ENOSPC;
	}

	if (0 > expanded_size) {
		module_err("Failed to build measurement code (template %d)\n",
		           executor.config.measurement_template);
		return (int)expanded_size;
	}

	/* All views alias the same physical pages, so the JIT writes (via view[0])
	 * already populate every view; jit_perm_rx() flushed view[0]'s icache.  The
	 * icache is VA-indexed, so invalidate the JIT'd range at each other view's
	 * virtual address too, enabling view rotation (invalidate_bpu_entries). */
	for (int v = 1; v < MAX_MEASUREMENT_VIEWS; ++v) {
		void *dst = executor.measurement_code_views[v];
		flush_icache_range((unsigned long)dst,
				   (unsigned long)dst + expanded_size);
	}

	/* Self-check: the first test-case word must land at the recorded offset in
	 * the JIT'd code (this is the address exported via print_code_base). */
	if (tc_size >= sizeof(uint32_t)) {
		size_t off = current_tc_insert_offset_bytes();
		uint32_t* at = (uint32_t*)((char*)executor.measurement_code_views[0] + off);
		uint32_t* tc = (uint32_t*)executor.test_case;
		if (*at != tc[0]) {
			module_err("tc_insert_offset self-check failed: view@0x%zx=0x%08x != tc[0]=0x%08x\n",
			           off, *at, tc[0]);
			return -EFAULT;
		}
	}

	jit_build_ns += ktime_to_ns(ktime_sub(ktime_get(), build_start));
	++jit_build_done;
	jit_cached_size = expanded_size;
	jit_cache_valid = true;
	return expanded_size;
}

/* Byte offset of the test case within the harness for the active template.
 * Cheap lookup; valid without a loaded TC (offsets are computed at module init). */
size_t current_tc_insert_offset_bytes(void) {
	return tc_insert_offset_bytes_by_template[executor.config.measurement_template];
}

