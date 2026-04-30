#include "main.h"
#include "jit.h"
#include <linux/random.h>

// Note on registers.
// Some of the registers are reserved for a specific purpose and should never be overwritten.
// These include:
//   * X15 - hardware trace
//   * X20 - performance counter 1
//   * X21 - performance counter 2
//   * X22 - performance counter 3
//   * X29 - address of sandbox_t

#define NINDEXES (64)

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

static uint64_t sandbox_region_to_offset(enum template_regions region) {
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
	uint64_t src_off  = sandbox_region_to_offset(src);
	uint64_t dest_off = sandbox_region_to_offset(dest);
	if(0 > src_off || 0 > dest_off) return -EINVAL;
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

static void shuffle(int *arr, int n) {
	for (int i = n - 1; i > 0; i--) {
		int j = get_random_u32() % (i + 1);

		int tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
}

static void get_index_order(int* arr, int n) {
	for(int i = 0; i < n; ++i) {
		arr[i] = i;
	}
	shuffle(arr, n);
}

static int load_indexes_to_reg(jit_t* jit, uint64_t reg, uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3, uint64_t i4, uint64_t i5, uint64_t i6, uint64_t i7) {
	int index_num_size_in_bits = 8;

	uint64_t to_load = 0;
	to_load ^= i0;
	to_load <<= index_num_size_in_bits;
	to_load ^= i1;
	to_load <<= index_num_size_in_bits;
	to_load ^= i2;
	to_load <<= index_num_size_in_bits;
	to_load ^= i3;
	to_load <<= index_num_size_in_bits;
	to_load ^= i4;
	to_load <<= index_num_size_in_bits;
	to_load ^= i5;
	to_load <<= index_num_size_in_bits;
	to_load ^= i6;
	to_load <<= index_num_size_in_bits;
	to_load ^= i7;

	jit_li64(jit, reg, to_load);
	return 0;
}

static int prime(jit_t* jit, int base_reg, int tmp1, int tmp2, int tmp3, int index_reg, int reps, int assoc) {
	const int l1d_size = 64 * 1024;
	const int zero_reg = 31;


//	int indexes[NINDEXES] = { 0 };
//	get_index_order(indexes, NINDEXES);
//	for(int i = 0; i < NINDEXES; ++i) ++indexes[i]; // make sure any entry is > 0
//							// so we can shift right in granularity of 8 bits,
//							// assuming that reg is 0 iff we finished iterating.
//
//	adjust_reg(jit, base_reg, tmp1, BASE_REGION, EVICTION_REGION);
//	jit_mov64(jit, tmp3, reps); 
//
//	jit_isb(jit);
//	jit_dsb_sy(jit);
//
//	uint8_t* start = jit_get_cur(jit);
//	for(int i = 0; i < 8; ++i) {
//		load_indexes_to_reg(jit, index_reg, indexes[i*indexes_per_iteration], indexes[i*indexes_per_iteration + 1], indexes[i*indexes_per_iteration + 2], indexes[i*indexes_per_iteration + 3],
//					indexes[i*indexes_per_iteration + 4], indexes[i*indexes_per_iteration + 5], indexes[i*indexes_per_iteration + 6], indexes[i*indexes_per_iteration + 7]);
//
//		uint8_t* start_indexes_iteration = jit_get_cur(jit);
//		jit_and32(jit, tmp1, index_reg, (1 << index_size_in_bits) - 1);
//		jit_sub64(jit, tmp1, tmp1, 1); // undo the addition of 1 to the generated index
//		jit_lsl64(jit, tmp1, tmp1, 6);
//
//		jit_mov64(jit, tmp2, assoc);
//
//		uint8_t* aggregate_start = jit_get_cur(jit);
//
//		jit_ldr64shift0(jit, zero_reg, base_reg, tmp1);
//		jit_add64(jit, tmp1, tmp1, l1d_index_collisions_dist);
//
//		jit_sub64(jit, tmp2, tmp2, 1);
//		jit_cbnz64(jit, tmp2, aggregate_start);
//
//		jit_lsr64(jit, index_reg, index_reg, index_size_in_bits); // we know that index_reg == 0 iff we finished all interations
//		jit_cbnz64(jit, index_reg, start_indexes_iteration);
//	}
//	jit_sub64(jit, tmp3, tmp3, 1);
//	jit_cbnz64(jit, tmp3, start);
//
//	jit_isb(jit);
//	jit_dsb_sy(jit);
//
//	adjust_reg(jit, base_reg, tmp1, EVICTION_REGION, BASE_REGION);
//	return 0;

	adjust_reg(jit, base_reg, tmp1, BASE_REGION, EVICTION_REGION);
	jit_mov64(jit, tmp2, reps); 
	uint8_t* start = jit_get_cur(jit);
	jit_li64(jit, tmp1, l1d_size);
	uint8_t* start_prime = jit_get_cur(jit);
	jit_sub64(jit, tmp1, tmp1, 64);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_str64shift0(jit, zero_reg, base_reg, tmp1);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_cbnz64(jit, tmp1, start_prime);
	jit_sub64(jit, tmp2, tmp2, 1);
	jit_cbnz64(jit, tmp2, start);
	adjust_reg(jit, base_reg, tmp1, EVICTION_REGION, BASE_REGION);
	return 0;
}

static int flush(jit_t* jit, int base_reg, int tmp) {
	int indexes[NINDEXES] = { 0 };

	adjust_reg(jit, base_reg, tmp, BASE_REGION, MAIN_REGION);

	jit_isb(jit);
	jit_dsb_sy(jit);

	get_index_order(indexes, NINDEXES);
	for(int nindex = 0; nindex < NINDEXES; ++nindex) {
		jit_mov64(jit, tmp, indexes[nindex]);
		jit_lsl64(jit, tmp, tmp, 6);
		jit_addr64(jit, tmp, tmp, base_reg);
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_dc(jit, tmp, DC_CIVAC);
		jit_isb(jit);
		jit_dsb_sy(jit);
	}

	adjust_reg(jit, base_reg, tmp, MAIN_REGION, FAULTY_REGION);

	jit_isb(jit);
	jit_dsb_sy(jit);

	get_index_order(indexes, NINDEXES);
	for(int nindex = 0; nindex < NINDEXES; ++nindex) {
		jit_mov64(jit, tmp, indexes[nindex]);
		jit_lsl64(jit, tmp, tmp, 6);
		jit_addr64(jit, tmp, tmp, base_reg);
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_dc(jit, tmp, DC_CIVAC);
		jit_isb(jit);
		jit_dsb_sy(jit);
	}

	jit_isb(jit);
	jit_dsb_sy(jit);

	adjust_reg(jit, base_reg, tmp, FAULTY_REGION, UPPER_OVERFLOW_REGION);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_dc(jit, base_reg, DC_CIVAC);
	jit_isb(jit);
	jit_dsb_sy(jit);
	adjust_reg(jit, base_reg, tmp, UPPER_OVERFLOW_REGION, BASE_REGION);
	return 0;
}

static int probe(jit_t* jit, int base_reg, int tmp1, int tmp2, int tmp3, int index_reg, int destination_reg, int assoc) {
	const int l1d_size = 64 * 1024;
//	int l1d_index_collisions_dist = l1d_size / assoc;
	const int zero_reg = 31;
	const int aggregate_chunks = l1d_size / 4096;
	const int indexes_per_iteration = 8;
	const int index_size_in_bits = 8;

	int indexes[NINDEXES] = { 0 };
	get_index_order(indexes, NINDEXES);

	jit_isb(jit);
	jit_dsb_sy(jit);

	adjust_reg(jit, base_reg, tmp1, BASE_REGION, EVICTION_REGION);
	jit_eor64(jit, destination_reg, destination_reg, destination_reg);


	for(int i = 0; i < 8; ++i) {
		load_indexes_to_reg(jit, index_reg, indexes[i*indexes_per_iteration], indexes[i*indexes_per_iteration + 1], indexes[i*indexes_per_iteration + 2], indexes[i*indexes_per_iteration + 3],
					indexes[i*indexes_per_iteration + 4], indexes[i*indexes_per_iteration + 5], indexes[i*indexes_per_iteration + 6], indexes[i*indexes_per_iteration + 7]);

		for(int j = 0; j < 8; ++j) {
			jit_ubfx64(jit, tmp1, index_reg, j*index_size_in_bits, index_size_in_bits);
			jit_lsl64(jit, tmp1, tmp1, 6);
			
			jit_isb(jit);
			jit_dsb_sy(jit);
			jit_read64_pmu(jit, 0, tmp3);
			jit_isb(jit);
			jit_dsb_sy(jit);

			for(int k = 0; k < aggregate_chunks; ++k) {
				jit_ldr64shift0(jit, zero_reg, base_reg, tmp1);
				jit_add64(jit, tmp1, tmp1, 4096);
			}

			jit_isb(jit);
			jit_dsb_sy(jit);
			jit_read64_pmu(jit, 0, tmp2);
			jit_isb(jit);
			jit_dsb_sy(jit);
			jit_subr64(jit, tmp2, tmp2, tmp3);
			jit_cbz32(jit, tmp2, jit_get_cur(jit) + 4 * 5);
			jit_ubfx64(jit, tmp1, index_reg, j*index_size_in_bits, index_size_in_bits);
			jit_mov64(jit, tmp2, 1);
			jit_lslr64(jit, tmp2, tmp2, tmp1);
			jit_orr64(jit, destination_reg, destination_reg, tmp2);
		}
	}

	jit_isb(jit);
	jit_dsb_sy(jit);

	adjust_reg(jit, base_reg, tmp1, EVICTION_REGION, BASE_REGION);
	return 0;
}

static int reload(jit_t* jit, int base_reg, int tmp1, int tmp2, int index_reg, int destination_reg) {
	const int zero_reg = 31;
	const int index_size_in_bits = 8;
	const int indexes_per_iteration = 8;
	int indexes[NINDEXES] = { 0 };

	adjust_reg(jit, base_reg, tmp1, BASE_REGION, MAIN_REGION);
	jit_eor64(jit, destination_reg, destination_reg, destination_reg);

	jit_isb(jit);
	jit_dsb_sy(jit);

	get_index_order(indexes, NINDEXES);
	for(int i = 0; i < 8; ++i) {
		load_indexes_to_reg(jit, index_reg, indexes[i*indexes_per_iteration], indexes[i*indexes_per_iteration + 1], indexes[i*indexes_per_iteration + 2], indexes[i*indexes_per_iteration + 3],
					indexes[i*indexes_per_iteration + 4], indexes[i*indexes_per_iteration + 5], indexes[i*indexes_per_iteration + 6], indexes[i*indexes_per_iteration + 7]);

		for(int j = 0; j < indexes_per_iteration; ++j) {

			jit_ubfx64(jit, tmp1, index_reg, j*index_size_in_bits, index_size_in_bits);
			jit_lsl64(jit, tmp1, tmp1, 6);

			jit_isb(jit);
			jit_dsb_sy(jit);
			jit_read64_pmu(jit, 0, tmp2);
			jit_isb(jit);
			jit_dsb_sy(jit);
			jit_str64shift0(jit, zero_reg, base_reg, tmp1);
			jit_isb(jit);
			jit_dsb_sy(jit);
			jit_read64_pmu(jit, 0, tmp1);
			jit_isb(jit);
			jit_dsb_sy(jit);
			jit_subr64(jit, tmp1, tmp1, tmp2);
			jit_cbnz64(jit, tmp1, jit_get_cur(jit) + 4 * 5);
			jit_mov64(jit, tmp2, 1);
			jit_ubfx64(jit, tmp1, index_reg, j*index_size_in_bits, index_size_in_bits);
			jit_lslr64(jit, tmp2, tmp2, tmp1);
			jit_orr64(jit, destination_reg, destination_reg, tmp2);
		}
	}

	adjust_reg(jit, base_reg, tmp1, MAIN_REGION, FAULTY_REGION);

	jit_isb(jit);
	jit_dsb_sy(jit);

	get_index_order(indexes, NINDEXES);
	for(int nindex = 0; nindex < NINDEXES; ++nindex) {
		jit_mov64(jit, tmp1, indexes[nindex]);
		jit_lsl64(jit, tmp1, tmp1, 6);
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_read64_pmu(jit, 0, tmp2);
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_str64shift0(jit, zero_reg, base_reg, tmp1);
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_read64_pmu(jit, 0, tmp1);
		jit_isb(jit);
		jit_dsb_sy(jit);
		jit_subr64(jit, tmp1, tmp1, tmp2);
		jit_cbnz64(jit, tmp1, jit_get_cur(jit) + 4 * 3);
		jit_mov64(jit, tmp2, 1);
		jit_orr64_shift(jit, destination_reg, destination_reg, tmp2, LSL, indexes[nindex]);
	}


	jit_isb(jit);
	jit_dsb_sy(jit);

	adjust_reg(jit, base_reg, tmp1, FAULTY_REGION, UPPER_OVERFLOW_REGION);
	jit_mov64(jit, tmp1, 0);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_read64_pmu(jit, 0, tmp2);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_str64shift0(jit, zero_reg, base_reg, tmp1);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_read64_pmu(jit, 0, tmp1);
	jit_isb(jit);
	jit_dsb_sy(jit);
	jit_subr64(jit, tmp1, tmp1, tmp2);
	jit_cbnz64(jit, tmp1, jit_get_cur(jit) + 4 * 2);
	jit_mov64(jit, tmp2, 1);
	jit_orr64_shift(jit, destination_reg, destination_reg, tmp2, LSL, 0);
	adjust_reg(jit, base_reg, tmp1, UPPER_OVERFLOW_REGION, BASE_REGION);
	return 0;
}

static size_t prime_probe_method(jit_t* jit, uint32_t* tc, size_t tc_size) {
	if(NULL == jit) return -EINVAL;
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

	prime(jit, base_reg, tmp_reg1, tmp_reg2, tmp_reg3, tmp_reg4, 32, 4);

	adjust_reg(jit, base_reg, tmp_reg1, BASE_REGION, MAIN_REGION);

	probe_pfcs_start(jit, pmu1_reg, pmu2_reg, pmu3_reg);

	jit_isb(jit);
	jit_dsb_sy(jit);

	for(int i = 0; i < (tc_size / 4); ++i) {
		jit_emit(jit, tc[i]);
	}

	jit_isb(jit);
	jit_dsb_sy(jit);

	probe_pfcs_end(jit, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);

	adjust_reg(jit, base_reg, tmp_reg1, MAIN_REGION, BASE_REGION);

	probe(jit, base_reg, tmp_reg1, tmp_reg2, tmp_reg3, tmp_reg4, hw_trace_reg, 4);

	epilogue(jit, base_reg, hw_trace_reg, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);
	jit_ret(jit);
	jit_perm_rx(jit);
	return (size_t)((uintptr_t)jit_get_cur(jit) - start);
}

static size_t flush_reload_method(jit_t* jit, uint32_t* tc, size_t tc_size) {
	if(NULL == jit) return -EINVAL;
	int base_reg = 29;
	int hw_trace_reg = 15;
	int pmu1_reg = 20;
	int pmu2_reg = 21;
	int pmu3_reg = 22;
	int tmp_reg1 = 16;
	int tmp_reg2 = 17;
	int tmp_reg3 = 18;

	uintptr_t start = (uintptr_t)jit_get_cur(jit);
	jit_perm_rw(jit);
	jit_bti(jit, 1, 1);
	prologue(jit, base_reg, tmp_reg1);
	set_registers_from_input(jit, base_reg, tmp_reg1);

	flush(jit, base_reg, tmp_reg1);

	adjust_reg(jit, base_reg, tmp_reg1, BASE_REGION, MAIN_REGION);

	probe_pfcs_start(jit, pmu1_reg, pmu2_reg, pmu3_reg);

	jit_isb(jit);
	jit_dsb_sy(jit);

	for(int i = 0; i < (tc_size / 4); ++i) {
		jit_emit(jit, tc[i]);
	}

	jit_isb(jit);
	jit_dsb_sy(jit);

	probe_pfcs_end(jit, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);

	adjust_reg(jit, base_reg, tmp_reg1, MAIN_REGION, BASE_REGION);

	reload(jit, base_reg, tmp_reg1, tmp_reg2, tmp_reg3, hw_trace_reg);

	epilogue(jit, base_reg, hw_trace_reg, pmu1_reg, pmu2_reg, pmu3_reg, tmp_reg1);
	jit_ret(jit);
	jit_perm_rx(jit);
	return (size_t)((uintptr_t)jit_get_cur(jit) - start);
}

typedef size_t (*measurement_template_builder)(jit_t*, uint32_t*, size_t);
struct measurement_method {
	enum Templates type;
	measurement_template_builder builder;
	const char* name;
};

static struct measurement_method methods[] = {
	{PRIME_AND_PROBE_TEMPLATE,	prime_probe_method,	"prime and probe"},
	{FLUSH_AND_RELOAD_TEMPLATE,	flush_reload_method, 	"flush and reload"},
};


int load_jit_template(size_t tc_size) {
	if(UNSET_TEMPLATE == executor.config.measurement_template) {
		module_err("Template is not set!");
		return -5;
	}

	const uint64_t max_size_after_expantion = MAX_MEASUREMENT_CODE_SIZE;
	uint32_t* destination = (uint32_t*)executor.measurement_code_views[0];
	jit_t* jit = jit_init(max_size_after_expantion, destination);

	size_t expanded_size = 0;
	for(int i = 0; i < (sizeof(methods) / sizeof(methods[0])); ++i) {
		if(executor.config.measurement_template == methods[i].type) {
			module_info("loading %s!\n", methods[i].name);
			expanded_size = methods[i].builder(jit, (uint32_t*)executor.test_case, tc_size);
		}
	}

	jit_free(jit);
	return expanded_size;
}

