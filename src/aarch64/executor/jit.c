#define _GNU_SOURCE

#include "jit.h"
#include "main.h"
#include <linux/kernel.h>
#include <linux/printk.h>
#include <linux/set_memory.h>
#include <linux/smp.h>
#include <asm/cacheflush.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>

extern int (*set_memory_rw_fn)(unsigned long, int);
extern int (*set_memory_ro_fn)(unsigned long, int);
extern int (*set_memory_x_fn)(unsigned long, int);
extern int (*set_memory_nx_fn)(unsigned long, int);

#define JIT_ASSERT(cond) \
    do { \
        if (unlikely(!(cond))) { \
		module_err("JIT ASSERT FAIL: %s:%d: %s\n", \
				__FILE__, __LINE__, #cond); \
		WARN_ON(!(cond));\
        } \
    } while (0)

int counter = 0;
static inline void emit(jit_t* jit, uint32_t insn) {
//	module_err("\t%d:\t[%px]\t=\t%px", counter++, jit->cur, (char*)(insn));
	JIT_ASSERT(((size_t)jit->cur & 3) == 0);
	JIT_ASSERT(jit->cur + 4 <= jit->base + jit->size);
	*(uint32_t*)jit->cur = insn;
	jit->cur += 4;
}


void jit_emit(jit_t* jit, uint32_t insn) {
//	module_err("jit_emit: %px", insn);
	emit(jit, insn);
}

static jit_t instance = { 0 };
jit_t* jit_init(size_t size, uint32_t* buffer) {
//	module_err("jit_init");

	counter = 0;

	instance.base = (uint8_t*)buffer;
	instance.cur  = instance.base;
	instance.size = size;
	return &instance;
}

void jit_free(jit_t* jit) {
//	module_err("jit_free");
	jit->base = NULL;
	jit->cur  = NULL;
	jit->size = 0;
	counter = 0;
}

uint8_t* jit_get_cur(jit_t* jit) {
//	module_err("jit_get_cur");
	return jit->cur;
}

void jit_set_cur(jit_t* jit, uint8_t* ptr) {
//	module_err("jit_set_cur");
	JIT_ASSERT(ptr >= jit->base);
	JIT_ASSERT(ptr <= jit->base + jit->size);
	JIT_ASSERT(((uintptr_t)ptr & 3) == 0);
	jit->cur = ptr;
}

uint8_t* jit_next_with_mask(jit_t* jit,
		uintptr_t value_mask,
		uintptr_t bit_mask,
		bool forward) {
//	module_err("jit_next_with_mask");
	uintptr_t current_loc = (uintptr_t)jit_get_cur(jit);
	uintptr_t candidate = (current_loc & ~bit_mask) | (value_mask & bit_mask);
	JIT_ASSERT((candidate & 3) == 0);

	if (forward) {
		if (candidate <= current_loc) {
			uintptr_t free_mask = ~bit_mask;
		
		        // add smallest increment in free bits
			uintptr_t inc = (~candidate) & free_mask;
			inc &= -inc;  // isolate lowest free zero bit
		
			candidate += inc;
		
		        // clear lower free bits to get lowest increament
		        candidate &= ~(inc - 1);
		
			candidate = (candidate & ~bit_mask) | (value_mask & bit_mask);
		}

		JIT_ASSERT((uintptr_t)jit->cur < candidate);
		JIT_ASSERT(0 == ((uintptr_t)jit->cur & 3));

		uint32_t* p = (uint32_t*)jit->cur;
		uint32_t* end = (uint32_t*)candidate;

		while (p < end) {
			*(p++) = 0xd503201f;
		}
	
		jit->cur = (uint8_t*)candidate;

	} else {
		if (candidate >= current_loc) {
			uintptr_t free_mask = ~bit_mask;
	
			// find lowest free one bit
			uintptr_t dec = candidate & free_mask;
			dec &= -dec;  // isolate lowest free one bit
	
			candidate -= dec;
	
			// set lower free bits to 1 to get the smallest decreament
			candidate |= (dec - 1) & free_mask;
	
			candidate = (candidate & ~bit_mask) | (value_mask & bit_mask);
		}

		JIT_ASSERT((uintptr_t)jit->cur > candidate);
		JIT_ASSERT(0 == ((uintptr_t)jit->cur & 3));

		uint32_t* p = (uint32_t*)jit->cur;
		uint32_t* end = (uint32_t*)candidate;

		while (p > end) {
			*(p--) = 0xd503201f;
		}
	
		jit->cur = (uint8_t*)candidate;
	}

	JIT_ASSERT(0 == ((uintptr_t)jit->cur & 3));
	JIT_ASSERT((value_mask & bit_mask) == ((uintptr_t)jit->cur & bit_mask));

	return jit->cur;
}

uint8_t* jit_align(jit_t* jit, size_t alignment, ssize_t offset) {
//	module_err("jit_jit_align");
	uintptr_t addr = (uintptr_t)jit->cur;
	ssize_t mod = addr % alignment;

    	if (offset >= 0) {
		if (mod <= offset) {
			addr += mod - offset;
		}
		else {
			addr += alignment - mod + offset;
		}
	} else {
		if (mod - offset <= (ssize_t)alignment) {
			addr += alignment - (mod - offset);
		}
		else {
			addr += alignment + alignment + offset - mod;
		}
	}

	uint8_t *new_cur = (uint8_t *)addr;
	JIT_ASSERT((size_t)jit->cur <= (size_t)new_cur);
	JIT_ASSERT(0 == ((size_t)jit->cur & 3));
	JIT_ASSERT(0 == ((size_t)new_cur & 3));

	uint32_t* p = (uint32_t *)jit->cur;
	uint32_t* end = (uint32_t*)new_cur;
	while (p < end) {
		*p++ = 0xd503201f;
	}

	jit->cur = new_cur;

	JIT_ASSERT((((size_t)jit->cur - offset) % alignment) == 0);
	return jit->cur;
}

int jit_perm_rw(jit_t* jit) {
//	module_err("jit_perm_rw");
	if(NULL == jit || NULL == jit->base) return -EINVAL;
	int npages = jit->size >> PAGE_SHIFT;
	if (set_memory_nx_fn((unsigned long)jit->base, npages)) {
		return -EFAULT;
	}
	if (set_memory_rw_fn((unsigned long)jit->base, npages)) {
		return -EFAULT;
	}
	flush_icache_range((unsigned long)jit->base,
			(unsigned long)jit->base + jit->size);
	return 0;
}

int jit_perm_rx(jit_t* jit) {
//	module_err("jit_perm_rx");
	if(NULL == jit || NULL == jit->base) return -EINVAL;
	int npages = jit->size >> PAGE_SHIFT;

	smp_mb();

	if (set_memory_ro_fn((unsigned long)jit->base, npages)) {
		return -EFAULT;
	}

	if (set_memory_x_fn((unsigned long)jit->base, npages)) {
		return -EFAULT;
	}

	flush_icache_range((unsigned long)jit->base,
			(unsigned long)jit->base + jit->size);
	return 0;
}

void jit_cfp_rctx(jit_t* jit, int rt) {
//	module_err("cfp rctx x%d",rt);
	uint32_t insn = 0xd50b7380;
	JIT_ASSERT(0 <= rt && rt <= 31);
	insn |= rt;
	emit(jit, insn);
}

void jit_isb(jit_t* jit) {
//	module_err("isb");
	emit(jit, 0xd5033fdf);
}

void jit_dsb_sy(jit_t* jit) {
//	module_err("dsb sy");
	emit(jit, 0xd5033f9f);
}

void jit_nop(jit_t* jit) {
//	module_err("nop");
	emit(jit, 0xd503201f);
}

void jit_svc0(jit_t* jit) {
//	module_err("svc #0");
	emit(jit, 0xd4000001);
}

void jit_ret(jit_t* jit) {
//	module_err("ret");
	emit(jit, 0xd65f03c0);
}
void jit_bti(jit_t* jit, bool calls, bool branches) {
//	module_err("bti");
	uint32_t insn = 0xd503241f;
	uint32_t targets = 0;
	targets |= 2 * (uint32_t)branches | (uint32_t)calls;
	insn |= (targets << 5);
	emit(jit, insn);
}

typedef struct {
	const char* symbol;
	uint8_t op1;
	uint8_t crm;
	uint8_t op2;
} dc_encoding_t;

static const dc_encoding_t dc_op_encoding[] = {
    [DC_IVAC]    = {"DC_IVAC", 0b000, 0b0110, 0b001},
    [DC_ISW]     = {"DC_ISW", 0b000, 0b0110, 0b010},

    [DC_IGVAC]   = {"DC_IGVAC", 0b000, 0b0110, 0b011},
    [DC_IGSW]    = {"DC_IGSW", 0b000, 0b0110, 0b100},
    [DC_IGDVAC]  = {"DC_IGDVAC", 0b000, 0b0110, 0b101},
    [DC_IGDSW]   = {"DC_IGDSW", 0b000, 0b0110, 0b110},

    [DC_CSW]     = {"DC_CSW", 0b000, 0b1010, 0b010},

    [DC_CGSW]    = {"DC_CGSW", 0b000, 0b1010, 0b100},
    [DC_CGDSW]   = {"DC_CGDSW", 0b000, 0b1010, 0b110},

    [DC_CISW]    = {"DC_CISW", 0b000, 0b1110, 0b010},

    [DC_CIGSW]   = {"DC_CIGSW", 0b000, 0b1110, 0b100},
    [DC_CIGDSW]  = {"DC_CIGDSW", 0b000, 0b1110, 0b110},

    [DC_ZVA]     = {"DC_ZVA", 0b011, 0b0100, 0b001},

    [DC_GVA]     = {"DC_GVA", 0b011, 0b0100, 0b011},
    [DC_GZVA]    = {"DC_GZVA", 0b011, 0b0100, 0b100},

    [DC_CVAC]    = {"DC_CVAC", 0b011, 0b1010, 0b001},

    [DC_CGVAC]   = {"DC_CGVAC", 0b011, 0b1010, 0b011},
    [DC_CGDVAC]  = {"DC_CGDVAC", 0b011, 0b1010, 0b101},

    [DC_CVAU]    = {"DC_CVAU", 0b011, 0b1011, 0b001},

    [DC_CVAP]    = {"DC_CVAP", 0b011, 0b1100, 0b001},

    [DC_CGVAP]   = {"DC_CGVAP", 0b011, 0b1100, 0b011},
    [DC_CGDVAP]  = {"DC_CGDVAP", 0b011, 0b1100, 0b101},

    [DC_CVADP]   = {"DC_CVADP", 0b011, 0b1101, 0b001},

    [DC_CGVADP]  = {"DC_CGVADP", 0b011, 0b1101, 0b011},
    [DC_CGDVADP] = {"DC_CGDVADP", 0b011, 0b1101, 0b101},

    [DC_CIVAC]   = {"DC_CIVAC", 0b011, 0b1110, 0b001},

    [DC_CIGVAC]  = {"DC_CIGVAC", 0b011, 0b1110, 0b011},
    [DC_CIGDVAC] = {"DC_CIGDVAC", 0b011, 0b1110, 0b101},

    [DC_CIPAE]   = {"DC_CIPAE", 0b100, 0b1110, 0b000},
    [DC_CIGDPAE] = {"DC_CIGDPAE", 0b100, 0b1110, 0b111},

    [DC_CIPAPA]  = {"DC_CIPAPA", 0b110, 0b1110, 0b001},
    [DC_CIGDPAPA]= {"DC_CIGDPAPA", 0b110, 0b1110, 0b101},
};

void jit_sys(jit_t* jit, int op1, int CRn, int CRm, int op2, int rt) {
//	module_err("sys #%d, %d, %d, #%d, x%d", op1, CRn, CRm, op2, rt);
	uint32_t insn = 0xd5080000u;
	JIT_ASSERT(0 <= rt && rt <= 31);
	JIT_ASSERT(0 <= op1 && op1 <= 7);
	JIT_ASSERT(0 <= CRn && CRn <= 15);
	JIT_ASSERT(0 <= CRm && CRm <= 15);
	JIT_ASSERT(0 <= op2 && op2 <= 7);
	insn |= rt;
	insn |= op2 << 5;
	insn |= CRm << 8;
	insn |= CRn << 12;
	insn |= op1 << 16;
	emit(jit, insn);
}

void jit_dc(jit_t* jit, int rt, dc_op_t dc_op) {
//	module_err("dv %s, x%d", dc_op_encoding[dc_op].symbol, rt);
	JIT_ASSERT(0 <= rt && rt <= 31);
	int CRn = 7;
	jit_sys(jit, dc_op_encoding[dc_op].op1, CRn, dc_op_encoding[dc_op].crm, dc_op_encoding[dc_op].op2, rt);
}

void jit_ubfx64(jit_t* jit, int rd, int rn, int lsb, int width) {
//	module_err("ubfx x%d, x%d, #%d, #%d", rd, rn, lsb, width);
	uint32_t insn = 0xd3400000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= lsb && lsb <= 63);
	JIT_ASSERT(1 <= width && width <= 64 - lsb);
	int immr = lsb;
	int imms = lsb + width - 1;
	insn |= rd;
	insn |= rn << 5;
	insn |= imms << 10;
	insn |= immr << 16;
	emit(jit, insn);
}

void jit_add64(jit_t* jit, int rd, int rn, int imm) {
//	module_err("add x%d, x%d, #%d", rd, rn, imm);
	uint32_t insn = 0x91000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT((0 <= imm && imm <= ((1 << 12) - 1)) || (((imm & 0xfff) == 0) && ((0 <= (imm >> 12)) && (imm >> 12) < (1 << 12))) );
	insn |= rd;
	insn |= rn << 5;
	if(imm >= (1 << 12)) {
		insn |= 1 << 22;
		imm >>= 12;
	}

	insn |= imm << 10;
//	module_err("add64: %px", insn);
	emit(jit, insn);
}

void jit_sub64(jit_t* jit, int rd, int rn, int imm) {
//	module_err("sub x%d, x%d, #%d", rd, rn, imm);
	uint32_t insn = 0xd1000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= imm && imm <= ((1 << 12) - 1));
	insn |= rd;
	insn |= rn << 5;
	insn |= imm << 10;
//	module_err("sub64: %px", insn);
	emit(jit, insn);
}

void jit_addr64(jit_t* jit, int rd, int rn, int rm) {
//	module_err("add x%d, x%d, x%d", rd, rn, rm);
	uint32_t insn = 0x8b000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rd;
	insn |= rn << 5;
	insn |= rm << 16;
	emit(jit, insn);
}

void jit_subs64(jit_t* jit, int rd, int rn, int imm) {
//	module_err("subs x%d, x%d, #%d", rd, rn, imm);
	uint32_t insn = 0xf1000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= imm && imm <= ((1 << 12) - 1));
	insn |= rd;
	insn |= rn << 5;
	insn |= imm << 10;
	emit(jit, insn);
}

void jit_cmp32(jit_t* jit, int rn, int imm) {
//	module_err("cmp w%d, #%d", rn, imm);
	uint32_t insn = 0x7100001f;
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= imm && imm <= ((1 << 12) - 1));
	insn |= rn << 5;
	insn |= imm << 10;
	emit(jit, insn);
}

void jit_orr64_shift(jit_t* jit, int rd, int rn, int rm, shift_type_t shift_type, int amount) {
	uint32_t insn = 0xaa000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	JIT_ASSERT(0 <= shift_type && shift_type <= 3);
	JIT_ASSERT(0 <= amount && amount <= 63);
	int shift = 0;
	const char* shift_type_symbol = NULL;
	switch(shift_type) {
		case LSL:
			shift = 0;
			shift_type_symbol = "LSL";
			break;
		case LSR:
			shift = 1;
			shift_type_symbol = "LSR";
			break;
		case ASR:
			shift = 2;
			shift_type_symbol = "ASR";
			break;
		case ROR:
			shift = 3;
			shift_type_symbol = "ROR";
			break;
	}
//	module_err("orr x%d, x%d, x%d, %s, #%d", rd, rn, rm, shift_type_symbol, amount);
	insn |= rd;
	insn |= rn << 5;
	insn |= amount << 10;
	insn |= rm << 16;
	insn |= shift << 22;
//	module_err("orr64_shift: %px", insn);
	emit(jit, insn);

}
void jit_orr64(jit_t* jit, int rd, int rn, int rm) {
//	module_err("orr x%d, x%d, x%d", rd, rn, rm);
	uint32_t insn = 0xaa000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rd;
	insn |= rn << 5;
	insn |= rm << 16;
	emit(jit, insn);
}

typedef struct {
	uint8_t immN;
	uint8_t imms;
	uint8_t immr;
	bool valid;
} encode_result_t;

static uint64_t ones(int n) {
	if (n <= 0) return 0;
	if (n >= 64) return ~0ULL;
	return (1ULL << n) - 1;
}

static uint64_t ror(uint64_t x, int r, int width) {
	r %= width;
	uint64_t mask = ones(width);
	x &= mask;
	return ((x >> r) | (x << (width - r))) & mask;
}

static int is_repeated(uint64_t x, int esize) {
	uint64_t mask = ones(esize);
	uint64_t chunk = x & mask;

	for (int i = esize; i < 64; i += esize) {
		if (((x >> i) & mask) != chunk) {
			return 0;
		}
	}
	return 1;
}

static int is_run_of_ones(uint64_t x, int width, int* rotation, int* run_len) {
	for (int r = 0; r < width; r++) {
		uint64_t y = ror(x, r, width);
		int seen_one = 0;
		int count = 0;
		for (int i = 0; i < width; i++) {
			if (y & (1ULL << i)) {
				seen_one = 1;
				count++;
			} else if (seen_one) {
				for (int j = i + 1; j < width; j++) {
					if (y & (1ULL << j)) return 0;
				}
				*rotation = r;
				*run_len = count;
				return 1;
			}
		}

		if (seen_one) {
			*rotation = r;
			*run_len = count;
			return 1;
		}
	}

	return 0;
}

static encode_result_t EncodeBitMask(uint64_t imm) {
	encode_result_t res = {0};
	res.valid = false;

	if (imm == 0 || imm == ~0ULL) {
		return res;
	}


	for (int len = 1; len <= 6; len++) {
		int esize = 1 << len;

		if (!is_repeated(imm, esize)) {
			continue;
		}

		uint64_t element = imm & ones(esize);

		int rotation = 0;
		int run_len = 0;
		if (!is_run_of_ones(element, esize, &rotation, &run_len)) {
			continue;
		}

		uint64_t levels = ones(len);

		uint64_t s = run_len - 1;
		uint64_t r = rotation;

		uint8_t imms = s | (~levels & 0x3F);
		uint8_t immr = r & 0x3F;

		uint8_t immN = (len == 6);

		res.immN = immN;
		res.imms = imms;
		res.immr = immr;
		res.valid = true;
		return res;
	}

	return res;
}

void jit_and64(jit_t* jit, int rd, int rn, int imm) {
//	module_err("and x%d, x%d, #%d", rd, rn, imm);
	uint32_t insn = 0x92000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= imm && imm <= (1 << 13) - 1);
	insn |= rd;
	insn |= rn << 5;
	encode_result_t res = EncodeBitMask(imm);
	JIT_ASSERT(res.valid);
	insn |= res.imms << 10;
	insn |= res.immr << 16;
	insn |=  res.immN << 22;
//	module_err("and64: %px", insn);
	emit(jit, insn);
}



void jit_and32(jit_t* jit, int rd, int rn, int imm) {
	module_err("and w%d, w%d, #%d", rd, rn, imm);
	uint32_t insn = 0x12000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= imm && imm <= (1 << 12) - 1);
	insn |= rd;
	insn |= rn << 5;
	encode_result_t res = EncodeBitMask(imm);
	JIT_ASSERT(res.valid);
	insn |= res.imms << 10;
	insn |= res.immr << 16;
//	module_err("and32: %px", insn);
	emit(jit, insn);
}
void jit_andr64(jit_t* jit, int rd, int rn, int rm) {
//	module_err("and x%d, x%d, x%d", rd, rn, rm);
	uint32_t insn = 0x8a000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rd;
	insn |= rn << 5;
	insn |= rm << 16;
//	module_err("andr64: %px", insn);
	emit(jit, insn);
}

void jit_eor64(jit_t* jit, int rd, int rn, int rm) {
//	module_err("eor x%d, x%d, x%d", rd, rn, rm);
	uint32_t insn = 0xca000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rd;
	insn |= rn << 5;
	insn |= rm << 16;
	emit(jit, insn);
}

void jit_cbnz64(jit_t* jit, int rt, uint8_t* target) {
	uint32_t insn = 0xb5000000 | rt;
	JIT_ASSERT(0 <= rt && rt <= 31);
	ssize_t imm = (ssize_t)target - (ssize_t)jit->cur;
//	module_err("cbnz x%d, %l [%px]", rt, imm, target);
	JIT_ASSERT(-(1<<20) <= imm && imm < (1<<20));
	JIT_ASSERT(0 == (imm & 3));
	insn |= ((imm >> 2) & 0x7ffff) << 5;
//	module_err("cbnz64: %px", insn);
	emit(jit, insn);
}

void jit_cbz32(jit_t* jit, int rt, uint8_t* target) {
	uint32_t insn = 0x34000000 | rt;
	JIT_ASSERT(0 <= rt && rt <= 31);
	ssize_t imm = (ssize_t)target - (ssize_t)jit->cur;
//	module_err("cbz w%d, %l [%px]", rt, imm, target);
	JIT_ASSERT(-(1<<20) <= imm && imm < (1<<20));
	JIT_ASSERT(0 == (imm & 3));
	insn |= ((imm >> 2) & 0x7ffff) << 5;
//	module_err("cbz32: %px", insn);
	emit(jit, insn);
}

void jit_b(jit_t* jit, uint8_t* target) {
	uint32_t insn = 0x14000000;
	ssize_t imm = (ssize_t)target - (ssize_t)jit->cur;
	JIT_ASSERT(-(1<<27) <= imm && imm < (1<<27));
	JIT_ASSERT(0 == (imm & 3));
//	module_err("b %l [%px]", imm, target);
	insn |= ((imm >> 2) & 0x3ffffff);
	emit(jit, insn);
}

// STP <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]
void jit_stp64(jit_t* jit, int rt1, int rt2, int rn, int imm) {
//	module_err("stp x%d, x%d, [x%d, #%d]", rt1, rt2, rn, imm);
	uint32_t insn = 0xa9000000;
	JIT_ASSERT(0 <= rt1 && rt1 <= 31);
	JIT_ASSERT(0 <= rt2 && rt2 <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 == (imm & 7));
	JIT_ASSERT(-512 <= imm && imm <= 504);
	insn |= rt1;
	insn |= (rn << 5);
	insn |= (rt2 << 10);
	int off = imm >> 3;
	insn |= ((off & 0x7F) << 15);
	emit(jit, insn);
}

// LDP <Xt1>, <Xt2>, [<Xn|SP>{, #<imm>}]
void jit_ldp64(jit_t* jit, int rt1, int rt2, int rn, int imm) {
//	module_err("ldp x%d, x%d, [x%d, #%d]", rt1, rt2, rn, imm);
	uint32_t insn = 0xa9400000;
	JIT_ASSERT(0 <= rt1 && rt1 <= 31);
	JIT_ASSERT(0 <= rt2 && rt2 <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 == (imm & 7));
	JIT_ASSERT(-512 <= imm && imm <= 504);
	insn |= rt1;
	insn |= (rn << 5);
	insn |= (rt2 << 10);
	int off = imm >> 3;
	insn |= ((off & 0x7F) << 15);
	emit(jit, insn);
}

// LDP <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
void jit_ldp64_post_index(jit_t* jit, int rt1, int rt2, int rn, int imm) {
//	module_err("ldp x%d, x%d, [x%d], #%d", rt1, rt2, rn, imm);
	uint32_t insn = 0xa8c00000;
	JIT_ASSERT(0 <= rt1 && rt1 <= 31);
	JIT_ASSERT(0 <= rt2 && rt2 <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 == (imm & 7));
	JIT_ASSERT(-512 <= imm && imm <= 504);
	insn |= rt1;
	insn |= (rn << 5);
	insn |= (rt2 << 10);
	int off = imm >> 3;
	insn |= ((off & 0x7F) << 15);
	emit(jit, insn);
}


// LDR <Wt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
void jit_ldr32shift2(jit_t* jit, int rt, int rn, int rm) {
//	module_err("ldr w%d, [x%d, x%d, LSL #2]", rt, rn, rm);
	uint32_t insn = 0xb8605800;
	JIT_ASSERT(0 <= rt && rt <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rt;
	insn |= (rn << 5);
	insn |= (rm << 16);
	emit(jit, insn);
}


void jit_subr64(jit_t* jit, int rd, int rn, int rm) {
//	module_err("sub x%d, x%d, x%d", rd, rn, rm);
	uint32_t insn = 0xcb000000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rd;
	insn |= (rn << 5);
	insn |= (rm << 16);
//	module_err("subr64: %px", insn);
	emit(jit, insn);
}

// LDR <Xt>, [<Xn|SP>]
void jit_ldr64(jit_t* jit, int rt, int rn) {
//	module_err("ldr x%d, [x%d]", rt, rn);
	uint32_t insn = 0xf8400400;
	JIT_ASSERT(0 <= rt && rt <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	insn |= rt;
	insn |= (rn << 5);
	emit(jit, insn);
}

// LDR <Xt>, [<Xn|SP>, <Xm>]
void jit_ldr64shift0(jit_t* jit, int rt, int rn, int rm) {
//	module_err("ldr x%d, [x%d, x%d]", rt, rn, rm);
	uint32_t insn = 0xf8606800;
	JIT_ASSERT(0 <= rt && rt <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rt;
	insn |= (rn << 5);
	insn |= (rm << 16);
//	module_err("ldr64shift0: %px", insn);
	emit(jit, insn);
}

// LDR <Xt>, [<Xn|SP>, (<Wm>|<Xm>){, <extend> {<amount>}}]
void jit_ldr64shift3(jit_t* jit, int rt, int rn, int rm) {
//	module_err("ldr x%d, [x%d, x%d, LSL #3]", rt, rn, rm);
	uint32_t insn = 0xf8607800;
	JIT_ASSERT(0 <= rt && rt <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rt;
	insn |= (rn << 5);
	insn |= (rm << 16);
	emit(jit, insn);
}

// STR <Xt>, [<Xn|SP>, <Xm>]
void jit_str64shift0(jit_t* jit, int rt, int rn, int rm) {
//	module_err("str x%d, [x%d, x%d]", rt, rn, rm);
	uint32_t insn = 0xf8206800;
	JIT_ASSERT(0 <= rt && rt <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rt;
	insn |= (rn << 5);
	insn |= (rm << 16);
	emit(jit, insn);

}
void jit_movr64(jit_t* jit, int rd, int rm) {
//	module_err("mov x%d, x%d", rd, rm);
	uint32_t insn = 0xaa0003e0;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rd;
	insn |= rm << 16;
//	module_err("movr64: %px", insn);
	emit(jit, insn);
}

// STP <Xt1>, <Xt2>, [<Xn|SP>, #<imm>]!
void jit_stp64_pre_index(jit_t* jit, int rt1, int rt2, int rn, int imm) {
//	module_err("stp x%d, x%d, [x%d, #%d]!", rt1, rt2, rn, imm);
	uint32_t insn = 0xa9800000;
	JIT_ASSERT(0 <= rt1 && rt1 <= 31);
	JIT_ASSERT(0 <= rt2 && rt2 <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 == (imm & 7));
	JIT_ASSERT(-512 <= imm && imm <= 504);
	insn |= rt1;
	insn |= (rn << 5);
	insn |= (rt2 << 10);
	int off = imm >> 3;
	insn |= ((off & 0x7F) << 15);
	emit(jit, insn);
}

// STP <Xt1>, <Xt2>, [<Xn|SP>], #<imm>
void jit_stp64_post_index(jit_t* jit, int rt1, int rt2, int rn, int imm) {
//	module_err("stp x%d, x%d, [x%d], #%d", rt1, rt2, rn, imm);
	uint32_t insn = 0xa8800000;
	JIT_ASSERT(0 <= rt1 && rt1 <= 31);
	JIT_ASSERT(0 <= rt2 && rt2 <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 == (imm & 7));
	JIT_ASSERT(-512 <= imm && imm <= 504);
	insn |= rt1;
	insn |= (rn << 5);
	insn |= (rt2 << 10);
	int off = imm >> 3;
	insn |= ((off & 0x7F) << 15);
	emit(jit, insn);
}

// STR <Xt>, [<Xn|SP>]
void jit_str64(jit_t* jit, int rt, int rn) {
//	module_err("str x%d, [x%d]", rt, rn);
	uint32_t insn = 0xf9000000;
	JIT_ASSERT(0 <= rt && rt <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	insn |= rt;
	insn |= (rn << 5);
	emit(jit, insn);
}

void jit_smc(jit_t* jit, int imm) {
//	module_err("smc #%d", imm);
	uint32_t insn = 0xd4000003;
	JIT_ASSERT(0 <= imm && imm <= ((1 << 16) - 1));
	insn |= imm << 5;
	emit(jit, insn);
}

void jit_mov64(jit_t* jit, int rd, int imm) {
//	module_err("mov x%d, #%d", rd, imm);
	uint32_t insn = 0xd2800000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= imm && imm <= ((1 << 16) - 1));
	insn |= rd;
	insn |= imm << 5;
//	module_err("mov64: %px", insn);
	emit(jit, insn);
}

void jit_movk64(jit_t* jit, int rd, int imm, int shift) {
//	module_err("movk x%d, #%px, LSL #%d", rd, (uint8_t*)imm, shift);
	uint32_t insn = 0xf2800000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= imm && imm <= ((1 << 16) - 1));
	JIT_ASSERT(0 == (shift & 0xF));
	JIT_ASSERT(0 <= shift && shift <= 48);
	insn |= rd;
	insn |= imm << 5;
	insn |= (shift / 16) << 21;
	emit(jit, insn);
}

void jit_li64(jit_t* jit, int rd, uint64_t imm) {
//	module_err("li64 %d, #%px", rd, (uint8_t*)imm);
	jit_mov64(jit, rd, imm & 0xffff);
	jit_movk64(jit, rd, (imm >> 16) & 0xffff, 16);
	jit_movk64(jit, rd, (imm >> 32) & 0xffff, 32);
	jit_movk64(jit, rd, (imm >> 48) & 0xffff, 48);
}

const char* cond_str[] = {
	"eq", "ne", "cs", "cc", "mi", "pl", "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al", "nv"
};
// CSEL <Xd>, <Xn>, <Xm>, <cond>
void jit_csel64(jit_t* jit, int rd, int rn, int rm, int cond) {
	uint32_t insn = 0x9a800000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	JIT_ASSERT(0 <= cond && cond <= ((1 << 4) - 1));
//	module_err("csel x%d, x%d, x%d, %s", rd, rn, rm, cond_str[cond]);
	insn |= rd;
	insn |= (rn << 5);
	insn |= (rm << 16);
	insn |= (cond << 12);
	emit(jit, insn);
}

void jit_br64(jit_t* jit, int rn) {
//	module_err("br x%d", rn);
	uint32_t insn = 0xd61f0000 | (rn << 5);
	JIT_ASSERT(0 <= rn && rn <= 31);
	emit(jit, insn);
}

void jit_blr64(jit_t* jit, int rn) {
//	module_err("blr x%d", rn);
	uint32_t insn = 0xd63f0000 | (rn << 5);
	JIT_ASSERT(0 <= rn && rn <= 31);
	emit(jit, insn);
}

// LSR <Xd>, <Xn>, #<shift>
void jit_lsr64(jit_t* jit, int rd, int rn, int shift) {
//	module_err("lsr x%d, x%d, #%d", rd, rn, shift);
	uint32_t insn = 0xd340fc00;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= shift && shift <= 63);
	insn |= rd;
	insn |= (rn << 5);
	insn |= (shift << 16);
//	module_err("lsr64: %px", insn);
	emit(jit, insn);
}

// LSR <Xd>, <Xn>, <Xm>
void jit_lsrr64(jit_t* jit, int rd, int rn, int rm) {
//	module_err("lsr x%d, x%d, x%d", rd, rn, rm);
	uint32_t insn = 0x9ac02100;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rd;
	insn |= (rn << 5);
	insn |= (rm << 16);
	emit(jit, insn);
}

// LSL <Xd>, <Xn>, <Xm>
void jit_lslr64(jit_t* jit, int rd, int rn, int rm) {
//	module_err("lsl x%d, x%d, x%d", rd, rn, rm);
	uint32_t insn = 0x9ac02000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rd;
	insn |= (rn << 5);
	insn |= (rm << 16);
//	module_err("lslr64: %px", insn);
	emit(jit, insn);
}

// LSL <Xd>, <Xn>, #<shift>
void jit_lsl64(jit_t* jit, int rd, int rn, int shift) {
//	module_err("lsl x%d, x%d, #%d", rd, rn, shift);
	uint32_t insn = 0xd3400000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= shift && shift <= 63);
	insn |= rd;
	insn |= (rn << 5);
	int immr = 64 - shift;
	insn |= (immr << 16);
	insn |= ((immr - 1) << 10);
	emit(jit, insn);
}

// ROR <Xd>, <Xs>, #<shift>
void jit_ror64(jit_t* jit, int rd, int rs, int shift) {
//	module_err("ror x%d, x%d, #%d", rd, rs, shift);
	uint32_t insn = 0x93c00000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rs && rs <= 31);
	JIT_ASSERT(0 <= shift && shift <= 63);
	insn |= rd;
	insn |= (rs << 5);
	insn |= (rs << 16);
	int imms = shift;
	insn |= (imms << 10);
	emit(jit, insn);

}

// UDIV <Xd>, <Xn>, <Xm>
void jit_udiv64(jit_t* jit, int rd, int rn, int rm) {
//	module_err("udiv x%d, x%d, x%d", rd, rn, rm);
	uint32_t insn = 0x9ac00800;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	insn |= rd;
	insn |= (rn << 5);
	insn |= (rm << 16);
	emit(jit, insn);
}

// MSUB <Xd>, <Xn>, <Xm>, <Xa>
void jit_msub64(jit_t* jit, int rd, int rn, int rm, int ra) {
//	module_err("msub x%d, x%d, x%d, x%d", rd, rn, rm, ra);
	uint32_t insn = 0x9b008000;
	JIT_ASSERT(0 <= rd && rd <= 31);
	JIT_ASSERT(0 <= rn && rn <= 31);
	JIT_ASSERT(0 <= rm && rm <= 31);
	JIT_ASSERT(0 <= ra && ra <= 31);
	insn |= rd;
	insn |= (rn << 5);
	insn |= (ra << 10);
	insn |= (rm << 16);
	emit(jit, insn);
}

void jit_set_phr_neoversen3(jit_t* jit, uint64_t value, int rtmp) {
//	module_err("jit_set_phr_neoversen3: PHR, %px (tmp reg: x%d)", (uint8_t*)value, rtmp);
	jit_mov64(jit, rtmp, 0);
	for (int b = 0; b < 75 * 4; b += 4)  {
		uint64_t chunk = (value >> b) & ((1 << 4) - 1);
		jit_next_with_mask(jit, 0, (1 << 10) - 1, true);
		uint8_t* branch_address = jit_get_cur(jit);
		jit_next_with_mask(jit, chunk << 2, (1 << 11) - 1, true);
		uint8_t* branch_destination = jit_get_cur(jit);
		jit_set_cur(jit, branch_address);
		jit_cbz32(jit, rtmp, branch_destination);
	}
}

void jit_mrs(jit_t* jit, uint8_t o0, uint8_t op1, uint8_t CRn, uint8_t CRm, uint8_t op2, uint8_t Rt) {
//	module_err("mrs x%d, S%d_%d_%d_%d_%d", Rt, o0, op1, CRn, CRm, op2);
	uint32_t insn = 0xd5300000;
	JIT_ASSERT(0 <= o0 && o0 <= 1);
	JIT_ASSERT(0 <= op1 && op1 <= 7);
	JIT_ASSERT(0 <= CRn && CRn <= 15);
	JIT_ASSERT(0 <= CRm && CRm <= 15);
	JIT_ASSERT(0 <= op2 && op2 <= 7);
	JIT_ASSERT(0 <= Rt && Rt <= 31);
	insn |= Rt;
	insn |= (op2 << 5);
	insn |= (CRm << 8);
	insn |= (CRn << 12);
	insn |= (op1 << 16);
	insn |= (o0 << 19);
//	module_err("mrs: %px", insn);
	emit(jit, insn);
}
void jit_msr(jit_t* jit, uint8_t o0, uint8_t op1, uint8_t CRn, uint8_t CRm, uint8_t op2, uint8_t Rt) {
//	module_err("msr x%d, S%d_%d_%d_%d_%d", Rt, o0, op1, CRn, CRm, op2);
	uint32_t insn = 0xd5100000;
	JIT_ASSERT(0 <= o0 && o0 <= 1);
	JIT_ASSERT(0 <= op1 && op1 <= 7);
	JIT_ASSERT(0 <= CRn && CRn <= 15);
	JIT_ASSERT(0 <= CRm && CRm <= 15);
	JIT_ASSERT(0 <= op2 && op2 <= 7);
	JIT_ASSERT(0 <= Rt && Rt <= 31);
	insn |= Rt;
	insn |= (op2 << 5);
	insn |= (CRm << 8);
	insn |= (CRn << 12);
	insn |= (op1 << 16);
	insn |= (o0 << 19);
	emit(jit, insn);
}

void jit_msr_nzcv(jit_t* jit, uint8_t Rt) {
//	module_err("[call msr nzvc, x%d]", Rt);
	jit_msr(jit, 1, 3, 4, 2, 0, Rt);
}
void jit_mrs_nzcv(jit_t* jit, uint8_t Rt) {
//	module_err("[call mrs x%d, nzcv]", Rt);
	jit_mrs(jit, 1, 3, 4, 2, 0, Rt);
}

void jit_read64_pmu(jit_t* jit, uint8_t pmu, uint8_t Rt) {
//	module_err("[call mrs x%d, pmevcntr%d]", Rt, pmu);
	jit_mrs(jit, 1, 3, 14, 8 | ((pmu >> 3) & 3), pmu & 7, Rt);
}

void jit_mrs_pmevcntr0_el0_64(jit_t* jit, int rd) {
//	module_err("[call mrs x%d, pmevcntr0 (redundent)]", rd);
	uint32_t insn = 0xd53be800;
	JIT_ASSERT(0 <= rd && rd <= 31);
	insn |= rd;
	emit(jit, insn);
}
// sets as follows:
// given PHR currently has: p299p298p297p296...p3p2p1p0
// and reg_bit dynamically has value of bit k (0 or 1)
// and pos is t (index inside the phr, aka value between 0 to 299, inclusive)
// it will set PHR as follows:
// if prev_zero is false:
// p299p298p297p296...pt+1k000....000
// if prev_zero is true:
// 0000...0k000....000
void jit_set_phr_neoversen3_dynamic(jit_t* jit, uint64_t reg_bit, uint64_t rtmp1, uint64_t rtmp2, int pos, bool prev_zero) {
//	module_err("jit_set_phr_neoversen3_dynamic: set PHR to %d at index %d (should prev zero? %d). tmp1: x%d, tmp2: x%d", reg_bit, (int)pos, prev_zero, rtmp1, rtmp2);
	if(prev_zero) {
		for (int b = 0; b < (299 - pos) / 4; ++b) {
			uint8_t* load_dest_placeholder = jit_get_cur(jit);
			jit_li64(jit, rtmp1, 0); // placeholder

			uint8_t* curr_pos = jit_get_cur(jit);
			jit_br64(jit, rtmp1);

			jit_next_with_mask(jit, (uintptr_t)(curr_pos) << 1, (1 << 11) - 1, true);
			uint8_t* curr_target = jit_get_cur(jit);

			jit_set_cur(jit, load_dest_placeholder);
			jit_li64(jit, rtmp1, (uintptr_t)curr_target);

			jit_set_cur(jit, curr_target);
			jit_isb(jit);
			jit_dsb_sy(jit);
		}
	}
	uint8_t* initial_pos = jit_get_cur(jit);

	jit_li64(jit, rtmp1, (uint64_t)0); // placeholder
	jit_li64(jit, rtmp2, (uint64_t)0); // placeholder
	jit_cmp32(jit, reg_bit, 0);
	jit_csel64(jit, rtmp1, rtmp2, rtmp1, COND_NE);

	uint8_t* branch_pos = (uint8_t*)((uintptr_t)jit_get_cur(jit) & (0x3ff));
	jit_br64(jit, rtmp1); 

	jit_next_with_mask(jit, (uintptr_t)(branch_pos) << 1, (1 << 11) - 1, true);
	uint8_t* zero_target = jit_get_cur(jit);

	uintptr_t value_mask_set_one = (((uintptr_t)branch_pos >> 2) ^ ((1 << (pos % 4)))) << 3;
	jit_next_with_mask(jit, value_mask_set_one, (1 << 11) - 1, true);
	uint8_t* one_target = jit_get_cur(jit);

	jit_set_cur(jit, initial_pos);
	jit_li64(jit, rtmp1, (uint64_t)zero_target);
	jit_li64(jit, rtmp2, (uint64_t)one_target);

	jit_set_cur(jit, one_target);
	jit_isb(jit);
	jit_dsb_sy(jit);

	for (int b = 0; b < (pos / 4); ++b) {
		uint8_t* load_dest_placeholder = jit_get_cur(jit);
		jit_li64(jit, rtmp1, 0); // placeholder

		uint8_t* curr_pos = jit_get_cur(jit);
		jit_br64(jit, rtmp1);

		jit_next_with_mask(jit, (uintptr_t)(curr_pos) << 1, (1 << 11) - 1, true);
		uint8_t* curr_target = jit_get_cur(jit);

		jit_set_cur(jit, load_dest_placeholder);
		jit_li64(jit, rtmp1, (uintptr_t)curr_target);

		jit_set_cur(jit, curr_target);
		jit_isb(jit);
		jit_dsb_sy(jit);

	}
}

