#include "main.h"
#include <linux/moduleparam.h>
#include <linux/stdarg.h>

int mte_tag_verbose = 0;   // default off; enable at runtime for CE<->HW persistency checks
module_param(mte_tag_verbose, int, 0644);
MODULE_PARM_DESC(mte_tag_verbose, "Log MTE tag (re)init and per-run actual tags/pointers for CE-vs-HW consistency checking");

/* Write 1 to /sys/module/revizor_executor/parameters/mte_kasan_selftest to deliberately make a
 * tag-mismatched kernel access, proving KASAN_HW_TAGS is live and its report is retrievable via dmesg.
 * Under kasan.fault=report this reports-and-continues (no panic). Verification only. */
static int mte_kasan_selftest_set(const char *val, const struct kernel_param *kp) {
	(void)val; (void)kp;
	char *p = kmalloc(128, GFP_KERNEL);
	if (NULL == p) {
		return -ENOMEM;
	}
	/* Flip the low bit of the pointer's allocation tag so the access tag mismatches the memory tag. */
	volatile char *mis = (volatile char *)((((uint64_t)p) & ~(0xFULL << 56))
	                                        | (((((((uint64_t)p) >> 56) & 0xF) ^ 0x1)) << 56));
	module_err("mte_kasan_selftest: deliberate tag-mismatched read (orig=%px mis=%px) — expect a KASAN report\n",
	           p, (void *)mis);
	READ_ONCE(*mis);   /* KASAN_HW_TAGS: tag-check fault -> report */
	module_err("mte_kasan_selftest: returned from the mismatched read (KASAN reported & continued)\n");
	kfree(p);
	return 0;
}
static const struct kernel_param_ops mte_kasan_selftest_ops = { .set = mte_kasan_selftest_set };
module_param_cb(mte_kasan_selftest, &mte_kasan_selftest_ops, NULL, 0644);

/* Deferred log buffer (see mte.h): the per-input tag logging runs in the smp_call_function_single IPI
 * callback (IRQs off on the pinned CPU), where a printk storm hangs the CPU. Buffer here with snprintf
 * (atomic-safe), flush with printk after the pinned call returns. */
#define MTE_DUMP_MAX_LINES 640
#define MTE_DUMP_LINE_LEN  184
static char g_mte_dump[MTE_DUMP_MAX_LINES][MTE_DUMP_LINE_LEN];
static int  g_mte_dump_n;

void mte_dump_reset(void) { g_mte_dump_n = 0; }

void mte_dump_add(const char* fmt, ...) {
	va_list ap;
	if (!mte_tag_verbose || g_mte_dump_n >= MTE_DUMP_MAX_LINES) {
		return;
	}
	va_start(ap, fmt);
	vsnprintf(g_mte_dump[g_mte_dump_n], MTE_DUMP_LINE_LEN, fmt, ap);
	va_end(ap);
	++g_mte_dump_n;
}

void mte_dump_flush(void) {
	int i;
	for (i = 0; i < g_mte_dump_n; ++i) {
		module_err("%s\n", g_mte_dump[i]);
	}
	if (g_mte_dump_n >= MTE_DUMP_MAX_LINES) {
		module_err("MTETAG (dump buffer full: %d lines, some dropped)\n", MTE_DUMP_MAX_LINES);
	}
	g_mte_dump_n = 0;
}

/* android14-5.15 GKI names the EL1 tag-check field SCTLR_ELx_TCF_* and exposes no EL1 ATA (bit 43)
 * macro; newer kernels use SCTLR_EL1_TCF_*. */
#ifndef SCTLR_EL1_TCF_MASK
#define SCTLR_EL1_TCF_MASK SCTLR_ELx_TCF_MASK
#endif
#ifndef SCTLR_EL1_TCF_SYNC
#define SCTLR_EL1_TCF_SYNC SCTLR_ELx_TCF_SYNC
#endif
#ifndef SCTLR_EL1_TCF_ASYNC
#define SCTLR_EL1_TCF_ASYNC SCTLR_ELx_TCF_ASYNC
#endif
#ifndef SCTLR_EL1_ATA
#define SCTLR_EL1_ATA (UL(1) << 43)
#endif

/* Allocate a physically-contiguous region from the linear map (Normal-Tagged on
 * CONFIG_ARM64_MTE), so STG tagging and tag checks apply to it. get_order returns a 2^order
 * block aligned to its own size, so a caller whose size >= its alignment need (e.g. the sandbox
 * vs L1D_SIZE) is correctly aligned. Generic (page allocation), so defined for MTE and non-MTE. */
void *mte_alloc_tagged_region(size_t size) {
	return (void *)__get_free_pages(GFP_KERNEL | __GFP_ZERO, get_order(size));
}

void mte_free_tagged_region(void *ptr, size_t size) {
	if (NULL != ptr) {
		free_pages((unsigned long)ptr, get_order(size));
	}
}

void* mte_canonical_ptr(const void* p) {
	uintptr_t a = (uintptr_t)p;
	uintptr_t sext = (a & (1ull << 55)) ? (0xFFull << 56) : 0;
	return (void*)((a & 0x00FFFFFFFFFFFFFFull) | sext);
}

#if CONFIG_ARM64_MTE_HW	// Real MTE hardware implementation

static inline void stg(const void* ptr) {
	asm volatile("stg %[address], [%[address]]"
			:
			: [address]"r"(ptr)
			: "memory");
}

static inline void *tag_ptr(void *p, u8 tag) {
    // The allocation tag is bits [59:56]; mask to 4 bits so a wider value cannot bleed into the
    // rest of the top byte.
    return (void *)(((u64)p & 0x00FFFFFFFFFFFFFFULL) | ((u64)(tag & 0xF) << 56));
}


/* Emit `n` allocation-tag nibbles as chunked MTETAG lines, comparable byte-for-byte with the contract
 * executor's dump. `off_base` is the byte offset (may be negative) of granule 0 relative to the shared
 * origin (the main_region base) so CE and HW lines line up by offset. */
static void mte_log_tag_nibbles(const char *when, const char *region, int64_t off_base,
                                const uint8_t *tags, uint64_t n) {
	static const char hexd[] = "0123456789abcdef";
	char buf[129];
	uint64_t off;
	for (off = 0; off < n; off += 128) {
		uint64_t m = (n - off < 128) ? (n - off) : 128;
		uint64_t k;
		for (k = 0; k < m; ++k) {
			buf[k] = hexd[tags[off + k] & 0xF];
		}
		buf[m] = 0;
		mte_dump_add("MTETAG side=HW when=%s region=%s off=%lld n=%llu tags=%s",
		             when, region, (long long)(off_base + (int64_t)(off * 16)), m, buf);
	}
}

static inline u8 mte_read_tag_raw(const void* ptr) {
	u64 v = (u64)ptr;
	asm volatile("ldg %0, [%0]" : "+r"(v) :: "memory");
	return (v >> 56) & 0xF;
}

uint8_t mte_read_tag(const void* ptr) {
	return mte_read_tag_raw(ptr);
}

void mte_init_sandbox_tags(const void* base, uint64_t length, uint8_t tag) {
	uint64_t loc = 0;
	for (; loc < length; loc += MTE_GRANULE_SIZE) {
		uintptr_t current_ptr = (uintptr_t)base + loc;
		const void* tagged_ptr = tag_ptr((void*)current_ptr, tag);
		stg(tagged_ptr);
	}
	if (mte_tag_verbose) {
		mte_dump_add("MTETAG side=HW when=init base=0x%lx len=0x%llx uniform_tag=%x ngran=%llu",
		             (unsigned long)base, length, tag & 0xF, length / MTE_GRANULE_SIZE);
	}
}

void mte_apply_sandbox_tags(const void* base, const uint8_t* tags, uint64_t n_granules) {
	for (uint64_t i = 0; i < n_granules; ++i) {
		uintptr_t current_ptr = (uintptr_t)base + i * MTE_GRANULE_SIZE;
		const void* tagged_ptr = tag_ptr((void*)current_ptr, tags[i]);
		stg(tagged_ptr);
	}
	if (mte_tag_verbose) {
		mte_dump_add("MTETAG side=HW when=apply base=0x%lx ngran=%llu",
		             (unsigned long)base, n_granules);
		mte_log_tag_nibbles("apply", "applied", 0, tags, n_granules);
	}
}

// MTE system register bit accessors
DEFINE_FULL_MSR_BIT_ACCESSORS(TCO, TCO, 25)
DEFINE_FULL_MSR_BIT_ACCESSORS(TCR_EL1, TCMA1, 58)

uint8_t enable_TCMA1_bit(void) {
	return set_TCMA1_bit(1);
}

uint8_t disable_TCMA1_bit(void) {
	return set_TCMA1_bit(0);
}

uint8_t enable_TCO_bit(void) {
	return set_TCO_bit(1);
}

uint8_t disable_TCO_bit(void) {
	return set_TCO_bit(0);
}

void mte_set_sync(void) {
	// ATA (bit 43) enables allocation-tag access; TCF (bits [41:40]) = 0b01 = synchronous faults.
	sysreg_clear_set(sctlr_el1, SCTLR_EL1_TCF_MASK, SCTLR_EL1_ATA | SCTLR_EL1_TCF_SYNC);
	isb();
}

#ifndef SYS_GCR_EL1
#define SYS_GCR_EL1 sys_reg(3, 0, 1, 0, 6)
#endif
#define GCR_EL1_EXCL_MASK 0xFFFFUL

uint64_t mte_gcr_read(void) {
	return read_sysreg_s(SYS_GCR_EL1);
}

#ifndef SYS_TFSR_EL1
#define SYS_TFSR_EL1 sys_reg(3, 0, 5, 6, 0)
#endif

// Read the async tag-fault status (TFSR_EL1) and clear it. Non-zero means a tag-check fault occurred
// since it was last cleared -- with the kernel's async TCF this is how a mismatched access is recorded
// without a synchronous abort. Used to attribute a fault to the exact input that caused it.
uint64_t mte_read_clear_tfsr(void) {
	uint64_t t = read_sysreg_s(SYS_TFSR_EL1);
	if (t) {
		write_sysreg_s(0, SYS_TFSR_EL1);
		isb();
	}
	return t;
}

// Force TCF=SYNC for the measurement window (returns the prior SCTLR_EL1 to restore). The kernel's
// async TCF makes tag faults imprecise and decoupled from the run that caused them; SYNC delivers them
// as a precise abort attributed to the executor, so a stray fault is caught immediately, not silently.
uint64_t mte_force_sync(void) {
	uint64_t saved = read_sysreg(sctlr_el1);
	sysreg_clear_set(sctlr_el1, SCTLR_EL1_TCF_MASK, SCTLR_EL1_ATA | SCTLR_EL1_TCF_SYNC);
	isb();
	return saved;
}

void mte_restore_sctlr(uint64_t saved) {
	write_sysreg(saved, sctlr_el1);
	isb();
}

uint64_t mte_gcr_clear_exclude(void) {
	uint64_t g = read_sysreg_s(SYS_GCR_EL1);
	write_sysreg_s(g & ~GCR_EL1_EXCL_MASK, SYS_GCR_EL1);
	isb();
	static bool logged = false;
	if (!logged) {
		logged = true;
		uint64_t after = read_sysreg_s(SYS_GCR_EL1);
		module_err("GCR_EL1 inherited=0x%llx (Exclude=0x%llx) -> after clear=0x%llx (Exclude=0x%llx)\n",
		           g, g & GCR_EL1_EXCL_MASK, after, after & GCR_EL1_EXCL_MASK);
	}
	return g;
}

void mte_gcr_restore(uint64_t saved_gcr) {
	write_sysreg_s(saved_gcr, SYS_GCR_EL1);
	isb();
}

void mte_save_control(struct mte_control_state* state) {
	state->sctlr_el1 = read_sysreg(sctlr_el1);
	state->tcr_el1 = read_TCR_EL1();
}

void mte_restore_control(const struct mte_control_state* state) {
	write_sysreg(state->sctlr_el1, sctlr_el1);
	write_TCR_EL1(state->tcr_el1);
	isb();
}

bool mte_region_is_tagged(const void *ptr, size_t size) {
	return pte_region_attr_is((void *)ptr, size, MT_NORMAL_TAGGED);
}

#else	// Non-MTE hardware: all stubs

static inline void stg(const void* ptr)				{ (void)ptr; }

void mte_init_sandbox_tags(const void* base, uint64_t length, uint8_t tag) { (void)base; (void)length; (void)tag; }

void mte_apply_sandbox_tags(const void* base, const uint8_t* tags, uint64_t n_granules) { (void)base; (void)tags; (void)n_granules; }

uint8_t mte_read_tag(const void* ptr) { (void)ptr; return 0xFF; }

uint8_t enable_TCMA1_bit(void)					{ return 0; }

uint8_t disable_TCMA1_bit(void)					{ return 0; }

uint8_t enable_TCO_bit(void)					{ return 0; }

uint8_t disable_TCO_bit(void)					{ return 0; }

uint64_t mte_gcr_read(void)					{ return 0; }

uint64_t mte_read_clear_tfsr(void)				{ return 0; }

uint64_t mte_force_sync(void)					{ return 0; }

void mte_restore_sctlr(uint64_t saved)				{ (void)saved; }

uint64_t mte_gcr_clear_exclude(void)				{ return 0; }

void mte_gcr_restore(uint64_t saved_gcr)			{ (void)saved_gcr; }

bool mte_region_is_tagged(const void *ptr, size_t size)		{ (void)ptr; (void)size; return true; }

#endif

