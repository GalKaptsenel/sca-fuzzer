#include "pac.h"
#include "main.h"

#define write_sysreg(val, reg) \
    asm volatile("msr " #reg ", %0" :: "r"(val))

#define read_sysreg(reg) ({ \
    uint64_t __val; \
    asm volatile("mrs %0, " #reg : "=r"(__val)); \
    __val; \
})

#define isb() asm volatile("isb" ::: "memory")

/* ------------------ FEATURE DETECTION ------------------ */

static inline int pauth_apa(void) {
	uint64_t isar1 = read_sysreg(ID_AA64ISAR1_EL1);
	uint64_t apa = (isar1 >> 4) & 0xF;
	return apa;
}

static inline int pauth_apa3(void) {
	uint64_t isar2 = read_sysreg(ID_AA64ISAR2_EL1);
	uint64_t apa3 = (isar2 >> 12) & 0xF;
	return apa3;
}

static inline int pauth_api(void) {
	uint64_t isar1 = read_sysreg(ID_AA64ISAR1_EL1);
	uint64_t api = (isar1 >> 8) & 0xF;
	return api;
}

#define DEFINE_KEY_INIT_FUNC(reg_name) \
static void pauth_set_key_##reg_name(uint64_t hi, uint64_t lo) { \
	write_sysreg(lo, reg_name##KeyLo_EL1); \
	write_sysreg(hi, reg_name##KeyHi_EL1); \
	isb(); \
}\
static uint64_t pauth_get_key_##reg_name##_lo(void) { \
	return read_sysreg(reg_name##KeyLo_EL1); \
}\
static uint64_t pauth_get_key_##reg_name##_hi(void) { \
	return read_sysreg(reg_name##KeyHi_EL1); \
}\



DEFINE_KEY_INIT_FUNC(APIA)
DEFINE_KEY_INIT_FUNC(APIB)
DEFINE_KEY_INIT_FUNC(APDA)
DEFINE_KEY_INIT_FUNC(APDB)
DEFINE_KEY_INIT_FUNC(APGA)

static void pauth_set_keys(
		uint64_t ia_hi, uint64_t ia_lo,
		uint64_t ib_hi, uint64_t ib_lo,
		uint64_t da_hi, uint64_t da_lo,
		uint64_t db_hi, uint64_t db_lo,
		uint64_t ga_hi, uint64_t ga_lo) {
	pauth_set_key_APIA(ia_hi, ia_lo);
	pauth_set_key_APIB(ib_hi, ib_lo);
	pauth_set_key_APDA(da_hi, da_lo);
	pauth_set_key_APDB(db_hi, db_lo);
	pauth_set_key_APGA(ga_hi, ga_lo);
}

#define SCTLR_EL1_EnIA (1UL << 31)
#define SCTLR_EL1_EnIB (1UL << 30)
#define SCTLR_EL1_EnDA (1UL << 27)
#define SCTLR_EL1_EnDB (1UL << 13)

static void pauth_enable_all(void) {
	uint64_t sctlr = read_sysreg(SCTLR_EL1);

	sctlr |= SCTLR_EL1_EnIA;
	sctlr |= SCTLR_EL1_EnIB;
	sctlr |= SCTLR_EL1_EnDA;
	sctlr |= SCTLR_EL1_EnDB;

	write_sysreg(sctlr, SCTLR_EL1);

	isb();
}

/*
 * Ensure TBI is enabled so PAC bits in top byte don't break translation.
 */
static void pauth_enable_tbi(void) {
	uint64_t tcr = read_sysreg(TCR_EL1);

	/*
	* TBI0 = bit 37 (EL0)
	* TBI1 = bit 38 (EL1)
	*
	* Enable both for safety
	*/
	tcr |= (1UL << 37); // TBI0
	tcr |= (1UL << 38); // TBI1

	write_sysreg(tcr, TCR_EL1);

	isb();
}

struct pauth_saved_state {
	uint64_t sctlr_el1;
	uint64_t tcr_el1;

	uint64_t apia_lo, apia_hi;
	uint64_t apib_lo, apib_hi;
	uint64_t apda_lo, apda_hi;
	uint64_t apdb_lo, apdb_hi;
	uint64_t apga_lo, apga_hi;
};

static DEFINE_PER_CPU(struct pauth_saved_state, pauth_state);
static DEFINE_PER_CPU(int, pauth_initialized);
#define this_pauth_state()   this_cpu_ptr(&pauth_state)
#define this_pauth_init()    (*this_cpu_ptr(&pauth_initialized))

static void pauth_save_state(void) {
	struct pauth_saved_state *st = this_pauth_state();

	st->sctlr_el1 = read_sysreg(SCTLR_EL1);
	st->tcr_el1   = read_sysreg(TCR_EL1);

	st->apia_lo = pauth_get_key_APIA_lo();
	st->apia_hi = pauth_get_key_APIA_hi();

	st->apib_lo = pauth_get_key_APIB_lo();
	st->apib_hi = pauth_get_key_APIB_hi();

	st->apda_lo = pauth_get_key_APDA_lo();
	st->apda_hi = pauth_get_key_APDA_hi();

	st->apdb_lo = pauth_get_key_APDB_lo();
	st->apdb_hi = pauth_get_key_APDB_hi();

	st->apga_lo = pauth_get_key_APGA_lo();
	st->apga_hi = pauth_get_key_APGA_hi();

	isb();
}

void pauth_restore_cpu(void) {
	struct pauth_saved_state *st = NULL;

	preempt_disable();
	if (!this_pauth_init()) {
		preempt_enable();
		return;
	}
	local_irq_disable();

	st = this_pauth_state();

	uint64_t sctlr = read_sysreg(SCTLR_EL1);

	sctlr &= ~(SCTLR_EL1_EnIA |
		   SCTLR_EL1_EnIB |
		   SCTLR_EL1_EnDA |
		   SCTLR_EL1_EnDB);

	write_sysreg(sctlr, SCTLR_EL1);
	isb();

	pauth_set_keys(
		st->apia_hi, st->apia_lo,
		st->apib_hi, st->apib_lo,
		st->apda_hi, st->apda_lo,
		st->apdb_hi, st->apdb_lo,
		st->apga_hi, st->apga_lo
	);

	write_sysreg(st->tcr_el1, TCR_EL1);
	isb();

	sctlr |= (SCTLR_EL1_EnIA & st->sctlr_el1);
	sctlr |= (SCTLR_EL1_EnIB & st->sctlr_el1);
	sctlr |= (SCTLR_EL1_EnDA & st->sctlr_el1);
	sctlr |= (SCTLR_EL1_EnDB & st->sctlr_el1);

	write_sysreg(sctlr, SCTLR_EL1);
	isb();

	this_pauth_init() = 0;

	local_irq_enable();
	preempt_enable();
}

int pauth_init_cpu(
		uint64_t ia_hi, uint64_t ia_lo,
		uint64_t ib_hi, uint64_t ib_lo,
		uint64_t da_hi, uint64_t da_lo,
		uint64_t db_hi, uint64_t db_lo,
		uint64_t ga_hi,  uint64_t ga_lo) {
	module_err("ID_AA64ISAR1_EL1: %px", (void *)(uintptr_t)read_sysreg(ID_AA64ISAR1_EL1));
	module_err("ID_AA64ISAR2_EL1: %px", (void *)(uintptr_t)read_sysreg(ID_AA64ISAR2_EL1));
	if (!pauth_apa() && !pauth_api() && !pauth_apa3()) {
		module_err("EOPNOTSUPP is returned");
		return -EOPNOTSUPP;
	}

	preempt_disable();
	if(this_pauth_init()) {
		preempt_enable();
		module_err("EBUSY is returned");
		return -EBUSY;
	}
	local_irq_disable();

	pauth_save_state();

 	pauth_enable_tbi();
	pauth_set_keys(
			ia_hi, ia_lo,
			ib_hi, ib_lo,
			da_hi, da_lo,
			db_hi, db_lo,
			ga_hi, ga_lo
	);
	pauth_enable_all();

	this_pauth_init() = 1;

	local_irq_enable();
	preempt_enable();

	return 0;
}

struct pauth_args {
	uint64_t ia_hi, ia_lo;
        uint64_t ib_hi, ib_lo;
        uint64_t da_hi, da_lo;
        uint64_t db_hi, db_lo;
        uint64_t ga_hi, ga_lo;
};

static void pauth_init_cpu_wrapper(void *info) {
        struct pauth_args *a = info;
	module_err("Initializing PAC per cpu");

        pauth_init_cpu(
                a->ia_hi, a->ia_lo,
                a->ib_hi, a->ib_lo,
                a->da_hi, a->da_lo,
                a->db_hi, a->db_lo,
                a->ga_hi, a->ga_lo
        );
}
static void pauth_restore_cpu_wrapper(void *info) {
        pauth_restore_cpu();
}

int pauth_init_all_cpus(
		uint64_t ia_hi, uint64_t ia_lo,
		uint64_t ib_hi, uint64_t ib_lo,
		uint64_t da_hi, uint64_t da_lo,
		uint64_t db_hi, uint64_t db_lo,
		uint64_t ga_hi,  uint64_t ga_lo) {
	struct pauth_args args;
	args.ia_hi = ia_hi;
	args.ia_lo = ia_lo;
	args.ib_hi = ib_hi;
	args.ib_lo = ib_lo;
	args.da_hi = da_hi;
	args.da_lo = da_lo;
	args.db_hi = db_hi;
	args.db_lo = db_lo;
	args.ga_hi = ga_hi;
	args.ga_lo = ga_lo;
	module_err("Initializing PAC on all CPUs");
	on_each_cpu(pauth_init_cpu_wrapper, &args, 1);
	return 0;
}

void pauth_restore_all_cpus(void) {
	on_each_cpu(pauth_restore_cpu_wrapper, NULL, 1);
}

inline uint64_t pacia(uint64_t ptr, uint64_t mod) {
	asm volatile("pacia %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

inline uint64_t pacib(uint64_t ptr, uint64_t mod) {
	asm volatile("pacib %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

inline uint64_t pacda(uint64_t ptr, uint64_t mod) {
	asm volatile("pacda %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

inline uint64_t pacdb(uint64_t ptr, uint64_t mod) {
	asm volatile("pacdb %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

inline uint64_t pacga(uint64_t x, uint64_t y) {
	uint64_t out = 0;
	asm volatile("pacga %0, %1, %2" : "=r"(out) : "r"(x), "r"(y));
	return out;
}

inline uint64_t autia(uint64_t ptr, uint64_t mod) {
	asm volatile("autia %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

inline uint64_t autib(uint64_t ptr, uint64_t mod) {
	asm volatile("autib %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

inline uint64_t autda(uint64_t ptr, uint64_t mod) {
	asm volatile("autda %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

inline uint64_t autdb(uint64_t ptr, uint64_t mod) {
	asm volatile("autdb %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

inline uint64_t autiza(uint64_t ptr) {
	asm volatile("autiza %0" : "+r"(ptr));
	return ptr;
}

inline uint64_t autizb(uint64_t ptr) {
	asm volatile("autizb %0" : "+r"(ptr));
	return ptr;
}

inline uint64_t autdza(uint64_t ptr) {
	asm volatile("autdza %0" : "+r"(ptr));
	return ptr;
}

inline uint64_t autdzb(uint64_t ptr) {
	asm volatile("autdzb %0" : "+r"(ptr));
	return ptr;
}

inline uint64_t paciza(uint64_t ptr) {
	asm volatile("paciza %0" : "+r"(ptr));
	return ptr;
}

inline uint64_t pacizb(uint64_t ptr) {
	asm volatile("pacizb %0" : "+r"(ptr));
	return ptr;
}

inline uint64_t pacdza(uint64_t ptr) {
	asm volatile("pacdza %0" : "+r"(ptr));
	return ptr;
}

inline uint64_t pacdzb(uint64_t ptr) {
	asm volatile("pacdzb %0" : "+r"(ptr));
	return ptr;
}

inline uint64_t xpaci(uint64_t ptr) {
	asm volatile("xpaci %0" : "+r"(ptr));
	return ptr;
}

inline uint64_t xpacd(uint64_t ptr) {
	asm volatile("xpacd %0" : "+r"(ptr));
	return ptr;
}

uint64_t pac_enable_all_keys(void) {
	uint64_t sctlr;
	asm volatile("mrs %0, sctlr_el1" : "=r"(sctlr));
	uint64_t new_sctlr = sctlr | SCTLR_EL1_EnIA | SCTLR_EL1_EnIB | SCTLR_EL1_EnDA | SCTLR_EL1_EnDB;
	asm volatile("msr sctlr_el1, %0; isb" :: "r"(new_sctlr) : "memory");
	return sctlr;
}

void pac_restore_sctlr(uint64_t saved_sctlr) {
	asm volatile("msr sctlr_el1, %0; isb" :: "r"(saved_sctlr) : "memory");
}

void pac_save_keys(struct pac_keys *out) {
	out->apia_lo = pauth_get_key_APIA_lo();
	out->apia_hi = pauth_get_key_APIA_hi();
	out->apib_lo = pauth_get_key_APIB_lo();
	out->apib_hi = pauth_get_key_APIB_hi();
	out->apda_lo = pauth_get_key_APDA_lo();
	out->apda_hi = pauth_get_key_APDA_hi();
	out->apdb_lo = pauth_get_key_APDB_lo();
	out->apdb_hi = pauth_get_key_APDB_hi();
	out->apga_lo = pauth_get_key_APGA_lo();
	out->apga_hi = pauth_get_key_APGA_hi();
	isb();
}

void pac_load_keys(const struct pac_keys *keys) {
	pauth_set_key_APIA(keys->apia_hi, keys->apia_lo);
	pauth_set_key_APIB(keys->apib_hi, keys->apib_lo);
	pauth_set_key_APDA(keys->apda_hi, keys->apda_lo);
	pauth_set_key_APDB(keys->apdb_hi, keys->apdb_lo);
	pauth_set_key_APGA(keys->apga_hi, keys->apga_lo);
	/* isb() is called inside each pauth_set_key_* */
}

void pac_swap_user_keys(const struct pac_keys *new_keys, struct pac_keys *saved_keys)
{
	preempt_disable();

	/* Save current user PAC keys from task_struct (NOT from hardware, since
	 * in EL1 the hardware APIA holds the kernel key, not the user key). */
	saved_keys->apia_lo = (uint64_t)current->thread.keys_user.apia.lo;
	saved_keys->apia_hi = (uint64_t)current->thread.keys_user.apia.hi;
	saved_keys->apib_lo = (uint64_t)current->thread.keys_user.apib.lo;
	saved_keys->apib_hi = (uint64_t)current->thread.keys_user.apib.hi;
	saved_keys->apda_lo = (uint64_t)current->thread.keys_user.apda.lo;
	saved_keys->apda_hi = (uint64_t)current->thread.keys_user.apda.hi;
	saved_keys->apdb_lo = (uint64_t)current->thread.keys_user.apdb.lo;
	saved_keys->apdb_hi = (uint64_t)current->thread.keys_user.apdb.hi;
	saved_keys->apga_lo = (uint64_t)current->thread.keys_user.apga.lo;
	saved_keys->apga_hi = (uint64_t)current->thread.keys_user.apga.hi;

	/* Update task_struct for all 5 keys (kernel_exit installs APIA from here on
	 * return to EL0; the others are installed on context switch too). */
	current->thread.keys_user.apia.lo = (unsigned long)new_keys->apia_lo;
	current->thread.keys_user.apia.hi = (unsigned long)new_keys->apia_hi;
	current->thread.keys_user.apib.lo = (unsigned long)new_keys->apib_lo;
	current->thread.keys_user.apib.hi = (unsigned long)new_keys->apib_hi;
	current->thread.keys_user.apda.lo = (unsigned long)new_keys->apda_lo;
	current->thread.keys_user.apda.hi = (unsigned long)new_keys->apda_hi;
	current->thread.keys_user.apdb.lo = (unsigned long)new_keys->apdb_lo;
	current->thread.keys_user.apdb.hi = (unsigned long)new_keys->apdb_hi;
	current->thread.keys_user.apga.lo = (unsigned long)new_keys->apga_lo;
	current->thread.keys_user.apga.hi = (unsigned long)new_keys->apga_hi;

	/* Load APIB/APDA/APDB/APGA into hardware immediately.  Do NOT touch APIA:
	 * the kernel uses hardware APIA for its own return-address auth, and
	 * changing it mid-ioctl (while kernel frames are on the stack) would
	 * corrupt those authenticated return addresses. */
	pauth_set_key_APIB(new_keys->apib_hi, new_keys->apib_lo);
	pauth_set_key_APDA(new_keys->apda_hi, new_keys->apda_lo);
	pauth_set_key_APDB(new_keys->apdb_hi, new_keys->apdb_lo);
	pauth_set_key_APGA(new_keys->apga_hi, new_keys->apga_lo);

	preempt_enable();
}

void trigger_pauth_fault(void) {
	uint64_t value = 100;
	uint64_t* orig_ptr = &value;
	uint64_t* pacda_123_orig_ptr = (uint64_t*)pacda((uint64_t)orig_ptr, 123);
	uint64_t* pacda_122_orig_ptr = (uint64_t*)pacda((uint64_t)orig_ptr, 122);
	uint64_t* pacdb_123_orig_ptr = (uint64_t*)pacdb((uint64_t)orig_ptr, 123);
	uint64_t* pacia_123_orig_ptr = (uint64_t*)pacia((uint64_t)orig_ptr, 123);
	uint64_t* double_sign_same = (uint64_t*)pacda((uint64_t)pacda_123_orig_ptr, 123);
	uint64_t* double_sign_other = (uint64_t*)pacdb((uint64_t)pacda_123_orig_ptr, 123);
	uint64_t* double_sign_other_mod = (uint64_t*)pacda((uint64_t)pacda_123_orig_ptr, 122);
	uint64_t* double_sign_very_other = (uint64_t*)pacia((uint64_t)pacda_123_orig_ptr, 123);

	uint64_t* autda_123_correct = (uint64_t*)pacda((uint64_t)pacda_123_orig_ptr, 123);
	uint64_t* autda_122_correct = (uint64_t*)pacda((uint64_t)pacda_123_orig_ptr, 122);
	uint64_t* autdb_123_correct = (uint64_t*)pacdb((uint64_t)pacda_123_orig_ptr, 123);
	uint64_t* autia_123_correct = (uint64_t*)pacia((uint64_t)pacda_123_orig_ptr, 123);
	uint64_t* double_aut = (uint64_t*)pacda((uint64_t)autda_123_correct, 123);
	uint64_t* double_aut_other = (uint64_t*)pacdb((uint64_t)autda_123_correct, 123);
	uint64_t* double_aut_other_mod = (uint64_t*)pacda((uint64_t)autda_123_correct, 122);
	uint64_t* double_aut_very_other = (uint64_t*)pacia((uint64_t)autda_123_correct, 123);

	uint64_t* xpacd_strip = (uint64_t*)xpacd((uint64_t)pacda_123_orig_ptr);
	uint64_t* xpaci_strip = (uint64_t*)xpaci((uint64_t)pacda_123_orig_ptr);
	uint64_t* xpacd_orig = (uint64_t*)xpacd((uint64_t)orig_ptr);
	uint64_t* xpaci_orig = (uint64_t*)xpaci((uint64_t)orig_ptr);
	uint64_t* xpacd_auted = (uint64_t*)xpacd((uint64_t)autda_123_correct);
	uint64_t* xpaci_auted = (uint64_t*)xpaci((uint64_t)autda_123_correct);

	module_err("orig_ptr = %px\npacda_123_orig_ptr  = %px\npacda_122_orig_ptr = %px\npacdb_123_orig_ptr = %px\npacia_123_orig_ptr = %px\ndouble_sign_same  = %px\ndouble_sign_other = %px\ndouble_sign_other_mod  = %px\ndouble_sign_very_other = %px\nautda_123_correct = %px\nautda_122_correct = %px\nautdb_123_correct = %px\nautia_123_correct = %px\ndouble_aut = %px\ndouble_aut_other = %px\ndouble_aut_other_mod = %px\ndouble_aut_very_other = %px\nxpacd_strip = %px\nxpaci_strip = %px\nxpacd_orig = %px\nxpaci_orig  = %px\nxpacd_auted  = %px\nxpaci_auted  = %px\nread: %px\n, read_misuse: %px\nread_xpacd_strip: %px\nread_xpaci_auted: %px\n",
		       	orig_ptr, pacda_123_orig_ptr, pacda_122_orig_ptr,
			pacdb_123_orig_ptr, pacia_123_orig_ptr, double_sign_same, double_sign_other, double_sign_other_mod,
			double_sign_very_other, autda_123_correct, autda_122_correct, autdb_123_correct, autia_123_correct,   
			double_aut, double_aut_other, double_aut_other_mod, double_aut_very_other, xpacd_strip, xpaci_strip,  
			xpacd_orig, xpaci_orig, xpacd_auted, xpaci_auted, (uint64_t*)*autda_123_correct, (uint64_t*)*double_sign_same, (uint64_t*)*xpacd_strip, (uint64_t*)*xpaci_auted); 
	module_err("should not print: %px", (uint64_t*)*autda_122_correct);

//
//        asm volatile(
//                /* Sign LR with SP */
//                "pacda x30, x1\n"
//
//                /* Corrupt the signed pointer */
//                "eor x30, x30, #0x100\n"
//
//                /* Authenticate (will FAIL) */
////                "autda x30, x1\n"
//
//                /* Return using corrupted LR → exception */
//                "ret\n"
//        );
}
