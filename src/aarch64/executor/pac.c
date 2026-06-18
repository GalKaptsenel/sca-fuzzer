#include "pac.h"
#include "main.h"

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

uint64_t pacia(uint64_t ptr, uint64_t mod) {
	asm volatile("pacia %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

uint64_t pacib(uint64_t ptr, uint64_t mod) {
	asm volatile("pacib %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

uint64_t pacda(uint64_t ptr, uint64_t mod) {
	asm volatile("pacda %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

uint64_t pacdb(uint64_t ptr, uint64_t mod) {
	asm volatile("pacdb %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

uint64_t pacga(uint64_t x, uint64_t y) {
	uint64_t out = 0;
	asm volatile("pacga %0, %1, %2" : "=r"(out) : "r"(x), "r"(y));
	return out;
}

uint64_t autia(uint64_t ptr, uint64_t mod) {
	asm volatile("autia %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

uint64_t autib(uint64_t ptr, uint64_t mod) {
	asm volatile("autib %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

uint64_t autda(uint64_t ptr, uint64_t mod) {
	asm volatile("autda %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

uint64_t autdb(uint64_t ptr, uint64_t mod) {
	asm volatile("autdb %0, %1" : "+r"(ptr) : "r"(mod));
	return ptr;
}

uint64_t autiza(uint64_t ptr) {
	asm volatile("autiza %0" : "+r"(ptr));
	return ptr;
}

uint64_t autizb(uint64_t ptr) {
	asm volatile("autizb %0" : "+r"(ptr));
	return ptr;
}

uint64_t autdza(uint64_t ptr) {
	asm volatile("autdza %0" : "+r"(ptr));
	return ptr;
}

uint64_t autdzb(uint64_t ptr) {
	asm volatile("autdzb %0" : "+r"(ptr));
	return ptr;
}

uint64_t paciza(uint64_t ptr) {
	asm volatile("paciza %0" : "+r"(ptr));
	return ptr;
}

uint64_t pacizb(uint64_t ptr) {
	asm volatile("pacizb %0" : "+r"(ptr));
	return ptr;
}

uint64_t pacdza(uint64_t ptr) {
	asm volatile("pacdza %0" : "+r"(ptr));
	return ptr;
}

uint64_t pacdzb(uint64_t ptr) {
	asm volatile("pacdzb %0" : "+r"(ptr));
	return ptr;
}

uint64_t xpaci(uint64_t ptr) {
	asm volatile("xpaci %0" : "+r"(ptr));
	return ptr;
}

uint64_t xpacd(uint64_t ptr) {
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

