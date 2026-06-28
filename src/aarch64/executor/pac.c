#include "pac.h"
#include "main.h"
#include <linux/irqflags.h>
#include <linux/preempt.h>
#include <linux/string.h>

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



/*
 * Everything below runs with a foreign APIA/APIB key live (see pac_run_op_with_keys),
 * so it must contain no authenticated return: build this region without pac-ret (BTI is
 * kept for indirect-call targets). Restored to the kernel default by pop_options at EOF.
 */
#pragma GCC push_options
#pragma GCC target("branch-protection=bti")

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

enum pac_op pac_sign_op_from_mnemonic(const char *m) {
	if (!strcmp(m, "pacia"))  return PAC_OP_PACIA;
	if (!strcmp(m, "pacib"))  return PAC_OP_PACIB;
	if (!strcmp(m, "pacda"))  return PAC_OP_PACDA;
	if (!strcmp(m, "pacdb"))  return PAC_OP_PACDB;
	if (!strcmp(m, "pacga"))  return PAC_OP_PACGA;
	if (!strcmp(m, "paciza")) return PAC_OP_PACIZA;
	if (!strcmp(m, "pacizb")) return PAC_OP_PACIZB;
	if (!strcmp(m, "pacdza")) return PAC_OP_PACDZA;
	if (!strcmp(m, "pacdzb")) return PAC_OP_PACDZB;
	return PAC_OP_INVALID;
}

enum pac_op pac_auth_op_from_mnemonic(const char *m) {
	if (!strcmp(m, "autia"))  return PAC_OP_AUTIA;
	if (!strcmp(m, "autib"))  return PAC_OP_AUTIB;
	if (!strcmp(m, "autda"))  return PAC_OP_AUTDA;
	if (!strcmp(m, "autdb"))  return PAC_OP_AUTDB;
	if (!strcmp(m, "autiza")) return PAC_OP_AUTIZA;
	if (!strcmp(m, "autizb")) return PAC_OP_AUTIZB;
	if (!strcmp(m, "autdza")) return PAC_OP_AUTDZA;
	if (!strcmp(m, "autdzb")) return PAC_OP_AUTDZB;
	return PAC_OP_INVALID;
}

static uint64_t pac_run_op(enum pac_op op, uint64_t ptr, uint64_t mod) {
	switch (op) {
	case PAC_OP_PACIA:  return pacia(ptr, mod);
	case PAC_OP_PACIB:  return pacib(ptr, mod);
	case PAC_OP_PACDA:  return pacda(ptr, mod);
	case PAC_OP_PACDB:  return pacdb(ptr, mod);
	case PAC_OP_PACGA:  return pacga(ptr, mod);
	case PAC_OP_PACIZA: return paciza(ptr);
	case PAC_OP_PACIZB: return pacizb(ptr);
	case PAC_OP_PACDZA: return pacdza(ptr);
	case PAC_OP_PACDZB: return pacdzb(ptr);
	case PAC_OP_AUTIA:  return autia(ptr, mod);
	case PAC_OP_AUTIB:  return autib(ptr, mod);
	case PAC_OP_AUTDA:  return autda(ptr, mod);
	case PAC_OP_AUTDB:  return autdb(ptr, mod);
	case PAC_OP_AUTIZA: return autiza(ptr);
	case PAC_OP_AUTIZB: return autizb(ptr);
	case PAC_OP_AUTDZA: return autdza(ptr);
	case PAC_OP_AUTDZB: return autdzb(ptr);
	default:            return 0;
	}
}

/*
 * Loading a foreign APIA/APIB key while the kernel is built with pac-ret makes the
 * next authenticated return (RETAA in this module, or any IRQ-return / context switch
 * into the pac-ret kernel) fail authentication -> EL1 FPAC fault. So the swap is
 * confined to a window with IRQs and preemption disabled, and this code is built
 * without pac-ret (the branch-protection=bti pragma region above) so the returns
 * executed inside the window do not authenticate. The kernel keys are restored
 * before the window ends.
 */
uint64_t pac_run_op_with_keys(enum pac_op op, uint64_t ptr, uint64_t mod,
                              bool keys_set, const struct pac_keys *exec_keys) {
	struct pac_keys saved_keys;
	uint64_t saved_sctlr, result;
	unsigned long flags;

	if (!keys_set) {
		return pac_run_op(op, ptr, mod);
	}

	local_irq_save(flags);
	preempt_disable();

	pac_save_keys(&saved_keys);
	pac_load_keys(exec_keys);
	saved_sctlr = pac_enable_all_keys();

	result = pac_run_op(op, ptr, mod);

	pac_restore_sctlr(saved_sctlr);
	pac_load_keys(&saved_keys);

	preempt_enable();
	local_irq_restore(flags);
	return result;
}

/* Auth op -> the sign op that produces the same PAC field, for non-faulting verification. */
static enum pac_op pac_auth_to_sign(enum pac_op op) {
	switch (op) {
	case PAC_OP_AUTIA:  return PAC_OP_PACIA;
	case PAC_OP_AUTIB:  return PAC_OP_PACIB;
	case PAC_OP_AUTDA:  return PAC_OP_PACDA;
	case PAC_OP_AUTDB:  return PAC_OP_PACDB;
	case PAC_OP_AUTIZA: return PAC_OP_PACIZA;
	case PAC_OP_AUTIZB: return PAC_OP_PACIZB;
	case PAC_OP_AUTDZA: return PAC_OP_PACDZA;
	case PAC_OP_AUTDZB: return PAC_OP_PACDZB;
	default:            return PAC_OP_INVALID;
	}
}

/* Strip the same key family's PAC field as this auth op (XPAC never faults). */
static uint64_t pac_auth_strip(enum pac_op op, uint64_t ptr) {
	switch (op) {
	case PAC_OP_AUTIA: case PAC_OP_AUTIB:
	case PAC_OP_AUTIZA: case PAC_OP_AUTIZB:
		return xpaci(ptr);
	default:
		return xpacd(ptr);
	}
}

/* Model the AUT* with non-faulting ops: strip (XPAC) then re-sign the canonical pointer with the
 * same key+context. The signature is correct iff the re-signed value equals the original pointer.
 * Only then run the real AUT* (guaranteed to succeed); otherwise flag would_fault and return the
 * canonical pointer WITHOUT executing AUT*. Assumes the intended keys/SCTLR are already live. */
static uint64_t pac_auth_verify_core(enum pac_op op, uint64_t ptr, uint64_t mod, bool *would_fault) {
	uint64_t canonical = pac_auth_strip(op, ptr);
	uint64_t resigned  = pac_run_op(pac_auth_to_sign(op), canonical, mod);
	if (resigned == ptr) {
		*would_fault = false;
		return pac_run_op(op, ptr, mod);   /* correctly signed -> safe to authenticate */
	}
	*would_fault = true;
	return canonical;                      /* wrong sig: a real AUT* here would FPAC-reset the box */
}

/* Crash-safe AUTH for the REVISOR_PAC_AUTH ioctl: a wrong signature must never reach a real AUT* at
 * EL1 (synchronous FPAC fault -> hard reset). Verifies first (pac_auth_verify_core) inside the same
 * one-op key window as pac_run_op_with_keys, and warns when a fault was suppressed. */
uint64_t pac_auth_with_keys(enum pac_op op, uint64_t ptr, uint64_t mod,
                            bool keys_set, const struct pac_keys *exec_keys) {
	struct pac_keys saved_keys;
	uint64_t saved_sctlr, result;
	unsigned long flags;
	bool would_fault = false;

	if (!keys_set) {
		result = pac_auth_verify_core(op, ptr, mod, &would_fault);
	} else {
		local_irq_save(flags);
		preempt_disable();

		pac_save_keys(&saved_keys);
		pac_load_keys(exec_keys);
		saved_sctlr = pac_enable_all_keys();

		result = pac_auth_verify_core(op, ptr, mod, &would_fault);

		pac_restore_sctlr(saved_sctlr);
		pac_load_keys(&saved_keys);

		preempt_enable();
		local_irq_restore(flags);
	}

	if (would_fault) {
		pr_warn("revizor: PAC AUTH would FPAC (op=%d ptr=%#llx ctx=%#llx keys_set=%d) -- suppressed, returned canonical %#llx\n",
		        op, (unsigned long long)ptr, (unsigned long long)mod, keys_set,
		        (unsigned long long)result);
	}
	return result;
}

#pragma GCC pop_options

