#ifndef EXECUTOR_PAC_H
#define EXECUTOR_PAC_H
#include <linux/types.h>
#include <linux/percpu.h>

struct pac_keys {
	uint64_t apia_lo, apia_hi;
	uint64_t apib_lo, apib_hi;
	uint64_t apda_lo, apda_hi;
	uint64_t apdb_lo, apdb_hi;
	uint64_t apga_lo, apga_hi;
};

void pac_save_keys(struct pac_keys *out);
void pac_load_keys(const struct pac_keys *keys);
uint64_t pac_enable_all_keys(void);    /* enables EnIA|EnIB|EnDA|EnDB in SCTLR_EL1; returns old SCTLR value */
void     pac_restore_sctlr(uint64_t saved_sctlr);

/*
 * Swap the calling task's user PAC keys to new_keys and save the previous
 * ones in saved_keys.  Updates both the hardware registers (APIB/APDA/APDB/APGA)
 * and task_struct so the change persists across context switches.  APIA is
 * stored in task_struct (so kernel_exit installs it on return to EL0) but is
 * NOT written to the hardware register here — the kernel owns the hardware
 * APIA register (CONFIG_ARM64_PTR_AUTH_KERNEL) and it is unsafe to change it
 * mid-ioctl while kernel return addresses are on the stack.
 */
void pac_swap_user_keys(const struct pac_keys *new_keys, struct pac_keys *saved_keys);

void trigger_pauth_fault(void);
void pauth_restore_cpu(void);
int pauth_init_cpu(
		uint64_t ia_hi, uint64_t ia_lo,
		uint64_t ib_hi, uint64_t ib_lo,
		uint64_t da_hi, uint64_t da_lo,
		uint64_t db_hi, uint64_t db_lo,
		uint64_t ga_hi,  uint64_t ga_lo);

void pauth_restore_all_cpus(void);
int pauth_init_all_cpus(
		uint64_t ia_hi, uint64_t ia_lo,
		uint64_t ib_hi, uint64_t ib_lo,
		uint64_t da_hi, uint64_t da_lo,
		uint64_t db_hi, uint64_t db_lo,
		uint64_t ga_hi,  uint64_t ga_lo);

uint64_t pacia(uint64_t ptr, uint64_t mod);
uint64_t pacib(uint64_t ptr, uint64_t mod);
uint64_t pacda(uint64_t ptr, uint64_t mod);
uint64_t pacdb(uint64_t ptr, uint64_t mod);
uint64_t pacga(uint64_t x, uint64_t y);
uint64_t paciza(uint64_t ptr);
uint64_t pacizb(uint64_t ptr);
uint64_t pacdza(uint64_t ptr);
uint64_t pacdzb(uint64_t ptr);
uint64_t autia(uint64_t ptr, uint64_t mod);
uint64_t autib(uint64_t ptr, uint64_t mod);
uint64_t autda(uint64_t ptr, uint64_t mod);
uint64_t autdb(uint64_t ptr, uint64_t mod);
uint64_t autiza(uint64_t ptr);
uint64_t autizb(uint64_t ptr);
uint64_t autdza(uint64_t ptr);
uint64_t autdzb(uint64_t ptr);
uint64_t xpaci(uint64_t ptr);
uint64_t xpacd(uint64_t ptr);


#endif // EXECUTOR_PAC_H
