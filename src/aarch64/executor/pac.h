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

enum pac_op {
	PAC_OP_PACIA, PAC_OP_PACIB, PAC_OP_PACDA, PAC_OP_PACDB, PAC_OP_PACGA,
	PAC_OP_PACIZA, PAC_OP_PACIZB, PAC_OP_PACDZA, PAC_OP_PACDZB,
	PAC_OP_AUTIA, PAC_OP_AUTIB, PAC_OP_AUTDA, PAC_OP_AUTDB,
	PAC_OP_AUTIZA, PAC_OP_AUTIZB, PAC_OP_AUTDZA, PAC_OP_AUTDZB,
	PAC_OP_INVALID,
};

/* Mnemonic -> op, restricted to the caller's category (sign ops incl. PACGA, or
 * auth ops); PAC_OP_INVALID for anything outside that category. */
enum pac_op pac_sign_op_from_mnemonic(const char *m);
enum pac_op pac_auth_op_from_mnemonic(const char *m);

/* Execute one PAC/AUT op. When keys_set, the executor's deterministic keys are
 * swapped in for the single op inside an IRQ/preempt-disabled, pac-ret-free window
 * (see pac.c); otherwise the live kernel keys are used. */
uint64_t pac_run_op_with_keys(enum pac_op op, uint64_t ptr, uint64_t mod,
                              bool keys_set, const struct pac_keys *exec_keys);

void pac_save_keys(struct pac_keys *out);
void pac_load_keys(const struct pac_keys *keys);
uint64_t pac_enable_all_keys(void);    /* enables EnIA|EnIB|EnDA|EnDB in SCTLR_EL1; returns old SCTLR value */
void     pac_restore_sctlr(uint64_t saved_sctlr);

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
