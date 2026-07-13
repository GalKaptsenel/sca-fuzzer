#ifndef EXECUTOR_PAC_API_H
#define EXECUTOR_PAC_API_H

/*
 * Shared kernel/userspace API for the PAC sign and auth ioctls.
 * Included by the kernel module (via chardevice.h) and the contract executor
 * (pac_sign_plugin.c).  Uses only standard C99 integer types so it compiles
 * in both environments without modification.
 */

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
#endif

/* ioctl numbers + magic come from the single source of truth. */
#include "executor_ioctl_nr.h"

/*
 * PAC key set. Userspace mirror of the kernel's struct pac_keys (executor/pac.h); defined here only
 * for userspace so it does not clash with the kernel definition. Keys are carried per-request in
 * struct pac_sign_req and per-input in the REIF PAC_KEYS section — the kernel stores none of its own.
 */
#ifndef __KERNEL__
struct pac_keys {
    uint64_t apia_lo, apia_hi;
    uint64_t apib_lo, apib_hi;
    uint64_t apda_lo, apda_hi;
    uint64_t apdb_lo, apdb_hi;
    uint64_t apga_lo, apga_hi;
};
_Static_assert(sizeof(struct pac_keys) == 80, "pac_keys ABI: 5 keys * {lo,hi} = 80 bytes");
#endif

/*
 * REVISOR_PAC_SIGN / REVISOR_PAC_AUTH
 *
 * Execute a PAC sign or auth instruction at EL1 under the request's `keys`, loaded for this one
 * instruction and restored afterwards. The keys travel with the request; the kernel keeps no
 * PAC-key state of its own, so nothing can leak across tests or campaigns. keys_present MUST be set
 * for sign/auth — a request without it is rejected (-EINVAL), never signed under a default. XPAC is
 * key-independent and ignores both fields.
 *
 * ptr          : raw pointer to sign, or signed pointer to authenticate.
 * ctx          : context/modifier (ignored for zero-context *IZ* / *DZ* variants).
 * mnemonic     : NUL-terminated — one of:
 *                  sign: pacia pacib pacda pacdb paciza pacizb pacdza pacdzb
 *                  auth: autia autib autda autdb autiza autizb autdza autdzb
 * result       : filled in by the kernel on return.
 * keys_present : nonzero iff `keys` is valid; required for sign/auth.
 * keys         : the key set to sign/auth under.
 *
 * AUTH note: on FEAT_FPAC hardware a failed AUTH at EL1 triggers a synchronous
 * exception (kernel oops).  Only call PAC_AUTH with a correctly-signed pointer.
 */
struct pac_sign_req {
    uint64_t ptr;
    uint64_t ctx;
    char     mnemonic[16];
    uint64_t result;
    uint64_t keys_present;
    struct pac_keys keys;
};

#define REVISOR_PAC_SIGN  _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_SIGN_CONSTANT, struct pac_sign_req)
#define REVISOR_PAC_AUTH  _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_AUTH_CONSTANT, struct pac_sign_req)
#define REVISOR_PAC_XPAC  _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_XPAC_CONSTANT, struct pac_sign_req)

#endif /* EXECUTOR_PAC_API_H */
