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
 * Matches struct pac_keys in executor/pac.h.
 * Named ce_pac_keys to avoid clashing with the kernel definition when this
 * header is referenced alongside kernel headers.
 */
#ifndef __KERNEL__
struct ce_pac_keys {
    uint64_t apia_lo, apia_hi;
    uint64_t apib_lo, apib_hi;
    uint64_t apda_lo, apda_hi;
    uint64_t apdb_lo, apdb_hi;
    uint64_t apga_lo, apga_hi;
};
#define REVISOR_GET_PAC_KEYS _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_PAC_KEYS_CONSTANT, struct ce_pac_keys)
/* Pass a struct ce_pac_keys to pin the exec keys; pass NULL to revert to the live kernel keys. */
#define REVISOR_SET_PAC_KEYS _IOW(REVISOR_IOC_MAGIC, REVISOR_SET_PAC_KEYS_CONSTANT, struct ce_pac_keys)
#endif

/*
 * REVISOR_PAC_SIGN / REVISOR_PAC_AUTH
 *
 * Execute a PAC sign or auth instruction at EL1 with the live kernel key
 * registers.  If executor.config.pac_keys_set is true the kernel loads the
 * configured exec keys before the instruction and restores afterwards,
 * matching the pac_load_keys / pac_restore pattern used during TC execution.
 *
 * ptr      : raw pointer to sign, or signed pointer to authenticate.
 * ctx      : context/modifier (ignored for zero-context *IZ* / *DZ* variants).
 * mnemonic : NUL-terminated — one of:
 *              sign: pacia pacib pacda pacdb paciza pacizb pacdza pacdzb
 *              auth: autia autib autda autdb autiza autizb autdza autdzb
 * result   : filled in by the kernel on return.
 *
 * AUTH note: on FEAT_FPAC hardware a failed AUTH at EL1 triggers a synchronous
 * exception (kernel oops).  Only call PAC_AUTH with a correctly-signed pointer.
 */
struct pac_sign_req {
    uint64_t ptr;
    uint64_t ctx;
    char     mnemonic[16];
    uint64_t result;
};

#define REVISOR_PAC_SIGN  _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_SIGN_CONSTANT, struct pac_sign_req)
#define REVISOR_PAC_AUTH  _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_AUTH_CONSTANT, struct pac_sign_req)
#define REVISOR_PAC_XPAC  _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_XPAC_CONSTANT, struct pac_sign_req)

#endif /* EXECUTOR_PAC_API_H */
