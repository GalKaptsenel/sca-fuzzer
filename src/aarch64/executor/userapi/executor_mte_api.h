#ifndef EXECUTOR_MTE_API_H
#define EXECUTOR_MTE_API_H

/*
 * Shared kernel/userspace API for the MTE tag-region ioctl. Included by the kernel module (via
 * chardevice.h) and the userland tool. Uses only standard C99 integer types so it compiles in both.
 */

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
#endif

#include "executor_ioctl_nr.h"
#include "executor_user_api.h"

#define UAPI_MTE_GRANULE_SIZE	16
/* The contiguous taggable block: lower_overflow | main | faulty | upper_overflow. */
#define MTE_TAGGABLE_BYTES	(2 * UAPI_OVERFLOW_REGION_SIZE + UAPI_MAIN_REGION_SIZE + UAPI_FAULTY_REGION_SIZE)
#define MTE_TAG_MAX_GRANULES	(MTE_TAGGABLE_BYTES / UAPI_MTE_GRANULE_SIZE)

/*
 * REVISOR_MTE_TAG_REGION: write per-granule allocation tags to a run of granules starting at
 * sandbox_offset bytes from the base of lower_overflow.
 * sandbox_offset : byte offset from lower_overflow base (granule-aligned).
 * n_granules     : number of 16-byte granules to tag (<= MTE_TAG_MAX_GRANULES; the run must stay
 *                  within the taggable block).
 * tags           : one 4-bit tag per granule, one tag per byte (bits[3:0]).
 */
struct mte_tag_region_req {
    uint64_t sandbox_offset;
    uint64_t n_granules;
    uint8_t  tags[MTE_TAG_MAX_GRANULES];
};

#define REVISOR_MTE_TAG_REGION		    _IOW(REVISOR_IOC_MAGIC, REVISOR_MTE_TAG_REGION_CONSTANT, struct mte_tag_region_req)

#endif // EXECUTOR_MTE_API_H
