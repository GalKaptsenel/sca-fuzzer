#ifndef COMMON_MSG_CONSTANTS_H
#define COMMON_MSG_CONSTANTS_H

#include <stdint.h>

#define RVZRCE_MAGIC ((uint64_t)0x4543525A5652ULL)  /* "RVZRCE" (contract-executor message envelope) */
#define RVZRCE_VERSION ((uint64_t)1ULL)

/* Architectures */
enum sim_arch {
    RVZR_ARCH_X86_64  = 1,
    RVZR_ARCH_AARCH64 = 2,
};

#endif // COMMON_MSG_CONSTANTS_H
