#ifndef COMMON_MSG_CONSTANTS_H
#define COMMON_MSG_CONSTANTS_H

#define RVZR_MAGIC 0x525A5652u  /* "RVZR" */
#define RVZR_VERSION 1

/* Architectures */
enum sim_arch {
    RVZR_ARCH_X86_64  = 1,
    RVZR_ARCH_AARCH64 = 2,
};

#endif // COMMON_MSG_CONSTANTS_H
