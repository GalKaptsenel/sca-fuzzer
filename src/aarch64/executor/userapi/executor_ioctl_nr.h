#ifndef EXECUTOR_IOCTL_NR_H
#define EXECUTOR_IOCTL_NR_H

/*
 * Canonical ioctl serial numbers for /dev/executor — the single source of truth.
 * Included by the kernel module (chardevice.h), the contract executor
 * (executor_pac_api.h) and the executor_userland tool. The Python mirror in
 * src/aarch64/aarch64_kernel.py must be kept in sync with these.
 *
 * Numbers are consecutive: do not leave gaps when adding or removing ioctls.
 * The _IO*() request macros that bind these numbers to argument types live with
 * their consumers (kernel/userspace types differ), but the numbers stay here.
 */
#define REVISOR_IOC_MAGIC                       'r'

#define REVISOR_CHECKOUT_TEST_CONSTANT          1
#define REVISOR_UNLOAD_TEST_CONSTANT            2
#define REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT   3
#define REVISOR_CHECKOUT_INPUT_CONSTANT         4
#define REVISOR_ALLOCATE_INPUT_CONSTANT         5
#define REVISOR_FREE_INPUT_CONSTANT             6
#define REVISOR_MEASUREMENT_CONSTANT            7
#define REVISOR_TRACE_CONSTANT                  8
#define REVISOR_CLEAR_ALL_INPUTS_CONSTANT       9
#define REVISOR_GET_TEST_LENGTH_CONSTANT        10
/* PAC ioctls (keys travel per-request in struct pac_sign_req; the kernel keeps no key state). */
#define REVISOR_PAC_SIGN_CONSTANT               11
#define REVISOR_PAC_AUTH_CONSTANT               12
#define REVISOR_PAC_XPAC_CONSTANT               13
/* MTE ioctls. */
#define REVISOR_MTE_TAG_REGION_CONSTANT         14

#endif // EXECUTOR_IOCTL_NR_H
