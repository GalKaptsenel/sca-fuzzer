#ifndef ARM64_EXECUTOR_CHARDEV_H
#define ARM64_EXECUTOR_CHARDEV_H

/* Shared kernel/userspace definitions for PAC sign/auth ioctls. */
#include "userapi/executor_pac_api.h"

/* ioctl serial numbers (single source of truth). */
#include "userapi/executor_ioctl_nr.h"

/* Self-describing per-input wire format parsed by copy_input_from_user. */
#include "userapi/executor_input_format.h"

/*
 * REVISOR_MTE_TAG_REGION: write a uniform allocation tag to a byte range within
 * the sandbox, starting at sandbox_offset bytes from the base of lower_overflow.
 * sandbox_offset : byte offset from lower_overflow base (must be granule-aligned).
 * length         : number of bytes to tag (granule-aligned, sandbox_offset + length
 *                  <= 2*OVERFLOW_REGION_SIZE + MAIN_REGION_SIZE + FAULTY_REGION_SIZE --
 *                  the contiguous lower_overflow|main|faulty|upper_overflow block).
 * tag            : 4-bit tag value (bits[3:0] used; upper bits ignored).
 */
struct mte_tag_region_req {
    uint64_t sandbox_offset;
    uint64_t length;
    uint8_t  tag;
};

#define REVISOR_CHECKOUT_TEST      	    _IO(REVISOR_IOC_MAGIC, REVISOR_CHECKOUT_TEST_CONSTANT)                   // Can read test case and write test case
#define REVISOR_UNLOAD_TEST    		    _IO(REVISOR_IOC_MAGIC, REVISOR_UNLOAD_TEST_CONSTANT)
#define REVISOR_GET_NUMBER_OF_INPUTS   	_IOR(REVISOR_IOC_MAGIC, REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT, uint64_t)
#define REVISOR_CHECKOUT_INPUT   	    _IOW(REVISOR_IOC_MAGIC, REVISOR_CHECKOUT_INPUT_CONSTANT, uint64_t)      // Can read and write the requested input id
#define REVISOR_ALLOCATE_INPUT      	_IOR(REVISOR_IOC_MAGIC, REVISOR_ALLOCATE_INPUT_CONSTANT, uint64_t)      // Allocate new input slot and returns its id
#define REVISOR_FREE_INPUT		        _IOW(REVISOR_IOC_MAGIC, REVISOR_FREE_INPUT_CONSTANT, uint64_t)	        // Free input id
#define REVISOR_MEASUREMENT		        _IOR(REVISOR_IOC_MAGIC, REVISOR_MEASUREMENT_CONSTANT, measurement_t)    // Returns measurement_t of current input id
#define REVISOR_TRACE			        _IO(REVISOR_IOC_MAGIC, REVISOR_TRACE_CONSTANT)
#define REVISOR_CLEAR_ALL_INPUTS	    _IO(REVISOR_IOC_MAGIC, REVISOR_CLEAR_ALL_INPUTS_CONSTANT)
#define REVISOR_GET_TEST_LENGTH		    _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_TEST_LENGTH_CONSTANT, uint64_t)
#define REVISOR_SET_PAC_KEYS		    _IOW(REVISOR_IOC_MAGIC, REVISOR_SET_PAC_KEYS_CONSTANT, struct pac_keys)
#define REVISOR_GET_PAC_KEYS		    _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_PAC_KEYS_CONSTANT, struct pac_keys)
/* REVISOR_PAC_SIGN (13), REVISOR_PAC_AUTH (14), REVISOR_PAC_XPAC (15) — defined in userapi/executor_pac_api.h */
#define REVISOR_MTE_TAG_REGION		    _IOW(REVISOR_IOC_MAGIC, REVISOR_MTE_TAG_REGION_CONSTANT, struct mte_tag_region_req)

#define REVISOR_DEVICE_NAME		        kernel_module_name
#define REVISOR_DEVICE_CLASS_NAME	    "revisor_device_class"
#define REVISOR_DEVICE_NODE_NAME	    REVISOR_DEVICE_NAME

int initialize_device_interface(void);
void free_device_interface(void);

#endif // ARM64_EXECUTOR_CHARDEV_H
