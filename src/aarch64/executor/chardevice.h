#ifndef ARM64_EXECUTOR_CHARDEV_H
#define ARM64_EXECUTOR_CHARDEV_H

#define REVISOR_IOC_MAGIC 'r'

#define REVISOR_CHECKOUT_TEST_CONSTANT		    1
#define REVISOR_UNLOAD_TEST_CONSTANT		    2
#define REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT	3
#define REVISOR_CHECKOUT_INPUT_CONSTANT		    4
#define REVISOR_ALLOCATE_INPUT_CONSTANT		    5
#define REVISOR_FREE_INPUT_CONSTANT		        6
#define REVISOR_MEASUREMENT_CONSTANT		    7
#define REVISOR_TRACE_CONSTANT			        8
#define REVISOR_CLEAR_ALL_INPUTS_CONSTANT	    9
#define REVISOR_GET_TEST_LENGTH_CONSTANT	    10
#define REVISOR_BATCHED_INPUTS_CONSTANT    	    11
#define REVISOR_PAC_SIGN_CONSTANT               12
#define REVISOR_PAC_AUTH_CONSTANT               13

/*
 * Shared request structure for REVISOR_PAC_SIGN / REVISOR_PAC_AUTH.
 * Layout must match the userspace copy in pac_sign_plugin.c.
 * mnemonic (sign): NUL-terminated "pacia"|"pacib"|"pacda"|"pacdb"|
 *                  "paciza"|"pacizb"|"pacdza"|"pacdzb"
 * mnemonic (auth): NUL-terminated "autia"|"autib"|"autda"|"autdb"|
 *                  "autiza"|"autizb"|"autdza"|"autdzb"
 * On return, result holds the kernel-signed/authenticated pointer value.
 */
struct pac_sign_req {
    uint64_t ptr;
    uint64_t ctx;
    char     mnemonic[16];
    uint64_t result;
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
#define REVISOR_BATCHED_INPUTS		    _IOWR(REVISOR_IOC_MAGIC, REVISOR_BATCHED_INPUTS_CONSTANT, struct input_batch*)
#define REVISOR_PAC_SIGN		        _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_SIGN_CONSTANT, struct pac_sign_req)
#define REVISOR_PAC_AUTH		        _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_AUTH_CONSTANT, struct pac_sign_req)

#define REVISOR_DEVICE_NAME		        kernel_module_name
#define REVISOR_DEVICE_CLASS_NAME	    "revisor_device_class"
#define REVISOR_DEVICE_NODE_NAME	    REVISOR_DEVICE_NAME

int initialize_device_interface(void);
void free_device_interface(void);

long revisor_ioctl(struct file* file, unsigned int cmd, unsigned long arg);

#endif // ARM64_EXECUTOR_CHARDEV_H
