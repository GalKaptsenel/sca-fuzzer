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
#define REVISOR_SWAP_PAC_KEYS_CONSTANT          12
#define REVISOR_GET_EXEC_PAC_KEYS_CONSTANT      13
#define REVISOR_SET_PAC_KEYS_CONSTANT           14
#define REVISOR_GET_PAC_KEYS_CONSTANT           15
#define REVISOR_MTE_TAG_REGION_CONSTANT         16

/*
 * REVISOR_SWAP_PAC_KEYS: atomically replace the calling task's user PAC keys.
 * in_keys  : keys to install (hardware + task_struct).
 * out_keys : previous keys (to pass back for restoration).
 * APIA is written to task_struct but NOT to the hardware register (the kernel
 * owns the hardware APIA for CONFIG_ARM64_PTR_AUTH_KERNEL); kernel_exit
 * installs APIA from task_struct on return to EL0.
 */
struct pac_keys_swap_req {
    struct pac_keys in_keys;
    struct pac_keys out_keys;
};

/*
 * REVISOR_GET_EXEC_PAC_KEYS: retrieve the keys that the executor will use for
 * signing.  If executor.config.pac_keys_set is true, returns those keys and
 * sets use_swap=1 (caller must swap to these keys before EL0 PAC ops).
 * Otherwise returns current hardware keys and sets use_swap=0 (no swap
 * needed: current EL0 keys already match the executor's signing keys).
 */
struct pac_exec_keys_info {
    struct pac_keys keys;
    uint8_t         use_swap;
};

/*
 * REVISOR_MTE_TAG_REGION: write a uniform allocation tag to a byte range within
 * the sandbox, starting at sandbox_offset bytes from the base of main_region.
 * sandbox_offset : byte offset from main_region base (must be granule-aligned).
 * length         : number of bytes to tag (granule-aligned,
 *                  sandbox_offset + length <= MAIN_REGION_SIZE + FAULTY_REGION_SIZE).
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
#define REVISOR_BATCHED_INPUTS		    _IOWR(REVISOR_IOC_MAGIC, REVISOR_BATCHED_INPUTS_CONSTANT, struct input_batch*)
#define REVISOR_SET_PAC_KEYS		    _IOW(REVISOR_IOC_MAGIC, REVISOR_SET_PAC_KEYS_CONSTANT, struct pac_keys)
#define REVISOR_GET_PAC_KEYS		    _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_PAC_KEYS_CONSTANT, struct pac_keys)
#define REVISOR_SWAP_PAC_KEYS		    _IOWR(REVISOR_IOC_MAGIC, REVISOR_SWAP_PAC_KEYS_CONSTANT, struct pac_keys_swap_req)
#define REVISOR_GET_EXEC_PAC_KEYS	    _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_EXEC_PAC_KEYS_CONSTANT, struct pac_exec_keys_info)
#define REVISOR_MTE_TAG_REGION		    _IOW(REVISOR_IOC_MAGIC, REVISOR_MTE_TAG_REGION_CONSTANT, struct mte_tag_region_req)

#define REVISOR_DEVICE_NAME		        kernel_module_name
#define REVISOR_DEVICE_CLASS_NAME	    "revisor_device_class"
#define REVISOR_DEVICE_NODE_NAME	    REVISOR_DEVICE_NAME

int initialize_device_interface(void);
void free_device_interface(void);

long revisor_ioctl(struct file* file, unsigned int cmd, unsigned long arg);

#endif // ARM64_EXECUTOR_CHARDEV_H
