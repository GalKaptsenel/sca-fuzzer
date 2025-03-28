#ifndef EXECUTOR_USERLAND_CHARDEV_H
#define EXECUTOR_USERLAND_CHARDEV_H

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





#endif // EXECUTOR_USERLAND_CHARDEV_H
