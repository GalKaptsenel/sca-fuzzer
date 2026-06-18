#include "main.h"
#include "pac.h"
#include <linux/mutex.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 15, 0)
#define CLASS_CREATE(DEV_CLASS, DEV_NAME) class_create(DEV_NAME);
#else
#define CLASS_CREATE(DEV_CLASS, DEV_NAME) class_create(DEV_CLASS, DEV_NAME);
#endif

/* Serializes all device access (ioctl/read/write) against the shared executor
 * state: the inputs rbtree, state machine, and test-case/measurement buffers
 * have no other synchronization. */
static DEFINE_MUTEX(executor_lock);

/*
 * Device-lock safety against faults under the lock.
 *
 * A fault (Oops) inside an ioctl/read/write while holding executor_lock — e.g. a
 * failed PAC AUTH at EL1 on FEAT_FPAC hardware, or a faulting memset over an RX
 * page — kills the faulting task WITHOUT releasing the lock. With a plain
 * mutex_lock() every later caller then blocks forever in uninterruptible (D)
 * state: an unkillable pile-up that can only be cleared by rebooting.
 *
 * We cannot safely force-release a mutex whose holder died in-kernel (its waiters
 * are on the mutex wait_list; the dead holder also pins a module reference, so
 * rmmod won't work either) — and the executor state it was mutating is left
 * half-updated. That fault class genuinely requires a reboot. What we CAN do is
 * stop it from becoming a silent, unkillable deadlock:
 *   (a) wait killably, so a blocked caller can always be killed (never D-state);
 *   (b) if a new caller finds the lock held far longer than any legitimate
 *       operation, conclude the holder faulted/wedged under it, latch a "wedged"
 *       flag, log loudly, and fail fast with -EIO instead of blocking. The user
 *       is told exactly what happened and that a module reload / reboot is needed.
 * This never re-initialises or force-unlocks the mutex (that would corrupt the
 * wait_list and the executor state).
 */
#define EXECUTOR_LOCK_DEADLOCK_MS 60000u   /* far beyond any legitimate operation */
static pid_t         executor_lock_owner;
static unsigned long executor_lock_acquired_jiffies;
static bool          executor_wedged;

static int executor_lock_acquire(const char* who) {
	if (READ_ONCE(executor_wedged)) {
		module_err("%s rejected: device is wedged — a previous operation faulted while "
			   "holding the lock. Reload the module or reboot to recover.\n", who);
		return -EIO;
	}

	if (!mutex_trylock(&executor_lock)) {
		if (time_after(jiffies, READ_ONCE(executor_lock_acquired_jiffies) +
		    msecs_to_jiffies(EXECUTOR_LOCK_DEADLOCK_MS))) {
			WRITE_ONCE(executor_wedged, true);
			module_err("CRITICAL: device lock held by pid %d for >%u ms — the holder is "
				   "wedged, almost certainly because it faulted while holding the lock "
				   "(check dmesg for an Oops/Internal error). Failing fast so callers do "
				   "not deadlock; without this the device would hang until reboot. Reload "
				   "the module or reboot to recover.\n",
				   executor_lock_owner, EXECUTOR_LOCK_DEADLOCK_MS);
			return -EIO;
		}
		if (mutex_lock_killable(&executor_lock)) {
			return -EINTR;   /* killable: a stuck device never produces unkillable waiters */
		}
	}

	executor_lock_owner = current->pid;
	WRITE_ONCE(executor_lock_acquired_jiffies, jiffies);
	return 0;
}

static void executor_lock_release(void) {
	executor_lock_owner = 0;
	mutex_unlock(&executor_lock);
}

static unsigned long copy_to_user_with_access_check(void __user* user_buffer,
 const void* from_buffer, size_t size) {
	if(!access_ok(user_buffer, size)) {
		module_err("Unable to access user buffer for writing!\n");
		return -EFAULT;
	}

	return copy_to_user(user_buffer, from_buffer, size);
}

static unsigned long copy_from_user_with_access_check(void* to_buffer,
 const void __user* user_buffer, size_t size) {
	if(!access_ok(user_buffer, size)) {
		module_err("Unable to access user buffer for reading!\n");
		return -EFAULT;
	}

	return copy_from_user(to_buffer, user_buffer, size);
}

static int load_input_id(int64_t* p_input_id, void __user* arg) {
	unsigned long not_read = 0;

	BUG_ON(NULL == p_input_id);

	not_read = copy_from_user_with_access_check(p_input_id, arg, sizeof(int64_t));

	if(0 > *p_input_id) {
		module_err("Invalid input id! (got input id: %lld)\n", *p_input_id);
	}

	return sizeof(int64_t) - not_read;
}

static void checkout_into_input_id(void __user* arg) {
	int64_t input_id = -1;

	load_input_id(&input_id, arg);

	if(0 <= input_id) {
		input_t* current_input = get_input(input_id);

		if(NULL != current_input) {
			executor.checkout_region = input_id;
		} else {
			module_err("Checkedout an input id that does not exist! (requested input id: %lld)\n",
			 input_id);
		}
	}
}

static void measurements_became_unavailable(void) {
}

static void reset_state_and_region_after_unloading_all_inputs(void) {
	executor.checkout_region = REGION_DEFAULT;

	switch(executor.state) {
		case TRACED_STATE:
			measurements_became_unavailable();
			fallthrough;
		case READY_STATE:
			executor.state = LOADED_TEST_STATE;
			break;
		case LOADED_INPUTS_STATE:
		       executor.state = CONFIGURATION_STATE;
		       break;
		default:
		       break;
	}
}

static void free_input_id(void __user* arg) {
	int64_t input_id = -1;

	load_input_id(&input_id, arg);

	if(0 <= input_id) {
		if(input_id == executor.checkout_region) {
			executor.checkout_region = REGION_DEFAULT;
		}

		remove_input(input_id);

		if(0 == executor.number_of_inputs) {
			reset_state_and_region_after_unloading_all_inputs();
		}
	}
}

static void clear_all_inputs(void) {
	destroy_inputs_db();
	initialize_inputs_db();
	reset_state_and_region_after_unloading_all_inputs();
}

static void kernel_measurement_to_user_measurement(user_measurement_t* um, const measurement_t* km) {
	memset(um, 0, sizeof(*um));
	memcpy(um->htrace, km->htrace, sizeof(um->htrace));
	memcpy(um->pfc, km->pfc, sizeof(um->pfc));
}

static long measure_input_id(void __user* arg) {
	if(TRACED_STATE != executor.state) {
		module_err("Measurements are available only after performing a trace!\n");
		return -EINVAL;
	}

	if(TEST_REGION == executor.checkout_region) {
		module_err("Checkout into the desired input!\n");
		return -EINVAL;
	}

	measurement_t* current_measurement = get_measurement(executor.checkout_region);
	BUG_ON(NULL == current_measurement);

	user_measurement_t umeasurement = { 0 };
	kernel_measurement_to_user_measurement(&umeasurement, current_measurement);

	if (copy_to_user_with_access_check(arg, &umeasurement, sizeof(umeasurement))) {
		module_err("Failed to copy measurement to userspace!\n");
		return -EFAULT;
	}

	return 0;
}

static void unload_test_and_update_state(void) {
	memset(executor.test_case, 0, MAX_TEST_CASE_SIZE);
	memset(executor.measurement_code_views[0], 0, MAX_MEASUREMENT_CODE_SIZE);
    	dsb(ish); // make sure all views are synced
	isb();
	executor.test_case_length = 0;

	switch(executor.state) {
		case TRACED_STATE:
			measurements_became_unavailable();
			fallthrough;
		case READY_STATE:
		      executor.state = LOADED_INPUTS_STATE;
		      break;
		case LOADED_TEST_STATE:
		      executor.state = CONFIGURATION_STATE;
		      break;
		default:
		      break;
	}
}

static void update_state_after_writing_test(void) {
	switch(executor.state) {
		case CONFIGURATION_STATE:
			executor.state = LOADED_TEST_STATE;
			break;
		case LOADED_INPUTS_STATE:
			executor.state = READY_STATE;
			break;
		default:
			break;
	}
}

static ssize_t load_test_and_update_state(const char __user* test, size_t length) {
	if (MAX_TEST_CASE_SIZE < length) {
		module_err("%lu bytes couldnt be written into test memory!, as it is more then %lu!\n", length, MAX_TEST_CASE_SIZE);
		return -ENOMEM;
	}

	if(copy_from_user_with_access_check(executor.test_case, test, length)) {
		module_err("Failed to read entire test case from userspace!");
		return -EFAULT;
	}

	executor.test_case_length = length;

	update_state_after_writing_test();

	return executor.test_case_length;
}

static int execute_result = 0;

static void execute_on_target(void* unused) {
	execute_result = execute();
}

static int trace(void) {
	int full_test_case_size = 0;

	if(READY_STATE != executor.state && TRACED_STATE != executor.state) {
		module_err("In order to trace, please load inputs and test case.\n");
		return -EINVAL;
	}

	full_test_case_size = load_jit_template(executor.test_case_length);
	if (0 > full_test_case_size) {
		module_err("Failed to load test case (error code: %d)\n", full_test_case_size);
		return full_test_case_size;
	}

	int err = execute_on_pinned_cpu(executor.config.pinned_cpu_id, execute_on_target, NULL);
	if(0 != err) {
		module_err("Failed to execute on CPU %d (err: %d)\n", executor.config.pinned_cpu_id, err);
	}

	executor.state = TRACED_STATE;
	return execute_result;
}

static int get_test_length(void __user* arg) {
	if(LOADED_TEST_STATE != executor.state &&
	 READY_STATE != executor.state &&
	  TRACED_STATE != executor.state) {
		module_err("Test has not been loaded!\n");
		return -EINVAL;
	}

	uint64_t size = executor.test_case_length;
	if (copy_to_user_with_access_check(arg, &size, sizeof(size))) {
		return -EFAULT;
	}

	return 0;
}

static void update_state_after_writing_input(void) {
	switch(executor.state) {
		case CONFIGURATION_STATE:
			executor.state = LOADED_INPUTS_STATE;
			break;
		case LOADED_TEST_STATE:
			executor.state = READY_STATE;
			break;
		default:
			break;
	}
}

static int pac_with_exec_keys(bool keys_set, uint64_t *saved_sctlr_out,
                               struct pac_keys *saved_keys_out)
{
	if (!keys_set) {
		return 0;
	}
	pac_save_keys(saved_keys_out);
	pac_load_keys(&executor.config.pac_keys);
	*saved_sctlr_out = pac_enable_all_keys();
	return 1;
}

static void restore_exec_keys(bool keys_set, uint64_t saved_sctlr,
                               const struct pac_keys *saved_keys)
{
	if (!keys_set) {
		return;
	}
	pac_restore_sctlr(saved_sctlr);
	pac_load_keys(saved_keys);
}

static long handle_pac_sign(void __user *arg)
{
	struct pac_sign_req req;
	if (copy_from_user_with_access_check(&req, arg, sizeof(req))) {
		return -EFAULT;
	}
	req.mnemonic[sizeof(req.mnemonic) - 1] = '\0';

	struct pac_keys saved_keys;
	uint64_t saved_sctlr = 0;
	pac_with_exec_keys(executor.config.pac_keys_set, &saved_sctlr, &saved_keys);

	uint64_t r;
	char *m = req.mnemonic;
	if      (!strcmp(m, "pacia"))  { r = pacia(req.ptr, req.ctx); }
	else if (!strcmp(m, "pacib"))  { r = pacib(req.ptr, req.ctx); }
	else if (!strcmp(m, "pacda"))  { r = pacda(req.ptr, req.ctx); }
	else if (!strcmp(m, "pacdb"))  { r = pacdb(req.ptr, req.ctx); }
	else if (!strcmp(m, "pacga"))  { r = pacga(req.ptr, req.ctx); }
	else if (!strcmp(m, "paciza")) { r = paciza(req.ptr); }
	else if (!strcmp(m, "pacizb")) { r = pacizb(req.ptr); }
	else if (!strcmp(m, "pacdza")) { r = pacdza(req.ptr); }
	else if (!strcmp(m, "pacdzb")) { r = pacdzb(req.ptr); }
	else {
		restore_exec_keys(executor.config.pac_keys_set, saved_sctlr, &saved_keys);
		return -EINVAL;
	}

	restore_exec_keys(executor.config.pac_keys_set, saved_sctlr, &saved_keys);
	req.result = r;
	return copy_to_user_with_access_check(arg, &req, sizeof(req)) ? -EFAULT : 0;
}

static long handle_pac_xpac(void __user *arg)
{
	struct pac_sign_req req;
	if (copy_from_user_with_access_check(&req, arg, sizeof(req))) {
		return -EFAULT;
	}
	req.mnemonic[sizeof(req.mnemonic) - 1] = '\0';

	char *m = req.mnemonic;
	uint64_t r;
	if      (!strcmp(m, "xpaci")) { r = xpaci(req.ptr); }
	else if (!strcmp(m, "xpacd")) { r = xpacd(req.ptr); }
	else { return -EINVAL; }

	req.result = r;
	return copy_to_user_with_access_check(arg, &req, sizeof(req)) ? -EFAULT : 0;
}

static long handle_pac_auth(void __user *arg)
{
	struct pac_sign_req req;
	if (copy_from_user_with_access_check(&req, arg, sizeof(req))) {
		return -EFAULT;
	}
	req.mnemonic[sizeof(req.mnemonic) - 1] = '\0';

	struct pac_keys saved_keys;
	uint64_t saved_sctlr = 0;
	pac_with_exec_keys(executor.config.pac_keys_set, &saved_sctlr, &saved_keys);

	uint64_t r;
	char *m = req.mnemonic;
	if      (!strcmp(m, "autia"))  { r = autia(req.ptr, req.ctx); }
	else if (!strcmp(m, "autib"))  { r = autib(req.ptr, req.ctx); }
	else if (!strcmp(m, "autda"))  { r = autda(req.ptr, req.ctx); }
	else if (!strcmp(m, "autdb"))  { r = autdb(req.ptr, req.ctx); }
	else if (!strcmp(m, "autiza")) { r = autiza(req.ptr); }
	else if (!strcmp(m, "autizb")) { r = autizb(req.ptr); }
	else if (!strcmp(m, "autdza")) { r = autdza(req.ptr); }
	else if (!strcmp(m, "autdzb")) { r = autdzb(req.ptr); }
	else {
		restore_exec_keys(executor.config.pac_keys_set, saved_sctlr, &saved_keys);
		return -EINVAL;
	}

	restore_exec_keys(executor.config.pac_keys_set, saved_sctlr, &saved_keys);
	req.result = r;
	return copy_to_user_with_access_check(arg, &req, sizeof(req)) ? -EFAULT : 0;
}

static long do_revisor_ioctl(struct file* file, unsigned int cmd, unsigned long arg) {
	int64_t result = 0;

	if(REVISOR_IOC_MAGIC != _IOC_TYPE(cmd)) {
		return -ENOTTY;
	}

	switch(_IOC_NR(cmd)) {
		case REVISOR_CHECKOUT_TEST_CONSTANT:
			executor.checkout_region = TEST_REGION;
			break;

		case REVISOR_UNLOAD_TEST_CONSTANT:
			unload_test_and_update_state();
			break;

		case REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT:
			result = executor.number_of_inputs;
			result = copy_to_user_with_access_check((void __user*)arg, &result,
			            sizeof(result));
			break;

		case REVISOR_CHECKOUT_INPUT_CONSTANT:
			checkout_into_input_id((void __user*)arg);
			break;

		case REVISOR_ALLOCATE_INPUT_CONSTANT:
			result = allocate_input();
			result = copy_to_user_with_access_check((void __user*)arg, &result, sizeof(result));
			break;

		case REVISOR_FREE_INPUT_CONSTANT:
			free_input_id((void __user*)arg);
			break;

		case REVISOR_MEASUREMENT_CONSTANT:
			result = measure_input_id((void __user*)arg);
			break;

		case REVISOR_TRACE_CONSTANT:
			result = trace();
			break;

		case REVISOR_CLEAR_ALL_INPUTS_CONSTANT:
			clear_all_inputs();
			break;

		case REVISOR_GET_TEST_LENGTH_CONSTANT:
			result = get_test_length((void __user*)arg);
			break;

		case REVISOR_SET_PAC_KEYS_CONSTANT: {
			/* A NULL argument clears the configured keys, so the executor
			 * falls back to the live hardware keys. */
			if (0 == arg) {
				executor.config.pac_keys_set = false;
				return 0;
			}
			if (copy_from_user_with_access_check(&executor.config.pac_keys,
			    (void __user *)arg, sizeof(struct pac_keys))) {
				return -EFAULT;
			}
			executor.config.pac_keys_set = true;
			return 0;
		}

		case REVISOR_GET_PAC_KEYS_CONSTANT: {
			struct pac_keys keys;
			if (executor.config.pac_keys_set) {
				keys = executor.config.pac_keys;
			} else {
				pac_save_keys(&keys);
			}
			return copy_to_user_with_access_check((void __user *)arg, &keys, sizeof(keys))
				? -EFAULT : 0;
		}

		case REVISOR_MTE_TAG_REGION_CONSTANT: {
			struct mte_tag_region_req req;
			if (copy_from_user_with_access_check(&req, (void __user *)arg, sizeof(req))) {
				return -EFAULT;
			}
			/* sandbox_offset is relative to lower_overflow; the taggable span is the contiguous
			 * block lower_overflow|main|faulty|upper_overflow (eviction is the P+P region and is
			 * left untagged). Overflow-safe bounds check (u64 sums could wrap). */
			const u64 taggable = 2 * OVERFLOW_REGION_SIZE + MAIN_REGION_SIZE + FAULTY_REGION_SIZE;
			if (req.length > taggable || req.sandbox_offset > taggable - req.length) {
				return -EINVAL;
			}
			mte_init_sandbox_tags(executor.sandbox->lower_overflow + req.sandbox_offset,
			                      req.length, req.tag & 0xF);
			return 0;
		}

		case REVISOR_PAC_SIGN_CONSTANT:
			return handle_pac_sign((void __user *)arg);

		case REVISOR_PAC_AUTH_CONSTANT:
			return handle_pac_auth((void __user *)arg);

		case REVISOR_PAC_XPAC_CONSTANT:
			return handle_pac_xpac((void __user *)arg);

		default:
			module_err("Invalid IOCTL! Entered default case..\n");
			return -ENOTTY;
	}

	return result;
}

static ssize_t do_revisor_read(struct file* File, char __user* user_buffer,
 size_t count, loff_t* off) {
	int number_of_bytes_to_copy  = 0;
	int not_copied = 0;
	uint64_t total_size = 0;
	void* from_buffer = NULL;

	BUG_ON(NULL == user_buffer);

	if(TEST_REGION == executor.checkout_region) {
		number_of_bytes_to_copy = min(count, (size_t)(executor.test_case_length - *off));
		from_buffer = executor.test_case;
		total_size = executor.test_case_length;
	} else {
		number_of_bytes_to_copy = min(count, (size_t)(sizeof(input_t) - *off));
		from_buffer = get_input(executor.checkout_region);
		total_size = sizeof(input_t);
	}

	if(total_size <= *off) {
		*off = 0;
		return 0; // return EOF
	}

	BUG_ON(NULL == from_buffer);

	not_copied = copy_to_user_with_access_check(user_buffer, from_buffer + *off,
	 number_of_bytes_to_copy);

	*off += (number_of_bytes_to_copy - not_copied);

	return number_of_bytes_to_copy - not_copied;
}

static ssize_t copy_input_from_user_and_update_state(const char __user* user_buffer, size_t count) {
	if(USER_CONTROLLED_INPUT_LENGTH > count) {
	    module_err("Input must be exactly of length USER_CONTROLLED_INPUT_LENGTH(=%lu)!\n",
	    USER_CONTROLLED_INPUT_LENGTH);
	    return -EINVAL;
	}

	void* to_buffer = get_input(executor.checkout_region);
	BUG_ON(NULL == to_buffer);

	if(copy_from_user_with_access_check(to_buffer, user_buffer, USER_CONTROLLED_INPUT_LENGTH)) {
		module_err("Was unable to read entire input from user\n");
		return -EFAULT;
	}

	update_state_after_writing_input();
	return USER_CONTROLLED_INPUT_LENGTH;
}

static ssize_t do_revisor_write(struct file* File, const char __user* user_buffer,
 size_t count, loff_t* off) {
	BUG_ON(NULL == user_buffer);

	if(TEST_REGION == executor.checkout_region) {
		ssize_t ret = load_test_and_update_state(user_buffer, count);
		if (0 > ret) {
			return ret;
		}
		return ret;
	}

	BUG_ON(0 > executor.checkout_region);
	return copy_input_from_user_and_update_state(user_buffer, count);
}

static long revisor_ioctl(struct file* file, unsigned int cmd, unsigned long arg) {
	long ret = executor_lock_acquire("ioctl");
	if (ret) {
		return ret;
	}
	ret = do_revisor_ioctl(file, cmd, arg);
	executor_lock_release();
	return ret;
}

static ssize_t revisor_read(struct file* file, char __user* user_buffer,
 size_t count, loff_t* off) {
	int err = executor_lock_acquire("read");
	ssize_t ret;
	if (err) {
		return err;
	}
	ret = do_revisor_read(file, user_buffer, count, off);
	executor_lock_release();
	return ret;
}

static ssize_t revisor_write(struct file* file, const char __user* user_buffer,
 size_t count, loff_t* off) {
	int err = executor_lock_acquire("write");
	ssize_t ret;
	if (err) {
		return err;
	}
	ret = do_revisor_write(file, user_buffer, count, off);
	executor_lock_release();
	return ret;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = revisor_ioctl,
    .read = revisor_read,
    .write = revisor_write,
};

int initialize_device_interface(void) {
	int status = 0;

	status = alloc_chrdev_region(&executor.device_mgmt.device_number, 0, 1, REVISOR_DEVICE_NAME);
	if (0 > status) {
		module_err("Unable to allocate character device number! Error code: %d\n", status);
		goto initialize_cleanup_exit_with_error;
	}

	cdev_init(&executor.device_mgmt.character_device, &fops);
	executor.device_mgmt.character_device.owner = THIS_MODULE;

	status = cdev_add(&executor.device_mgmt.character_device, executor.device_mgmt.device_number, 1);
	if (0 > status) {
		module_err("Unable to add character device to the system! Error code: %d\n", status);
		goto initialize_cleanup_allocated_device_numbers;
	}

	executor.device_mgmt.device_class = CLASS_CREATE(THIS_MODULE, REVISOR_DEVICE_CLASS_NAME);

	if (IS_ERR(executor.device_mgmt.device_class)) {
		module_err("Unable to create device class\n");
		status = PTR_ERR(executor.device_mgmt.device_class);
		goto initialize_cleanup_cdev_del;
	}

	if (IS_ERR(device_create(executor.device_mgmt.device_class, NULL,
	 executor.device_mgmt.device_number, NULL, REVISOR_DEVICE_NODE_NAME))) {
		module_err("Unable to create device node\n");
		status = -EINVAL;
		goto initialize_cleanup_class_destroy;
	}

	module_info("Registered device number MAJOR: %d, MINOR: %d and created cdev\n",
	 MAJOR(executor.device_mgmt.device_number), MINOR(executor.device_mgmt.device_number));

	return 0;

initialize_cleanup_class_destroy:
	class_destroy(executor.device_mgmt.device_class);
initialize_cleanup_cdev_del:
	cdev_del(&executor.device_mgmt.character_device);
initialize_cleanup_allocated_device_numbers:
	unregister_chrdev_region(executor.device_mgmt.device_number, 1);
initialize_cleanup_exit_with_error:
	return status;
}

void free_device_interface(void) {
    if (executor.device_mgmt.device_class) {
        device_destroy(executor.device_mgmt.device_class, executor.device_mgmt.device_number);
        class_destroy(executor.device_mgmt.device_class);
    }

    if (executor.device_mgmt.character_device.dev) {
        cdev_del(&executor.device_mgmt.character_device);
    }

    if (executor.device_mgmt.device_number) {
        unregister_chrdev_region(executor.device_mgmt.device_number, 1);
    }

    module_info("Unregistered device number MAJOR: %d, MINOR: %d and deleted cdev\n",
     MAJOR(executor.device_mgmt.device_number), MINOR(executor.device_mgmt.device_number));
}
