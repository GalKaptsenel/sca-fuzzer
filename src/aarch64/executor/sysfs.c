#include "main.h"

static ssize_t number_of_inputs_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return sprintf(buf, "%llu\n", executor.number_of_inputs);
}

static ssize_t warmups_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return sprintf(buf, "%lu\n", executor.config.uarch_reset_rounds);
}

static ssize_t warmups_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
    sscanf(buf, "%lu", &executor.config.uarch_reset_rounds);
    return count;
}

static ssize_t print_sandbox_base_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return sprintf(buf, "%px\n", executor.sandbox.main_region);
}

static ssize_t print_code_base_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return sprintf(buf, "%px\n", executor.test_case);
}

static ssize_t enable_pre_run_flush_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
    unsigned int value = 0;
    sscanf(buf, "%u", &value);
    executor.config.pre_run_flush = (0 == value) ? 0 : 1;
    return count;
}

static ssize_t enable_pre_run_flush_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return sprintf(buf, "%d\n", executor.config.pre_run_flush);
}

static ssize_t measurement_mode_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
	switch(buf[0]) {
		case 'F':
			executor.config.measurement_template = FLUSH_AND_RELOAD_TEMPLATE;
			break;
		case 'P':
			executor.config.measurement_template = PRIME_AND_PROBE_TEMPLATE;
			break;
		default:
			module_err("Invalid measurement mode.. Clearing the measurement method, please chose an existing one!.\n");
			executor.config.measurement_template = UNSET_TEMPLATE;
			return -1;
	}

    return count;
}

static ssize_t measurement_mode_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	int result = 0;

	switch(executor.config.measurement_template) {
		case FLUSH_AND_RELOAD_TEMPLATE:
			result = sprintf(buf, "Flush and Reload (F+R)\n");
			break;
		case PRIME_AND_PROBE_TEMPLATE:
			result = sprintf(buf, "Prime and Probe (P+P)\n");
			break;
		default:
			result = sprintf(buf, "Measurement mode is unset!\n");
	}

	return result;
}

static ssize_t pin_to_core_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	struct aarch64_cpu_info info = { 0 };
	int result = 0;

	if(CPU_ID_DEFAULT == executor.config.pinned_cpu_id) {
		result = sprintf(buf, "Not Pinned\n");
		goto pin_to_core_show_out;
	}


	if(!cpu_online(executor.config.pinned_cpu_id)) {
		result = scnprintf(buf, PAGE_SIZE, "CPU %d is offline\n", executor.config.pinned_cpu_id);
		goto pin_to_core_show_out;
	}

	int err = smp_call_function_single(executor.config.pinned_cpu_id, get_cpu_info, &info, 1);
	if(0 != err) {
		result = scnprintf(buf, PAGE_SIZE, "Failed to run on CPU %d: %d\n", executor.config.pinned_cpu_id, err);
		goto pin_to_core_show_out;
	}

	result = scnprintf(buf, PAGE_SIZE,
                     "CPU ID      : %llu\n"
                     "MPIDR_EL1   : 0x%016llx\n"
                     "MIDR_EL1    : 0x%016llx\n"
                     "CTR_EL0     : 0x%016llx\n",
                     info.cpu_id, info.mpidr_el1, info.midr_el1, info.ctr_el0);

pin_to_core_show_out:
	return result;
}
static ssize_t pin_to_core_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {

	char tmp_buf[32] = { 0 };
	size_t len = count;
	
	if (len >= sizeof(tmp_buf)) {
		len = sizeof(tmp_buf) - 1;
	}
	
	memcpy(tmp_buf, buf, len);
	tmp_buf[len] = '\0';
	
	while (len > 0 && isspace(tmp_buf[len - 1])) {
		tmp_buf[len - 1] = '\0';
		--len;
	}

	int requested_cpu = -1;
	int ret = 0;
	
	ret = kstrtoint(tmp_buf, 10, &requested_cpu);
	if (0 > ret) {
		executor.config.pinned_cpu_id = CPU_ID_DEFAULT;
		module_err("Invalid input, pinning cleared (will run on current CPU)\n");
		return count;
	}
	
	if (CPU_ID_DEFAULT == requested_cpu) {
	    executor.config.pinned_cpu_id = CPU_ID_DEFAULT;
	    module_err("Pin to core cleared, will run on current CPU\n");
	} else if (requested_cpu < 0 || requested_cpu >= nr_cpu_ids || !cpu_online(requested_cpu)) {
	    executor.config.pinned_cpu_id = CPU_ID_DEFAULT;
	    module_err("Requested CPU %d is invalid or offline, pinning cleared (will run on current CPU)\n", requested_cpu);
	} else {
	    executor.config.pinned_cpu_id = requested_cpu;
	    module_info("Pinned execution to CPU %d\n", requested_cpu);
	}
	
	return count;
}

static struct kobj_attribute number_of_inputs_attribute = __ATTR(number_of_inputs, 0444, number_of_inputs_show, NULL);
static struct kobj_attribute warmups_attribute = __ATTR(warmups, 0666, warmups_show, warmups_store);
static struct kobj_attribute print_sandbox_base_attribute = __ATTR(print_sandbox_base, 0444, print_sandbox_base_show, NULL);
static struct kobj_attribute print_code_base_attribute = __ATTR(print_code_base, 0444, print_code_base_show, NULL);
static struct kobj_attribute enable_pre_run_flush_attribute = __ATTR(enable_pre_run_flush, 0666, enable_pre_run_flush_show, enable_pre_run_flush_store);
static struct kobj_attribute measurement_mode_attribute = __ATTR(measurement_mode, 0666, measurement_mode_show, measurement_mode_store);
static struct kobj_attribute pin_to_core_attribute = __ATTR(pin_to_core, 0666, pin_to_core_show, pin_to_core_store);

static struct attribute *sysfs_attributes[] = {
	&number_of_inputs_attribute.attr,
	&warmups_attribute.attr,
	&print_sandbox_base_attribute.attr,
	&print_code_base_attribute.attr,
	&enable_pre_run_flush_attribute.attr,
	&measurement_mode_attribute.attr,
	&pin_to_core_attribute.attr, 
	NULL, /* need to NULL terminate the list of attributes */
};

static struct kobject *kobj_interface = NULL;

int initialize_sysfs(void) {
	int err = 0;

	// Create a pseudo file system interface
	kobj_interface = kobject_create_and_add(SYSFS_DIRNAME, kernel_kobj->parent);
	if (NULL == kobj_interface) {
		module_err("Failed to create a sysfs directory\n");
		err = -ENOMEM;
		goto sysfs_init_failed_execution;
	}

	{
		struct attribute *attr = NULL;
		int i = 0;
		for (i = 0, attr = sysfs_attributes[i];
			       	(!err) && attr;
				++i, attr = sysfs_attributes[i]) {
			err = sysfs_create_file(kobj_interface, attr);
		}
	}

	if (err) {
		module_err("Failed to create a sysfs group\n");
		goto sysfs_init_failed_remove_directory;
	}

	return 0;

sysfs_init_failed_remove_directory:
	kobject_put(kobj_interface);
	kobj_interface = NULL;

sysfs_init_failed_execution:
	return err;
}
EXPORT_SYMBOL(initialize_sysfs);

void free_sysfs(void) {

	if(kobj_interface) {
		kobject_put(kobj_interface);
		kobj_interface = NULL;
	}
}
EXPORT_SYMBOL(free_sysfs);

