#include "main.h"
#include "pmu.h"

static ssize_t warmups_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return scnprintf(buf, PAGE_SIZE,"%lu\n", executor.config.uarch_reset_rounds);
}

static ssize_t warmups_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
    long value = 0;
    if (kstrtol(buf, 10, &value) || value < 0) {
        module_err("Invalid warmups value (expected a non-negative integer)\n");
        return -EINVAL;
    }
    executor.config.uarch_reset_rounds = value;
    return count;
}

static ssize_t print_sandbox_base_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return scnprintf(buf, PAGE_SIZE,"%px\n", executor.sandbox->main_region);
}

static ssize_t print_code_base_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    /* Address where the test case actually executes inside the JIT'd harness:
     * the view base plus the (constant per template) offset of the test case. */
    char* base = (char*)executor.measurement_code_views[0]
               + current_tc_insert_offset_bytes();
    return scnprintf(buf, PAGE_SIZE,"%px\n", base);
}

static ssize_t va_bits_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    /* Effective kernel-regime (TTBR1) VA size in bits = 64 - TCR_EL1.T1SZ. This is the authority
     * for the PAC VA size: the sandbox runs at EL1 over a TTBR1 pointer, so the auth
     * and strip ops act on the PAC field at bits [54:va_bits]. Reported to userland so a run derives
     * the size from the machine instead of a hard-coded constant (which corrupts pointers on a
     * mismatch). */
    unsigned long long tcr = read_sysreg(tcr_el1);
    unsigned int t1sz = (unsigned int)((tcr >> 16) & 0x3f);
    return scnprintf(buf, PAGE_SIZE, "%u\n", 64u - t1sz);
}

static ssize_t tbi0_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return scnprintf(buf, PAGE_SIZE, "%u\n", (unsigned int)((read_sysreg(tcr_el1) >> 37) & 1));
}

static ssize_t tbi1_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return scnprintf(buf, PAGE_SIZE, "%u\n", (unsigned int)((read_sysreg(tcr_el1) >> 38) & 1));
}

static ssize_t tbid0_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return scnprintf(buf, PAGE_SIZE, "%u\n", (unsigned int)((read_sysreg(tcr_el1) >> 51) & 1));
}

static ssize_t tbid1_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    return scnprintf(buf, PAGE_SIZE, "%u\n", (unsigned int)((read_sysreg(tcr_el1) >> 52) & 1));
}

/* Address-auth algorithm level from the ID registers: ID_AA64ISAR1_EL1.APA[7:4] (QARMA5) or
 * ID_AA64ISAR2_EL1.APA3[15:12] (QARMA3). Level >= 3 => FEAT_PAuth2, >= 4 => FEAT_FPAC. */
static unsigned int pac_apa(void)  { return (unsigned int)((read_sysreg_s(SYS_ID_AA64ISAR1_EL1) >> 4) & 0xf); }
static unsigned int pac_apa3(void) { return (unsigned int)((read_sysreg_s(SYS_ID_AA64ISAR2_EL1) >> 12) & 0xf); }

static ssize_t pac_qarma_version_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    unsigned int v = pac_apa() ? 5u : (pac_apa3() ? 3u : 0u);
    return scnprintf(buf, PAGE_SIZE, "%u\n", v);
}

static ssize_t pac_pauth2_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
    unsigned int level = pac_apa() ? pac_apa() : pac_apa3();
    return scnprintf(buf, PAGE_SIZE, "%u\n", level >= 3u ? 1u : 0u);
}

static ssize_t enable_pre_run_flush_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
    bool value = false;
    if (kstrtobool(buf, &value)) {
        return -EINVAL;
    }
    /* legacy combined knob: drives BOTH independent knobs for back-compat */
    executor.config.pre_run_flush = value;
    executor.config.phr_flush = value;
    executor.config.view_rotation = value;
    return count;
}

static ssize_t enable_pre_run_flush_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE,"%d\n", executor.config.pre_run_flush);
}

static ssize_t enable_phr_flush_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
    bool value = false;
    if (kstrtobool(buf, &value)) {
        return -EINVAL;
    }
    executor.config.phr_flush = value;
    return count;
}
static ssize_t enable_phr_flush_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE,"%d\n", executor.config.phr_flush);
}

static ssize_t enable_ssbs_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
    bool value = false;
    if (kstrtobool(buf, &value)) {
        return -EINVAL;
    }
    executor.config.enable_ssbs = value;
    return count;
}
static ssize_t enable_ssbs_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE,"%d\n", executor.config.enable_ssbs);
}

static ssize_t enable_view_rotation_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
    bool value = false;
    if (kstrtobool(buf, &value)) {
        return -EINVAL;
    }
    executor.config.view_rotation = value;
    return count;
}
static ssize_t enable_view_rotation_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE,"%d\n", executor.config.view_rotation);
}

static ssize_t measurement_mode_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
	if (sysfs_streq(buf, "P+P")) {
		executor.config.measurement_template = PRIME_AND_PROBE_TEMPLATE;
	} else if (sysfs_streq(buf, "F+R")) {
		executor.config.measurement_template = FLUSH_AND_RELOAD_TEMPLATE;
	} else if (sysfs_streq(buf, "P+R")) {
		executor.config.measurement_template = PRIME_AND_RELOAD_TEMPLATE;
	} else {
		module_err("Invalid measurement mode '%s'; expected 'P+P', 'F+R' or 'P+R'\n", buf);
		return -EINVAL;
	}

	invalidate_jit_cache(); // template drives the built harness; force a rebuild
	return count;
}

static ssize_t jit_memoize_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
	bool value = false;
	if (kstrtobool(buf, &value)) {
		return -EINVAL;
	}
	jit_memoize_enabled = value;
	invalidate_jit_cache(); // force one rebuild so the next trace() reflects the new setting
	return count;
}
static ssize_t jit_memoize_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE, "%d\n", jit_memoize_enabled);
}

static ssize_t jit_stats_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	uint64_t skipped = jit_build_calls - jit_build_done;
	uint64_t avg_ns = jit_build_done ? jit_build_ns / jit_build_done : 0;
	return scnprintf(buf, PAGE_SIZE,
	                 "calls   %llu\n"
	                 "builds  %llu\n"
	                 "skipped %llu\n"
	                 "build_ns_total %llu\n"
	                 "build_ns_avg   %llu\n",
	                 jit_build_calls, jit_build_done, skipped, jit_build_ns, avg_ns);
}
static ssize_t jit_stats_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
	jit_build_calls = 0; // any write resets the counters
	jit_build_done = 0;
	jit_build_ns = 0;
	return count;
}

static ssize_t measurement_mode_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	int result = 0;

	switch(executor.config.measurement_template) {
		case FLUSH_AND_RELOAD_TEMPLATE:
			result = scnprintf(buf, PAGE_SIZE,"Flush and Reload (F+R)\n");
			break;
		case PRIME_AND_PROBE_TEMPLATE:
			result = scnprintf(buf, PAGE_SIZE,"Prime and Probe (P+P)\n");
			break;
		case PRIME_AND_RELOAD_TEMPLATE:
			result = scnprintf(buf, PAGE_SIZE,"Prime and Reload (P+R)\n");
			break;
		default:
			result = scnprintf(buf, PAGE_SIZE,"Measurement mode is unset!\n");
	}

	return result;
}

static ssize_t pin_to_core_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	struct aarch64_cpu_info info = { 0 };
	int result = 0;

	if(CPU_ID_DEFAULT == executor.config.pinned_cpu_id) {
		result = scnprintf(buf, PAGE_SIZE,"Not Pinned\n");
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
	
	while (0 < len && isspace(tmp_buf[len - 1])) {
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
	} else if (0 > requested_cpu || requested_cpu >= nr_cpu_ids || !cpu_online(requested_cpu)) {
	    executor.config.pinned_cpu_id = CPU_ID_DEFAULT;
	    module_err("Requested CPU %d is invalid or offline, pinning cleared (will run on current CPU)\n", requested_cpu);
	} else {
	    executor.config.pinned_cpu_id = requested_cpu;
	    module_info("Pinned execution to CPU %d\n", requested_cpu);
	}
	
	return count;
}

static struct kobj_attribute warmups_attribute = __ATTR(warmups, 0644,warmups_show, warmups_store);
static struct kobj_attribute print_sandbox_base_attribute = __ATTR(print_sandbox_base, 0444, print_sandbox_base_show, NULL);
static struct kobj_attribute print_code_base_attribute = __ATTR(print_code_base, 0444, print_code_base_show, NULL);
static struct kobj_attribute va_bits_attribute = __ATTR(va_bits, 0444, va_bits_show, NULL);
static struct kobj_attribute tbi0_attribute = __ATTR(tbi0, 0444, tbi0_show, NULL);
static struct kobj_attribute tbi1_attribute = __ATTR(tbi1, 0444, tbi1_show, NULL);
static struct kobj_attribute tbid0_attribute = __ATTR(tbid0, 0444, tbid0_show, NULL);
static struct kobj_attribute tbid1_attribute = __ATTR(tbid1, 0444, tbid1_show, NULL);
static struct kobj_attribute pac_qarma_version_attribute = __ATTR(pac_qarma_version, 0444, pac_qarma_version_show, NULL);
static struct kobj_attribute pac_pauth2_attribute = __ATTR(pac_pauth2, 0444, pac_pauth2_show, NULL);
static struct kobj_attribute enable_pre_run_flush_attribute = __ATTR(enable_pre_run_flush, 0644,enable_pre_run_flush_show, enable_pre_run_flush_store);
static struct kobj_attribute enable_phr_flush_attribute = __ATTR(enable_phr_flush, 0644,enable_phr_flush_show, enable_phr_flush_store);
static struct kobj_attribute enable_view_rotation_attribute = __ATTR(enable_view_rotation, 0644,enable_view_rotation_show, enable_view_rotation_store);
static struct kobj_attribute enable_ssbs_attribute = __ATTR(enable_ssbs, 0644,enable_ssbs_show, enable_ssbs_store);
static struct kobj_attribute measurement_mode_attribute = __ATTR(measurement_mode, 0644,measurement_mode_show, measurement_mode_store);
static struct kobj_attribute pin_to_core_attribute = __ATTR(pin_to_core, 0644,pin_to_core_show, pin_to_core_store);
static struct kobj_attribute jit_memoize_attribute = __ATTR(jit_memoize, 0644, jit_memoize_show, jit_memoize_store);
static struct kobj_attribute jit_stats_attribute = __ATTR(jit_stats, 0644, jit_stats_show, jit_stats_store);

/* DEBUG: dump the raw per-set L1D_CACHE_REFILL delta captured during the last reload().
 * One line per set: "<set> <main_refills> <faulty_refills>"  (0 == hit / line was resident). */
static ssize_t reload_refills_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	int i = 0;
	ssize_t n = 0;
	/* columns: set  L1main L1faulty  L2main L2faulty   (counter0=reload_pmu_event, counter1=L2D_CACHE_REFILL) */
	for (i = 0; i < 64; ++i) {
		n += scnprintf(buf + n, PAGE_SIZE - n, "%d %llu %llu %llu %llu\n", i,
		               executor.debug_reload_refills[i], executor.debug_reload_refills[64 + i],
		               executor.debug_reload_refills2[i], executor.debug_reload_refills2[64 + i]);
	}
	return n;
}
static struct kobj_attribute reload_refills_attribute = __ATTR(reload_refills, 0444, reload_refills_show, NULL);

/* F+R reload hit test: <0 = L1D_CACHE_REFILL delta==0 (default); >=0 = PMCCNTR cycle delta < (1<<shift). */
static ssize_t reload_timer_shift_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE, "%d\n", executor.config.reload_timer_shift);
}
static ssize_t reload_timer_shift_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
	int v = -1;
	if (0 > kstrtoint(buf, 10, &v)) {
		return -EINVAL;
	}
	executor.config.reload_timer_shift = v;
	invalidate_jit_cache();
	return count;
}
static struct kobj_attribute reload_timer_shift_attribute = __ATTR(reload_timer_shift, 0644, reload_timer_shift_show, reload_timer_shift_store);

/* F+R reload set-visit order: 0 = fixed bit-reversed (default); 1 = fresh random permutation per run. */
static ssize_t reload_random_order_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE, "%d\n", executor.config.reload_random_order);
}
static ssize_t reload_random_order_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
	int v = 0;
	if (0 > kstrtoint(buf, 10, &v)) {
		return -EINVAL;
	}
	executor.config.reload_random_order = (0 != v);
	invalidate_jit_cache();
	return count;
}
static struct kobj_attribute reload_random_order_attribute = __ATTR(reload_random_order, 0644, reload_random_order_show, reload_random_order_store);

/* PMU event id counted by reload()'s hit test (accepts hex). 0x03 = L1D_CACHE_REFILL, 0x17 = L2D_CACHE_REFILL. */
static ssize_t reload_pmu_event_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE, "0x%x\n", executor.config.reload_pmu_event);
}
static ssize_t reload_pmu_event_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
	int v = 0x03;
	if (0 > kstrtoint(buf, 0, &v)) {
		return -EINVAL;
	}
	executor.config.reload_pmu_event = v & 0xffff;
	return count;
}
static struct kobj_attribute reload_pmu_event_attribute = __ATTR(reload_pmu_event, 0644, reload_pmu_event_show, reload_pmu_event_store);

/* F+R reload: 0 = single 64-set sweep (default); 1 = per-set isolation (re-run test per set). */
static ssize_t reload_isolate_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE, "%d\n", executor.config.reload_isolate);
}
static ssize_t reload_isolate_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count) {
	int v = 0;
	if (0 > kstrtoint(buf, 10, &v)) {
		return -EINVAL;
	}
	executor.config.reload_isolate = (0 != v);
	invalidate_jit_cache();
	return count;
}
static struct kobj_attribute reload_isolate_attribute = __ATTR(reload_isolate, 0644, reload_isolate_show, reload_isolate_store);

/* SSBS bit read back during the last measurement: bit12=1 => bypass allowed (mitigation OFF); 0 => mitigated. */
static ssize_t ssbs_state_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	int bit = (executor.debug_ssbs >> 12) & 1;
	return scnprintf(buf, PAGE_SIZE, "SSBS=%d (%s)  raw=0x%llx\n", bit,
	                 bit ? "store-bypass ALLOWED, mitigation OFF" : "mitigated (SSB disabled)",
	                 executor.debug_ssbs);
}
static struct kobj_attribute ssbs_state_attribute = __ATTR(ssbs_state, 0444, ssbs_state_show, NULL);

/* Reload totals: L2D refills (demand+prefetch) vs MEM_ACCESS (demand only). refills > accesses => prefetch. */
static ssize_t reload_traffic_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	uint64_t l2d = executor.debug_reload_total_l2d, mem = executor.debug_reload_total_mem;
	return scnprintf(buf, PAGE_SIZE, "l2d_cache_refill=%llu mem_access=%llu excess(prefetch)=%lld\n",
	                 l2d, mem, (long long)l2d - (long long)mem);
}
static struct kobj_attribute reload_traffic_attribute = __ATTR(reload_traffic, 0444, reload_traffic_show, NULL);

/* ---- system/ subdirectory: host info unrelated to the Revizor control knobs ---- */

static ssize_t pmu_event_counters_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE, "%u\n", pmu_event_counters());
}

static ssize_t measurement_supported_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE, "%d\n", pmu_measurement_supported() ? 1 : 0);
}

static ssize_t cpu_info_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	struct aarch64_cpu_info info = { 0 };
	get_cpu_info(&info);
	return scnprintf(buf, PAGE_SIZE,
	                 "CPU ID      : %llu\n"
	                 "MIDR_EL1    : 0x%016llx\n"
	                 "MPIDR_EL1   : 0x%016llx\n"
	                 "CTR_EL0     : 0x%016llx\n",
	                 info.cpu_id, info.midr_el1, info.mpidr_el1, info.ctr_el0);
}

static ssize_t abi_version_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf) {
	return scnprintf(buf, PAGE_SIZE, "%u\n", REVISOR_EXECUTOR_ABI_VERSION);
}

static struct kobj_attribute pmu_event_counters_attribute = __ATTR(pmu_event_counters, 0444, pmu_event_counters_show, NULL);
static struct kobj_attribute measurement_supported_attribute = __ATTR(measurement_supported, 0444, measurement_supported_show, NULL);
static struct kobj_attribute cpu_info_attribute = __ATTR(cpu_info, 0444, cpu_info_show, NULL);
static struct kobj_attribute abi_version_attribute = __ATTR(abi_version, 0444, abi_version_show, NULL);

static struct attribute *system_attributes[] = {
	&pmu_event_counters_attribute.attr,
	&measurement_supported_attribute.attr,
	&cpu_info_attribute.attr,
	&abi_version_attribute.attr,
	&va_bits_attribute.attr,
	&tbi0_attribute.attr,
	&tbi1_attribute.attr,
	&tbid0_attribute.attr,
	&tbid1_attribute.attr,
	&pac_qarma_version_attribute.attr,
	&pac_pauth2_attribute.attr,
	NULL,
};

static struct attribute *sysfs_attributes[] = {
	&warmups_attribute.attr,
	&print_sandbox_base_attribute.attr,
	&print_code_base_attribute.attr,
	&enable_pre_run_flush_attribute.attr,
	&enable_phr_flush_attribute.attr,
	&enable_view_rotation_attribute.attr,
	&enable_ssbs_attribute.attr,
	&measurement_mode_attribute.attr,
	&pin_to_core_attribute.attr,
	&jit_memoize_attribute.attr,
	&jit_stats_attribute.attr,
	&reload_refills_attribute.attr,
	&reload_timer_shift_attribute.attr,
	&reload_random_order_attribute.attr,
	&reload_pmu_event_attribute.attr,
	&reload_isolate_attribute.attr,
	&ssbs_state_attribute.attr,
	&reload_traffic_attribute.attr,
	NULL, /* need to NULL terminate the list of attributes */
};

static struct kobject *kobj_interface = NULL;
static struct kobject *kobj_system = NULL;

static int create_sysfs_group(struct kobject *parent, const char *name,
                              struct attribute **attrs, struct kobject **out) {
	struct kobject *kobj = kobject_create_and_add(name, parent);
	int i = 0;

	if (NULL == kobj) {
		return -ENOMEM;
	}
	for (i = 0; NULL != attrs[i]; ++i) {
		int err = sysfs_create_file(kobj, attrs[i]);
		if (0 != err) {
			kobject_put(kobj);
			return err;
		}
	}
	*out = kobj;
	return 0;
}

void free_sysfs(void) {
	if (NULL != kobj_system)    { kobject_put(kobj_system);    kobj_system = NULL; }
	if (NULL != kobj_interface) { kobject_put(kobj_interface); kobj_interface = NULL; }
}

int initialize_sysfs(void) {
	struct attribute *attr = NULL;
	int err = 0;
	int i = 0;

	kobj_interface = kobject_create_and_add(SYSFS_DIRNAME, kernel_kobj->parent);
	if (NULL == kobj_interface) {
		module_err("Failed to create a sysfs directory\n");
		return -ENOMEM;
	}

	for (i = 0, attr = sysfs_attributes[i];
	     (0 == err) && (NULL != attr);
	     ++i, attr = sysfs_attributes[i]) {
		err = sysfs_create_file(kobj_interface, attr);
	}
	if (0 == err) {
		err = create_sysfs_group(kobj_interface, "system", system_attributes, &kobj_system);
	}
	if (0 != err) {
		module_err("Failed to create a sysfs group\n");
		free_sysfs();
		return err;
	}

	return 0;
}

