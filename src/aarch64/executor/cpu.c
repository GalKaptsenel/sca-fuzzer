#include "main.h"

void get_cpu_info(void *info) {
	struct aarch64_cpu_info* out = (struct aarch64_cpu_info*)info;
	out->cpu_id = smp_processor_id();
	out->mpidr_el1 = read_sysreg(mpidr_el1);
	out->midr_el1 = read_sysreg(midr_el1);
	out->ctr_el0 = read_sysreg(ctr_el0);
}
EXPORT_SYMBOL(get_cpu_info);

int execute_on_pinned_cpu(int target_cpu, void (*fn)(void *), void *arg) {
	int result = 0;

	if(smp_processor_id() == target_cpu || CPU_ID_DEFAULT == target_cpu) {
		get_cpu();
		fn(arg);
		put_cpu();
		return 0;
	}

	if (target_cpu < 0 || target_cpu >= nr_cpu_ids || !cpu_online(target_cpu)) {
		module_err("Target CPU %d is invalid or offline\n", target_cpu);
		result = -EINVAL;
		goto execute_on_pinned_cpu_out;
	}
	
	result = smp_call_function_single(target_cpu, fn, arg, 1);
	
execute_on_pinned_cpu_out:
	return result;
}
EXPORT_SYMBOL(execute_on_pinned_cpu);
