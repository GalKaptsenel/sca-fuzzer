#include "main.h"

void get_cpu_info(void *info) {
	struct aarch64_cpu_info* out = (struct aarch64_cpu_info*)info;
	out->cpu_id = smp_processor_id();
	out->mpidr_el1 = read_sysreg(mpidr_el1);
	out->midr_el1 = read_sysreg(midr_el1);
	out->ctr_el0 = read_sysreg(ctr_el0);
}

struct geo_req {
	unsigned int level;
	enum cache_type type;
	struct cache_geometry out;
};

static void read_cache_geometry(void *arg) {
	struct geo_req *r = arg;
	unsigned int ind = (r->type == CACHE_TYPE_INSTRUCTION) ? 1 : 0;
	unsigned long long clidr = read_sysreg(clidr_el1);
	unsigned long long ccsidr;
	int ccidx;

	write_sysreg(((r->level - 1) << 1) | ind, csselr_el1);
	isb();
	ccsidr = read_sysreg(ccsidr_el1);
	ccidx = (int)((read_sysreg(id_aa64mmfr2_el1) >> 20) & 0xf);

	r->out = decode_cache_geometry(clidr, ccsidr, ccidx, r->level, r->type);
	r->out.cpu = smp_processor_id();
}

struct cache_geometry cache_geometry_on_cpu(int cpu, unsigned int level, enum cache_type type) {
	struct geo_req r = { .level = level, .type = type, .out = { .valid = 0, .cpu = -1 } };
	execute_on_pinned_cpu(cpu, read_cache_geometry, &r);
	return r.out;
}

int execute_on_pinned_cpu(int target_cpu, void (*fn)(void *), void *arg) {
	int this_cpu = get_cpu();   // preempt-disable so the CPU read is stable (no DEBUG_PREEMPT / TOCTOU)

	if (this_cpu == target_cpu || CPU_ID_DEFAULT == target_cpu) {
		fn(arg);
		put_cpu();
		return 0;
	}
	put_cpu();

	if (0 > target_cpu || target_cpu >= nr_cpu_ids || !cpu_online(target_cpu)) {
		module_err("Target CPU %d is invalid or offline\n", target_cpu);
		return -EINVAL;
	}

	return smp_call_function_single(target_cpu, fn, arg, 1);
}
