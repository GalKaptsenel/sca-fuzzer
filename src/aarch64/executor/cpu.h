#ifndef ARM64_EXECUTOR_CPU_H
#define ARM64_EXECUTOR_CPU_H


#define DEFINE_READ_MSR(msr)						\
	static inline uint64_t read_##msr(void) {			\
		uint64_t val;						\
		asm volatile("mrs %0, " #msr : "=r" (val));		\
		return val;						\
	}

#define DEFINE_WRITE_MSR(msr)						\
	static inline void write_##msr(uint64_t val) {			\
		asm volatile("msr " #msr ", %0" :: "r" (val));		\
		asm volatile("isb" ::: "memory");			\
	}

#define DEFINE_MSR_ACCESSORS(msr)					\
	DEFINE_READ_MSR(msr)						\
	DEFINE_WRITE_MSR(msr)

#define DEFINE_SET_BIT_FUNC(msr, field, shift)					\
	enum { field##_BIT_SHIFT = shift };					\
	static inline uint8_t set_##field##_bit(uint8_t value) {		\
		uint64_t reg_val = read_##msr();				\
		uint8_t prev = (reg_val & (1UL << field##_BIT_SHIFT)) != 0;	\
										\
		if (value) {							\
			reg_val |= (1UL << field##_BIT_SHIFT);			\
		}								\
		else {								\
			reg_val &= ~(1UL << field##_BIT_SHIFT);			\
		}								\
										\
		write_##msr(reg_val);						\
		return prev;							\
}

#define DEFINE_FULL_MSR_BIT_ACCESSORS(msr, field, shift)			\
	DEFINE_MSR_ACCESSORS(msr)						\
	DEFINE_SET_BIT_FUNC(msr, field, shift)					\

struct aarch64_cpu_info {
	uint64_t cpu_id;      // CPU index (logical)
	uint64_t mpidr_el1;   // Affinity info (core/cluster ID)
	uint64_t midr_el1;    // CPU model ID
	uint64_t ctr_el0;     // Cache type register
};

void get_cpu_info(void *info);
int execute_on_pinned_cpu(int target_cpu, void (*fn)(void *), void *arg);

#endif // ARM64_EXECUTOR_CPU_H
