#ifndef ARM64_EXECUTOR_CACHE_GEOMETRY_H
#define ARM64_EXECUTOR_CACHE_GEOMETRY_H

#include "utils.h"

// L1D size/associativity are detected at runtime (CCSIDR_EL1); this is only the compile-time upper
// bound on the eviction region. A core whose detected L1D exceeds it is rejected at JIT-build time.
#ifndef L1D_SIZE_MAX
#define L1D_SIZE_MAX (128 * (KB))
#endif

enum cache_type {
	CACHE_TYPE_NONE = 0,
	CACHE_TYPE_INSTRUCTION,
	CACHE_TYPE_DATA,
	CACHE_TYPE_UNIFIED,
};

struct cache_geometry {
	unsigned int level;
	enum cache_type type;
	unsigned int size;
	unsigned int assoc;
	unsigned int sets;
	unsigned int line;
	int cpu;
	int valid;
};

// Pure decode of CLIDR_EL1 + CCSIDR_EL1 for `level` (ccidx = ID_AA64MMFR2_EL1.CCIDX != 0). `requested`
// disambiguates DATA vs INSTRUCTION for a separate L1 (CLIDR ctype 3). No I/O; host-unit-testable.
struct cache_geometry decode_cache_geometry(unsigned long long clidr, unsigned long long ccsidr,
                                            int ccidx, unsigned int level, enum cache_type requested);

// Kernel-only: read the geometry on `cpu` via smp_call (implemented in cpu.c).
struct cache_geometry cache_geometry_on_cpu(int cpu, unsigned int level, enum cache_type type);

#endif // ARM64_EXECUTOR_CACHE_GEOMETRY_H
