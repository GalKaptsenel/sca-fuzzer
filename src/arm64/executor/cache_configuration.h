#ifndef ARM64_EXECUTOR_CACHE_CONFIG_H
#define ARM64_EXECUTOR_CACHE_CONFIG_H

#include "utils.h"

#ifndef L1D_ASSOCIATIVITY
#warning "Unsupported/undefined L1D associativity. Falling back to 2-way"
#define L1D_ASSOCIATIVITY	(2)
#endif

#ifdef L1D_SIZE_K
#define L1D_SIZE		((L1D_SIZE_K) * (KB))
#else
#warning "Unsupported/undefined L1D size. Falling back to 32KB"
#define L1D_SIZE		(32 * (KB))
#endif

#define L1D_CONFLICT_DISTANCE 	((L1D_SIZE) / (L1D_ASSOCIATIVITY)) 

#endif // ARM64_EXECUTOR_CACHE_CONFIG_H 
