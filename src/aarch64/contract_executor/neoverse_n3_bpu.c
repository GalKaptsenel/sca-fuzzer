#include "neoverse_n3_bpu.h"
#include "tage_py.h"

/* Neoverse-N3 branch predictor: a TAGE model implemented in Python (tage_py), exposed here
 * behind the generic branch_predictor interface so the BPU clause can be model-agnostic. */

static int n3_initialized = 0;

static void neoverse_n3_init(void) {
	if (n3_initialized) return;
	if (0 != tagebp_init("src/aarch64/contract_executor", "bootstrap_director")) {
		__builtin_trap(); // sanity check
	}
	n3_initialized = 1;
}

/* Reset TAGE prediction tables in-place between simulations (clears LRU caches and PHR
 * without recreating Python objects). The Python instance lives for the process lifetime. */
static void neoverse_n3_reset(void) {
	if (n3_initialized) tagebp_reset();
}

static int neoverse_n3_predict(uintptr_t pc) {
	return tagebp_predict(pc) ? 1 : 0;
}

static void neoverse_n3_update(uintptr_t pc, int taken, uintptr_t target) {
	tagebp_update(pc, taken, target);
}

const struct branch_predictor neoverse_n3_bpu = {
	.name    = "neoverse-n3-tage",
	.init    = neoverse_n3_init,
	.reset   = neoverse_n3_reset,
	.predict = neoverse_n3_predict,
	.update  = neoverse_n3_update,
};
