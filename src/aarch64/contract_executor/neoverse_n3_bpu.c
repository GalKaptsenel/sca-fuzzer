#include "neoverse_n3_bpu.h"
#include "tage_py.h"
#include <stdio.h>
#include <unistd.h>
#include <libgen.h>
#include <limits.h>

/* Neoverse-N3 branch predictor: a TAGE model implemented in Python (tage_py), exposed here
 * behind the generic branch_predictor interface so the BPU clause can be model-agnostic. */

static int n3_initialized = 0;

static void neoverse_n3_init(void) {
	if (n3_initialized) return;
	/* bootstrap_director.py sits next to the executable; resolve its dir from the binary's own
	 * location so the CE works regardless of the current working directory. */
	char exe[PATH_MAX];
	ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
	if (n <= 0) {
		fprintf(stderr, "[ERR] neoverse_n3 init: readlink(/proc/self/exe) failed\n");
		__builtin_trap();
	}
	exe[n] = '\0';
	if (0 != tagebp_init(dirname(exe), "bootstrap_director")) {
		fprintf(stderr, "[ERR] neoverse_n3 init: tagebp_init(bootstrap_director) failed\n");
		__builtin_trap();
	}
	n3_initialized = 1;
}

/* Reset TAGE prediction tables in-place between simulations (clears LRU caches and PHR
 * without recreating Python objects). The Python instance lives for the process lifetime. */
static void neoverse_n3_reset(void) {
	if (n3_initialized) tagebp_reset();
}

static int neoverse_n3_predict(uintptr_t pc) {
	int prediction = tagebp_predict(pc);
	if (prediction < 0) {
		fprintf(stderr, "[ERR] TAGE predict failed for pc=%#lx\n", (unsigned long)pc);
		__builtin_trap();
	}
	return prediction;
}

static void neoverse_n3_update(uintptr_t pc, int taken, uintptr_t target) {
	tagebp_update(pc, taken, target);
}

static void neoverse_n3_advance(uintptr_t pc, int taken, uintptr_t target) {
	tagebp_advance(pc, taken, target);
}

static void neoverse_n3_checkpoint(void) { tagebp_checkpoint(); }
static void neoverse_n3_rollback(void)   { tagebp_rollback(); }
static void neoverse_n3_commit(void)     { tagebp_commit(); }

const struct branch_predictor neoverse_n3_bpu = {
	.name       = "neoverse-n3-tage",
	.init       = neoverse_n3_init,
	.reset      = neoverse_n3_reset,
	.predict    = neoverse_n3_predict,
	.update     = neoverse_n3_update,
	.advance    = neoverse_n3_advance,
	.checkpoint = neoverse_n3_checkpoint,
	.rollback   = neoverse_n3_rollback,
	.commit     = neoverse_n3_commit,
};
