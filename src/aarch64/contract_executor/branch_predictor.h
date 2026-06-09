#ifndef BRANCH_PREDICTOR_H
#define BRANCH_PREDICTOR_H

#include <stdint.h>

/* A pluggable branch-direction predictor. The (generic) BPU execution clause is driven by
 * one of these; the concrete predictor is dependency-injected, so the clause logic carries
 * no model-specific knowledge. */
struct branch_predictor {
	const char* name;
	void (*init)(void);                                        /* one-time setup (idempotent, optional) */
	void (*reset)(void);                                       /* clear per-simulation state (optional) */
	int  (*predict)(uintptr_t pc);                             /* nonzero => predicted taken            */
	void (*update)(uintptr_t pc, int taken, uintptr_t target); /* train with the resolved outcome       */
};

#endif // BRANCH_PREDICTOR_H
