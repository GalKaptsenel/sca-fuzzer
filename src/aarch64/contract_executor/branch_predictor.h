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
	/* update() updates the prediction tables with the resolved outcome at retire only (counters; no
	 * history advance). `target` is for predictors that train on it (indirect branches); a direction
	 * model ignores it. The speculative global history is driven separately by the clause: at each
	 * branch advance() with the direction taken on the current path; when a misprediction opens a
	 * window checkpoint() first, and on recovery rollback() then advance() with the resolved
	 * direction. */
	void (*update)(uintptr_t pc, int taken, uintptr_t target); /* retire: update tables (counters)       */
	void (*advance)(uintptr_t pc, int taken, uintptr_t target);/* advance speculative global history     */
	void (*checkpoint)(void);                                  /* snapshot history (window opens)        */
	void (*rollback)(void);                                    /* restore history (misprediction)        */
};

#endif // BRANCH_PREDICTOR_H
