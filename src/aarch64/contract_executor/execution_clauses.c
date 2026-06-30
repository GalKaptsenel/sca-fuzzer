#include "execution_clauses.h"
#include "execution_clause_cond.h"
#include "execution_clause_bpas.h"
#include "execution_clause_bpu.h"
#include "simulation_input.h"   /* EXEC_CLAUSE_* */

/* The registry of all execution clauses. Add a clause = add its file + one entry here.
 * Order matters for composition: memory-effect clauses (bpas) must dispatch before
 * control-flow clauses (cond/bpu) so a mispredicted branch nests inside the store-bypass
 * window (its checkpoint then captures the stale memory). */
static const struct execution_clause_descriptor* const REGISTRY[] = {
	&bpas_execution_clause,
	&cond_execution_clause,
	&bpu_execution_clause,
};

int execution_clause_count(void) {
	return (int)(sizeof(REGISTRY) / sizeof(REGISTRY[0]));
}

const struct execution_clause_descriptor* execution_clause_at(int index) {
	/* All callers deref the result; a corrupted frame->owner must trap, not OOB-read the registry. */
	if (index < 0 || index >= execution_clause_count()) __builtin_trap();
	return REGISTRY[index];
}

int execution_clauses_supported(uint64_t clauses) {
	switch (clauses) {
		case 0:                                      /* seq */
		case EXEC_CLAUSE_COND:
		case EXEC_CLAUSE_BPAS:
		case EXEC_CLAUSE_BPU:
		case EXEC_CLAUSE_COND | EXEC_CLAUSE_BPAS:    /* cond-bpas */
			return 1;
		default:
			return 0;
	}
}
