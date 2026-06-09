#ifndef EXECUTION_CLAUSE_BPU_H
#define EXECUTION_CLAUSE_BPU_H

#include "execution_clauses.h"
#include "branch_predictor.h"

/* Branch misprediction driven by an injected branch predictor (model-agnostic). */
extern const struct execution_clause_descriptor bpu_execution_clause;

/* Dependency-inject the predictor this clause uses. MUST be called (at the composition root)
 * before a run enables EXEC_CLAUSE_BPU; there is no default predictor. */
void bpu_clause_set_predictor(const struct branch_predictor* predictor);

#endif // EXECUTION_CLAUSE_BPU_H
