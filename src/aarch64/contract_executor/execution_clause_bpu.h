#ifndef EXECUTION_CLAUSE_BPU_H
#define EXECUTION_CLAUSE_BPU_H

#include "execution_clauses.h"

/* Branch misprediction driven by the predictor the input selects (config.branch_predictor). */
extern const struct execution_clause_descriptor bpu_execution_clause;

#endif // EXECUTION_CLAUSE_BPU_H
