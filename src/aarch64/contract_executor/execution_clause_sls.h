#ifndef EXECUTION_CLAUSE_SLS_H
#define EXECUTION_CLAUSE_SLS_H

#include "execution_clauses.h"

/* Straight-line speculation: at a branch, also explore the sequential fall-through pc+4. */
extern const struct execution_clause_descriptor sls_execution_clause;

#endif // EXECUTION_CLAUSE_SLS_H
