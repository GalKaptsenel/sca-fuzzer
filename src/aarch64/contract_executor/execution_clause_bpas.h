#ifndef EXECUTION_CLAUSE_BPAS_H
#define EXECUTION_CLAUSE_BPAS_H

#include "execution_clauses.h"

/* Speculative store bypass: let a store execute, then on the next instruction undo its memory
 * write so the speculative window reads the stale value. */
extern const struct execution_clause_descriptor bpas_execution_clause;

#endif // EXECUTION_CLAUSE_BPAS_H
