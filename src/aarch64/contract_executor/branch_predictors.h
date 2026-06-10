#ifndef BRANCH_PREDICTORS_H
#define BRANCH_PREDICTORS_H

#include <stdint.h>
#include "branch_predictor.h"

/* Resolve an `enum branch_predictor_id` to its predictor; NULL for NONE or an unknown id. */
const struct branch_predictor* branch_predictor_by_id(uint64_t id);

#endif // BRANCH_PREDICTORS_H
