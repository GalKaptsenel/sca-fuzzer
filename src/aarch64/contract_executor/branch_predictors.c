#include "branch_predictors.h"
#include "simulation_input.h"   /* enum branch_predictor_id */
#include "neoverse_n3_bpu.h"

const struct branch_predictor* branch_predictor_by_id(uint64_t id) {
	switch (id) {
		case BRANCH_PREDICTOR_NEOVERSE_N3: return &neoverse_n3_bpu;
		default:                           return NULL;   /* NONE / unknown */
	}
}
