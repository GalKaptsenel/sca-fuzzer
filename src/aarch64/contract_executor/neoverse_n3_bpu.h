#ifndef NEOVERSE_N3_BPU_H
#define NEOVERSE_N3_BPU_H

#include "branch_predictor.h"

/* Neoverse-N3 branch predictor (TAGE), exposed as a branch_predictor for injection into the
 * BPU execution clause. The low-level model lives in neoverse_n3_bpu.c (backed by tage_py). */
extern const struct branch_predictor neoverse_n3_bpu;

#endif // NEOVERSE_N3_BPU_H
