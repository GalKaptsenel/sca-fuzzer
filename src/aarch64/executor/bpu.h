#ifndef ARM64_EXECUTOR_BPU_H
#define ARM64_EXECUTOR_BPU_H

#include <linux/types.h>

/* =============================================================================
 * Instruction encodings for BPU training stubs
 * =============================================================================
 *
 * Training stubs are 3-instruction sequences entered via BLR:
 *
 *   [k-1]  BTI C         — indirect-call landing pad (required with BTI)
 *   [k]    CBNZ/CBZ X0  — training branch at the EXACT target PC
 *   [k+1]  RET           — return after one training execution
 *
 * X0 = &executor.sandbox on entry (caller convention, always non-zero):
 *   CBNZ X0  →  always TAKEN      →  saturates base predictor toward TAKEN
 *   CBZ  X0  →  always NOT-TAKEN  →  saturates base predictor toward NOT-TAKEN
 *
 * Calling the stub 16 times saturates any 4-bit counter (0–15) from any
 * starting value, guaranteeing a misprediction on the next architectural
 * execution in the opposite direction.
 */
#define INSN_BTI_C      0xD503245FU  /* bti c */
#define INSN_CBNZ_X0    0xB5000020U  /* cbnz x0, +4  (always TAKEN) */
#define INSN_CBZ_X0     0xB4000020U  /* cbz  x0, +4  (always NOT-TAKEN) */
#define INSN_RET        0xD65F03C0U  /* ret */

#define MAX_BRANCH_TRAINING_ENTRIES 64

typedef struct {
    uint32_t byte_offset;  /* offset of the branch within the test case */
    uint8_t  train_taken;  /* 1 = train TAKEN, 0 = train NOT-TAKEN */
} branch_training_entry_t;

typedef struct {
    branch_training_entry_t entries[MAX_BRANCH_TRAINING_ENTRIES];
    int                     count;
} branch_training_config_t;

/* Rotate code views so consecutive inputs use different branch PCs (avoids TAGE
 * aliasing between inputs). Returns the selected view. */
void *invalidate_bpu_entries(void);

/* Overwrite the PHR with 75 taken-branch updates (its full depth). */
void flush_bpu_phr(void);

/* Write a TAKEN training stub at view[loc..loc+2] and return it as a callable. */
void *load_training_entry_at(uint32_t *view, size_t loc);

int format_branch_training_config(char *buf, size_t size);

/* Parse and store the active config (empty string clears it); applied per
 * measurement by reapply_branch_training(). */
void __nocfi set_branch_training_config(const char *buf, size_t len);

/* Re-apply the stored config to active_view (the view for the upcoming run). */
void __nocfi reapply_branch_training(void *active_view);

#endif /* ARM64_EXECUTOR_BPU_H */
