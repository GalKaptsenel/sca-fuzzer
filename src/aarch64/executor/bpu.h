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
#define INSN_MOVZ_X1_16 0xD2800201U  /* movz x1, #16 */
#define INSN_CBNZ_X0    0xB5000020U  /* cbnz x0, +4   (always TAKEN:     x0=&sandbox≠0) */
#define INSN_CBZ_X0     0xB4000020U  /* cbz  x0, +4   (always NOT-TAKEN: x0=&sandbox≠0) */
#define INSN_SUBS_X1    0xF1000421U  /* subs x1, x1, #1 */
#define INSN_CBNZ_X1    0xB5FFFFC1U  /* cbnz x1, -8   (loops back 2 instrs to CBNZ/CBZ X0) */
#define INSN_NOP        0xD503201FU  /* nop */
#define INSN_RET        0xD65F03C0U  /* ret */

/* =============================================================================
 * Branch training configuration
 * =============================================================================
 *
 * Describes a single branch that should be trained to predict a specific
 * direction before each measurement, guaranteeing a misprediction (and thus
 * a speculative window) on its first architectural execution.
 */
#define MAX_BRANCH_TRAINING_ENTRIES 64

typedef struct {
    uint32_t byte_offset;  /* byte offset of the branch within the test case */
    uint8_t  train_taken;  /* 1 = saturate base predictor to TAKEN
                              0 = saturate base predictor to NOT-TAKEN */
} branch_training_entry_t;

typedef struct {
    branch_training_entry_t entries[MAX_BRANCH_TRAINING_ENTRIES];
    int                     count;
} branch_training_config_t;

/* =============================================================================
 * Core BPU operations (moved from measurement.c)
 * ============================================================================= */

/* Rotate through MAX_MEASUREMENT_VIEWS code views so consecutive inputs
 * exercise different branch PCs, preventing TAGE entries from one input
 * aliasing into the next.  Returns a pointer to the selected view. */
void *invalidate_bpu_entries(void);

/* Flush the PHR (Path History Register) to a deterministic state.
 * 38 iterations: "b 2f" always taken (38 updates) + "b.ne 1b" taken when
 * x0≠0 after subs (37 updates) = 75 total taken-branch updates, which
 * equals the full 300-bit PHR depth (75 records × 4 bits/record). */
void flush_bpu_phr(void);

/* DEBUG: perturb PHR to a random state (1-16 taken branches, random distances). */
void set_phr_random(void);

/* DEBUG: pre-test PHR action when branch training is on. 0=flush, 1=random, 2=none. */
extern int debug_phr_mode;

/* Write [BTI C | CBNZ X0, +4 | RET] at view[loc..loc+2], flush icache, and
 * return &view[loc] as a callable function pointer.  The CBNZ X0 at view[loc+1]
 * trains the base predictor at that PC to TAKEN.  Call the pointer ≤16 times
 * to saturate the counter. */
void *load_training_entry_at(uint32_t *view, size_t loc);

/* =============================================================================
 * Branch training interface
 * ============================================================================= */

/* Parse a "byte_offset:direction,..." text string into *out.
 * Returns the number of entries successfully parsed. */
int parse_branch_training_config(const char *buf, size_t len,
                                 branch_training_config_t *out);

/* Format the currently stored training configuration into buf (sysfs show). */
int format_branch_training_config(char *buf, size_t size);

/* Apply cfg to active_view only.
 * Trains the branch at the exact virtual address used by the upcoming
 * test-case execution, with PRNG-driven PHR perturbation between iterations
 * to isolate the effect to the base PHT.
 * Must be called after load_template() so tc_insert_offset_words is set. */
void __nocfi apply_branch_training(void *active_view,
                                   const branch_training_config_t *cfg);

/* Parse and store as the active training config.  Training is applied lazily
 * by reapply_branch_training() on each measurement, after load_template()
 * has populated tc_insert_offset_words.  An empty string clears the config. */
void __nocfi set_branch_training_config(const char *buf, size_t len);

/* Re-apply the previously stored training config to active_view only.
 * active_view is the code view selected for the upcoming measurement
 * (returned by invalidate_bpu_entries(), or measurement_code_views[0]). */
void __nocfi reapply_branch_training(void *active_view);

#endif /* ARM64_EXECUTOR_BPU_H */
