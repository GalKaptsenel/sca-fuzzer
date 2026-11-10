#ifndef ARM64_EXECUTOR_BPU_H
#define ARM64_EXECUTOR_BPU_H

#include <linux/types.h>

/* Rotate code views so consecutive inputs use different branch PCs (avoids TAGE
 * aliasing between inputs). Returns the selected view. */
void *invalidate_bpu_entries(void);

/* Overwrite the PHR with 75 taken-branch updates (its full depth). */
void flush_bpu_phr(void);

#endif /* ARM64_EXECUTOR_BPU_H */
