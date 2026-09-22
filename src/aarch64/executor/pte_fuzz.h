#ifndef ARM64_EXECUTOR_PTE_FUZZ_H
#define ARM64_EXECUTOR_PTE_FUZZ_H

#include "inputs.h"

/*
 * Per-input page-table ENVIRONMENT fuzzing (the non-interference environment axis).
 *
 * An input's REVISOR_SEC_PTE_SETTINGS overrides (input_t.pte_overrides) are applied to the sandbox
 * pages just before the input runs and reverted right after, so a genuine variant (no overrides) and a
 * decoy variant (overrides) run byte-identical code under different page-table state. Decoupled from
 * measurement.c: the run loop only calls apply before the run and revert after.
 */

/*
 * Apply `input`'s PTE overrides to the sandbox pages, saving the originals for revert. A no-op
 * (returns 0) when the input carries none. On any inapplicable override -- unknown page index or level,
 * or a page not mapped by a writable 4K leaf -- it reverts whatever it already applied and returns
 * -errno, leaving the page tables pristine (no silent fallback). Not reentrant: each apply must be
 * paired with a revert before the next apply (the measurement loop is serialized).
 */
int pte_env_apply(const input_t* input);

/* Revert the most recent pte_env_apply. Safe to call when nothing is applied (a no-op). */
void pte_env_revert(void);

#endif // ARM64_EXECUTOR_PTE_FUZZ_H
