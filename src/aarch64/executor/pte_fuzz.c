#include "main.h"
#include "globals.h"
#include "inputs.h"
#include "sandbox.h"
#include "pagetable.h"
#include "pte_fuzz.h"
#include <linux/errno.h>

/*
 * Saved original descriptors, for revert. The measurement loop is serialized (one apply/revert pair at
 * a time, run with IRQs off on a pinned CPU), so module-global state is safe. Sized to the wire cap.
 */
static struct {
	void* va;
	uint64_t old;
} pte_saved[REVISOR_INPUT_MAX_PTE_OVERRIDES];
static uint32_t pte_saved_n;

/*
 * Map a sandbox page index to its virtual address. The index is the shared contract with the writer's
 * sandbox page map (default: 0 = main_region, 1 = faulty_region). To fuzz more pages, extend this map
 * and the writer's; nothing else changes. Returns NULL for an unknown index.
 */
static void* sandbox_page_va(uint16_t page_index) {
	switch (page_index) {
	case 0:
		return executor.sandbox->main_region;
	case 1:
		return executor.sandbox->faulty_region;
	default:
		return NULL;
	}
}

int pte_env_apply(const input_t* input) {
	pte_saved_n = 0;
	if (!input->pte_present) {
		return 0;
	}

	for (uint32_t i = 0; i < input->pte_override_count; ++i) {
		const struct revisor_pte_override_entry* o = &input->pte_overrides[i];
		void* va;
		uint64_t live, newval, old = 0;

		if (REVISOR_PTE_LEVEL_LEAF != o->level) {   /* only leaf-descriptor overrides for now */
			pte_env_revert();
			return -EOPNOTSUPP;
		}
		va = sandbox_page_va(o->page_index);
		if (NULL == va) {
			pte_env_revert();
			return -EINVAL;
		}

		live = leaf_pte_val(va);
		newval = (live & ~o->mask) | o->value;
		if (!write_leaf_pte(va, newval, &old)) {    /* refuses a non-4K-leaf mapping */
			pte_env_revert();
			return -EOPNOTSUPP;
		}
		pte_saved[pte_saved_n].va = va;
		pte_saved[pte_saved_n].old = old;
		++pte_saved_n;
	}
	return 0;
}

void pte_env_revert(void) {
	while (pte_saved_n > 0) {
		uint64_t dummy = 0;
		--pte_saved_n;
		write_leaf_pte(pte_saved[pte_saved_n].va, pte_saved[pte_saved_n].old, &dummy);
	}
}
