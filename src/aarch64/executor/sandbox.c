#include "main.h"

void initialize_sandbox(sandbox_t* sandbox) {
	mte_randomly_tag_region(sandbox->lower_overflow, sizeof(sandbox->lower_overflow));
	mte_randomly_tag_region(sandbox->main_region, sizeof(sandbox->main_region));
	mte_randomly_tag_region(sandbox->faulty_region, sizeof(sandbox->faulty_region));
	mte_randomly_tag_region(sandbox->upper_overflow, sizeof(sandbox->upper_overflow));
	disable_mte_for_region(sandbox->eviction_region, sizeof(sandbox->eviction_region));
}

