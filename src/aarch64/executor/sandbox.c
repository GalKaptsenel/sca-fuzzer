#include "main.h"

void initialize_sandbox(sandbox_t* sandbox) {
	enable_TCMA1_bit(); // TODO: Should also to return to orignal value whenever test is done
	mte_randomly_tag_region(sandbox->main_region, sizeof(sandbox->main_region));
	mte_randomly_tag_region(sandbox->faulty_region, sizeof(sandbox->faulty_region));
	disable_mte_for_region(executor.sandbox.eviction_region, sizeof(executor.sandbox.eviction_region));
}
EXPORT_SYMBOL(initialize_sandbox);

