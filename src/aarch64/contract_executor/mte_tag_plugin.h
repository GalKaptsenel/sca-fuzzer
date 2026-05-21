#ifndef MTE_TAG_PLUGIN_H
#define MTE_TAG_PLUGIN_H

#include "simulation.h"

void mte_tag_plugin_init(void);
void mte_tag_plugin_cleanup(void);
void* mte_tag_hook(struct simulation_state* sim_state);

#endif // MTE_TAG_PLUGIN_H
