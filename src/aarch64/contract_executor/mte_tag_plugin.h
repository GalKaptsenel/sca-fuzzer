#ifndef MTE_TAG_PLUGIN_H
#define MTE_TAG_PLUGIN_H

#include "simulation.h"

void mte_tag_plugin_init(void);
void mte_tag_plugin_cleanup(void);
void* mte_emulator_hook(struct simulation_state* sim_state);

/* Per-input MTE tag memory (one allocation tag per 16B granule), seeded from the input's MTE_TAGS
 * section. Present only in MTE-test mode (initial != NULL); otherwise there is no tag memory. The
 * speculation engine snapshots/restores it at each checkpoint for misspeculation rollback. */
void   mte_tagmem_init(uintptr_t base, const uint8_t* initial, size_t count);
void   mte_tagmem_free(void);
size_t mte_tagmem_bytes(void);
void   mte_tagmem_snapshot(uint8_t* dst);
void   mte_tagmem_restore(const uint8_t* src);
int     mte_tagmem_active(void);
uint8_t mte_tagmem_tag_at(uintptr_t addr);

/* True for the MTE memory-tag instructions (LDG/STG/STGP family): emulated and native-skipped, so
 * base_hook must not translate their base register kaddr<->uaddr. */
int     mte_is_mem_tag_access(uint32_t enc);

#endif // MTE_TAG_PLUGIN_H
