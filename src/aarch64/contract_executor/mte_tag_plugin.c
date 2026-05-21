#include "mte_tag_plugin.h"
#include "simulation_state.h"
#include <stdint.h>
#include <stdbool.h>

/*
 * MTE sentinel: ADDG Xd, Xd, #0, #0
 *
 * ADDG encoding: bits[31:22] = opcode, bits[21:16] = uimm6, bits[15:14] = 00,
 *                bits[13:10] = uimm4, bits[9:5] = Rn, bits[4:0] = Rd.
 * When uimm6=0 and uimm4=0, bits[31:10] match ADDG_SENTINEL_VALUE.
 * The additional Rd==Rn check makes it a true NOP (ADDG Xd, Xd, #0, #0).
 */
#define ADDG_SENTINEL_MASK  0xFFFFFC00u
#define ADDG_SENTINEL_VALUE 0x91C00000u

static bool is_mte_sentinel(uint32_t enc)
{
    if ((enc & ADDG_SENTINEL_MASK) != ADDG_SENTINEL_VALUE) {
        return false;
    }
    uint32_t rd = enc & 0x1Fu;
    uint32_t rn = (enc >> 5) & 0x1Fu;
    return rd == rn;
}

void mte_tag_plugin_init(void) {}

void mte_tag_plugin_cleanup(void) {}

/*
 * mte_tag_hook — intercept ADDG Xd, Xd, #0, #0 sentinel instructions placed
 * by MTEInstrumentation.instrument_stage1() right before every sandboxed
 * memory access.
 *
 * In the CE run the sentinel is a pure NOP: no register is modified.
 * The speculation_nesting of the trace entry for the instruction immediately
 * AFTER the sentinel (the actual memory access at PC+4) is read by the Python
 * executor to classify each slot as arch (nesting==0) or speculative (>0).
 *
 * Stage 2 then replaces each sentinel with:
 *   arch  path → NOP            (correct tag already in register from AND+ADD)
 *   spec  path → IRG Xd, Xd    (random tag; tests MTE non-interference)
 *
 * Returns: PC+4 (consumed) when a sentinel is detected, NULL otherwise.
 */
void* mte_tag_hook(struct simulation_state* sim_state)
{
    uint32_t enc = *(uint32_t*)sim_state->cpu_state.pc;
    if (!is_mte_sentinel(enc)) {
        return NULL;
    }
    return (void*)(sim_state->cpu_state.pc + 4);
}
