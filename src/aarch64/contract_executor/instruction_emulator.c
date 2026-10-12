#include "instruction_emulator.h"
#include "simulation.h"
#include "pac_sign_plugin.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* AArch64 permanently-undefined (UDF #0): the idle filler for a site's materialized slots. The CE
 * restores the live code from sim_input every instruction step, so a slot is re-materialized to the
 * genuine op on every reach (architectural OR speculative) — each reach signs/retags the pointer that
 * is actually there, so the AUT* always verifies and the single trace is the genuine render on both
 * paths. Between reaches the slots hold UDF: if one ever executes without being materialized (a bug),
 * it faults loudly instead of running stale bytes. */
#define UDF_WORD 0x00000000U

static void udf_fill(uint32_t base, uint32_t n_slots)
{
    uint32_t* live   = (uint32_t*)simulation.sim_code.code;
    uint32_t* shadow = (uint32_t*)simulation.sim_input.code;
    for (uint32_t i = 0; i < n_slots; ++i) {
        live[base + i] = shadow[base + i] = UDF_WORD;
    }
}

/* Reset every site's materialized slots to the idle UDF filler, once before each trace. The last slot
 * of a site holds its real op (the committed AUT*) and is never touched here. */
void emul_reset(void)
{
    const uint8_t* raw = simulation.sim_input.emul_table;
    if (NULL == raw) {
        return;
    }
    const struct emul_row* rows = (const struct emul_row*)raw;
    size_t n = simulation.sim_input.emul_table_size / sizeof(struct emul_row);
    for (size_t i = 0; i < n; ++i) {
        udf_fill(rows[i].offset / 4, rows[i].size / 4 - 1);
    }
}

void* instruction_emulator_hook(struct simulation_state* sim_state)
{
    const uint8_t* raw = simulation.sim_input.emul_table;
    if (NULL == sim_state || NULL == raw) {
        return NULL;
    }
    const struct emul_row* rows = (const struct emul_row*)raw;
    size_t n = simulation.sim_input.emul_table_size / sizeof(struct emul_row);
    uintptr_t off = sim_state->cpu_state.pc - (uintptr_t)simulation.sim_code.code;

    for (size_t i = 0; i < n; ++i) {
        if (rows[i].offset == off) {
            /* reaching the site: materialize the genuine op for the pointer currently in the register */
            switch (rows[i].op) {
                case EMUL_OP_PAC_AUTH:
                    pac_emulate_auth(sim_state, rows[i].offset, rows[i].size);
                    break;
                default:
                    fprintf(stderr, "[CE FATAL] emulation row %zu: unknown op %u -- aborting\n",
                            i, rows[i].op);
                    abort();
            }
            return NULL;   /* re-run this PC, now the materialized instruction */
        }
        if ((uintptr_t)rows[i].offset + rows[i].size == off) {
            /* left the site: restore the idle UDF filler so a later bug can't run stale bytes */
            udf_fill(rows[i].offset / 4, rows[i].size / 4 - 1);
        }
    }
    return NULL;
}
