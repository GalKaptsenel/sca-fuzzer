#include "instruction_emulator.h"
#include "simulation.h"
#include "simulation_execution_clause_hook.h"
#include "pac_sign_plugin.h"
#include <stdlib.h>
#include <stdio.h>

/* Per-message "row already emulated" flags: an emulated row stays materialized in the code (code
 * memory is not checkpointed), so a speculative re-entry runs the planted instructions directly
 * instead of re-invoking the op-emulator. Reset once before each trace. */
static uint8_t *g_done = NULL;
static size_t   g_done_n = 0;

void emul_reset(void)
{
    free(g_done);
    g_done = NULL;
    g_done_n = simulation.sim_input.emul_table_size / sizeof(struct emul_row);
    if (0 != g_done_n) {
        g_done = calloc(g_done_n, 1);
        if (NULL == g_done) {
            fprintf(stderr, "[CE FATAL] emul_reset: out of memory for %zu rows\n", g_done_n);
            abort();
        }
    }
}

void* instruction_emulator_hook(struct simulation_state* sim_state)
{
    const uint8_t* raw = simulation.sim_input.emul_table;
    if (NULL == sim_state || NULL == raw) {
        return NULL;
    }
    /* Materialize only on the architectural path: the value signed/retagged into the slot must be the
     * architectural one (a speculative pre-reach would capture a wrong-path pointer). A speculative
     * reach before materialization runs the inert junk slots + a speculative AUT* (which strips, never
     * verifies), so it is safe; once materialized, the planted code persists and runs under speculation
     * too. */
    if (0 != is_in_speculation()) {
        return NULL;
    }
    const struct emul_row* rows = (const struct emul_row*)raw;
    uintptr_t off = sim_state->cpu_state.pc - (uintptr_t)simulation.sim_code.code;

    for (size_t i = 0; i < g_done_n; ++i) {
        if (rows[i].offset != off || 0 != g_done[i]) {
            continue;
        }
        g_done[i] = 1;
        switch (rows[i].op) {
            case EMUL_OP_PAC_AUTH:
                pac_emulate_auth(sim_state, rows[i].offset, rows[i].size);
                break;
            default:
                fprintf(stderr, "[CE FATAL] emulation row %zu: unknown op %u -- aborting\n",
                        i, rows[i].op);
                abort();
        }
        return NULL;   /* continue; the now-materialized instruction at this PC runs normally */
    }
    return NULL;
}
