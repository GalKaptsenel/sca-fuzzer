#include "instruction_emulator.h"
#include "simulation.h"
#include "pac_sign_plugin.h"
#include "mte_tag_plugin.h"
#include "mte_seal_plugin.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* The op-emulator for each `enum emul_op`, indexed by op id (a hole stays NULL). Keying the table by
 * op makes a signature drift a compile error and the dispatch a single typed call. */
static const emul_op_fn emul_ops[] = {
    [EMUL_OP_PAC_AUTH]  = pac_emulate_auth,
    [EMUL_OP_MTE_RETAG] = mte_emulate_retag,
};

/* Route a site phase to its op-emulator. */
static void emul_route(struct simulation_state* sim_state, const struct emul_row* row, emul_phase_t phase)
{
    emul_op_fn op = (row->op < sizeof(emul_ops) / sizeof(emul_ops[0])) ? emul_ops[row->op] : NULL;
    if (NULL == op) {
        fprintf(stderr, "[CE FATAL] emulation row: unknown op %u -- aborting\n", row->op);
        abort();
    }
    op(sim_state, row, phase);
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

    /* MATERIALIZE when PC is at a row's first slot: the op injects the genuine value into the GVT image
     * (base_hook then reloads it into the vector file before this same PC re-runs the UMOV). */
    for (size_t i = 0; i < n; ++i) {
        if (rows[i].offset == off) {
            emul_route(sim_state, &rows[i], EMUL_PHASE_MATERIALIZE);
            return NULL;
        }
    }
    return NULL;
}
