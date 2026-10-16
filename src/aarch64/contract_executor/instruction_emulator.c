#include "instruction_emulator.h"
#include "simulation.h"
#include "simulation_execution_clause_hook.h"
#include "pac_sign_plugin.h"
#include "mte_tag_plugin.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NOP_WORD 0xD503201FU

/* AArch64 permanently-undefined (UDF #0): the idle filler an op-emulator plants into its slots. The CE
 * restores the live code from sim_input every instruction step, so a site is re-materialized to the
 * genuine op on every reach (architectural OR speculative) — each reach signs/retags the pointer that
 * is actually there, so the AUT* always verifies and the single trace is the genuine render on both
 * paths. Between reaches the slots hold UDF: if one ever executes without being materialized (a bug),
 * it faults loudly instead of running stale bytes. */
#define UDF_WORD 0x00000000U

void emul_udf_fill(uint32_t first_slot, uint32_t n_slots)
{
    uint32_t* live   = (uint32_t*)simulation.sim_code.code;
    uint32_t* shadow = (uint32_t*)simulation.sim_input.code;
    for (uint32_t i = 0; i < n_slots; ++i) {
        live[first_slot + i] = shadow[first_slot + i] = UDF_WORD;
    }
}

/* op-emulator EMUL_OP_ARCH_CONFINE_MAIN: the slot holds the committed confine `AND Xn,Xn,#main_mask`
 * (rendered by the sealer; all geometry stays in Python). The emulator only gates it by nesting — a
 * speculative reach (nesting>0) NOPs it so the access may reach the faulty page; IDLE and an
 * architectural reach keep it (confine to main). shadow holds the committed AND and is never
 * overwritten, so each reach restores from it. The op owns its idle: the safe default is the confine
 * present, not UDF. */
static void arch_confine_main_emulate(struct simulation_state* sim_state, uintptr_t offset,
                                      uint32_t size, emul_phase_t phase)
{
    (void)sim_state;
    uint32_t* live   = (uint32_t*)simulation.sim_code.code;
    const uint32_t* shadow = (const uint32_t*)simulation.sim_input.code;
    uint32_t base = (uint32_t)(offset / 4);
    uint32_t n_slots = size / 4;
    int unconfine = (EMUL_PHASE_MATERIALIZE == phase) && (spec_nesting() > 0);
    for (uint32_t i = 0; i < n_slots; ++i) {
        live[base + i] = unconfine ? NOP_WORD : shadow[base + i];
    }
}

/* The op-emulator for each `enum emul_op`, indexed by op id (a hole stays NULL). Keying the table by
 * op makes a signature drift a compile error and the dispatch a single typed call. */
static const emul_op_fn emul_ops[] = {
    [EMUL_OP_PAC_AUTH]          = pac_emulate_auth,
    [EMUL_OP_MTE_RETAG]         = mte_emulate_retag,
    [EMUL_OP_ARCH_CONFINE_MAIN] = arch_confine_main_emulate,
};

/* Route a site phase to its op-emulator; the op owns everything it does with the slots. */
static void emul_route(struct simulation_state* sim_state, const struct emul_row* row, emul_phase_t phase)
{
    emul_op_fn op = (row->op < sizeof(emul_ops) / sizeof(emul_ops[0])) ? emul_ops[row->op] : NULL;
    if (NULL == op) {
        fprintf(stderr, "[CE FATAL] emulation row: unknown op %u -- aborting\n", row->op);
        abort();
    }
    op(sim_state, row->offset, row->size, phase);
}

/* Idle every site before the trace. IDLE needs no live state, so sim_state is NULL here. */
void emul_reset(void)
{
    const uint8_t* raw = simulation.sim_input.emul_table;
    if (NULL == raw) {
        return;
    }
    const struct emul_row* rows = (const struct emul_row*)raw;
    size_t n = simulation.sim_input.emul_table_size / sizeof(struct emul_row);
    for (size_t i = 0; i < n; ++i) {
        emul_route(NULL, &rows[i], EMUL_PHASE_IDLE);
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
            emul_route(sim_state, &rows[i], EMUL_PHASE_MATERIALIZE);
            return NULL;   /* re-run this PC, now the materialized instruction */
        }
        if ((uintptr_t)rows[i].offset + rows[i].size == off) {
            emul_route(sim_state, &rows[i], EMUL_PHASE_IDLE);   /* left the site: back to idle */
        }
    }
    return NULL;
}
