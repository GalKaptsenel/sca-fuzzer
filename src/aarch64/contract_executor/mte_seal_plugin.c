/*
 * Seal-site op-emulators for MTE: materialize the retag ADDG before a sealed access (merge the pointer
 * onto the accessed cell's tag) and the de-merge ADDG after it (re-converge). These run on the emulated
 * path; on a FEAT_MTE host the materialized ADDGs and the tagged access execute natively instead.
 * Kept separate from mte_tag_plugin.c (the self-contained tag-memory emulator) because they pull in the
 * emulation-table framework and the instruction decoder.
 */
#include "mte_seal_plugin.h"
#include "mte_tag_plugin.h"
#include "simulation.h"
#include "simulation_state.h"
#include "simulation_output.h"
#include "instruction_encodings.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void mte_emulate_retag(struct simulation_state *sim_state, uintptr_t offset, uint32_t size,
		       emul_phase_t phase) {
	uint32_t *live   = (uint32_t *)simulation.sim_code.code;
	uint32_t *shadow = (uint32_t *)simulation.sim_input.code;   // the logger reads encodings from here
	uint32_t base = (uint32_t)(offset / 4);
	uint32_t n_slots = size / 4;
	if (EMUL_PHASE_IDLE == phase) {
		emul_udf_fill(base, n_slots);
		return;
	}
	/* An MTE_RETAG row only exists when MTE sealing is active, so tag memory is seeded; a missing one
	 * is a bug, not a tolerable state — fail loud, as the op dispatch does. */
	if (NULL == sim_state || !mte_tagmem_active()) {
		fprintf(stderr, "[CE FATAL] mte_emulate_retag: sim_state=%p tagmem_active=%d -- aborting\n",
			(void *)sim_state, mte_tagmem_active());
		abort();
	}
	uint32_t access = shadow[base + n_slots];                   // the access, just past the retag slot
	uint32_t rn = get_rn(access);                               // the base register it (and the retag) use
	struct cpu_state *cpu = &sim_state->cpu_state;
	trace_cpu_state_t tcs = { 0 };
	for (uint32_t r = 0; r < 31; ++r) { tcs.gpr[r] = cpu_state_read_base_reg(cpu, r); }
	tcs.sp = cpu->sp;
	uintptr_t ea = (uintptr_t)parse_memory_access_instruction(access, &tcs).effective_address;
	if (ce_granule(ea) < 0) {                                   // the access must land in the tagged span
		fprintf(stderr, "[CE FATAL] mte_emulate_retag: EA 0x%lx outside the tagged span -- aborting\n",
			(unsigned long)ea);
		abort();
	}
	uint32_t cell = mte_tagmem_tag_at(ea);                      // tag of the granule the access touches
	/* Adding delta to Rn's tag shifts the effective address's tag by the same delta (EA = Rn + offset,
	 * so EA += delta<<56); target the EA tag, not Rn's own — they differ for a register-offset access. */
	uint32_t ea_tag = (uint32_t)((ea >> 56) & 0xFu);
	uint32_t delta = (cell - ea_tag) & 0xFu;
	if (mte_tag_log_enabled()) {
		fprintf(stderr, "MTERETAG side=CE off=0x%lx ea=0x%lx gran=%ld ea_tag=%u cell=%u delta=%u rn=%u\n",
			(unsigned long)offset, (unsigned long)ea, ce_granule(ea), ea_tag, cell, delta, rn);
	}
	uint32_t addg = 0x91800000U | ((uint32_t)delta << 10) | (rn << 5) | rn;
	for (uint32_t i = 0; i < n_slots; ++i) {
		live[base + i] = shadow[base + i] = addg;
	}
}

void mte_emulate_demerge(struct simulation_state *sim_state, uintptr_t offset, uint32_t size,
			 emul_phase_t phase) {
	uint32_t *live   = (uint32_t *)simulation.sim_code.code;
	uint32_t *shadow = (uint32_t *)simulation.sim_input.code;
	uint32_t base = (uint32_t)(offset / 4);
	uint32_t n_slots = size / 4;
	if (EMUL_PHASE_IDLE == phase) {
		emul_udf_fill(base, n_slots);
		return;
	}
	if (NULL == sim_state || !mte_tagmem_active() || 0 == base) {
		fprintf(stderr, "[CE FATAL] mte_emulate_demerge: sim_state=%p tagmem_active=%d base=%u -- aborting\n",
			(void *)sim_state, mte_tagmem_active(), base);
		abort();
	}
	uint32_t access = shadow[base - 1];                   // the sealed access sits just before this slot
	if (!is_memory_access(access)) {
		fprintf(stderr, "[CE FATAL] mte_emulate_demerge: instruction before slot (0x%08x) is not a "
			"memory access -- aborting\n", access);
		abort();
	}
	uint32_t rn = get_rn(access);                         // the sealed pointer register the ADDG retags
	struct cpu_state *cpu = &sim_state->cpu_state;
	trace_cpu_state_t tcs = { 0 };
	for (uint32_t r = 0; r < 31; ++r) { tcs.gpr[r] = cpu_state_read_base_reg(cpu, r); }
	tcs.sp = cpu->sp;
	uintptr_t ea = (uintptr_t)parse_memory_access_instruction(access, &tcs).effective_address;
	if (ce_granule(ea) < 0) {
		fprintf(stderr, "[CE FATAL] mte_emulate_demerge: EA 0x%lx outside the tagged span -- aborting\n",
			(unsigned long)ea);
		abort();
	}
	uint32_t cell = mte_tagmem_tag_at(ea);
	uint32_t ea_tag = (uint32_t)((ea >> 56) & 0xFu);
	uint32_t delta = (cell - ea_tag) & 0xFu;              // re-converge the pointer tag to the cell tag
	uint32_t addg = 0x91800000U | ((uint32_t)delta << 10) | (rn << 5) | rn;
	for (uint32_t i = 0; i < n_slots; ++i) {
		live[base + i] = shadow[base + i] = addg;
	}
}
