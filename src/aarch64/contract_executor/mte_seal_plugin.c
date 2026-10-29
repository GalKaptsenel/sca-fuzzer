/*
 * Seal-site op-emulator for MTE: supply the accessed cell's allocation tag to a sealed access so the
 * genuine binary's tag check matches. The site is [UMOV x27,applied; BFI tag; access; UMOV x27,cell;
 * BFI tag]: the pre-BFI applies the tag the variant tests (genuine == the cell tag), the post-BFI
 * re-converges the pointer onto the cell tag. MATERIALIZE reads the cell tag at the access's effective
 * address and injects it into both lanes (row->gst_offset = applied, row->gst_offset2 = re-converge);
 * base_hook reloads the vector file from the image before the UMOV re-runs. On a FEAT_MTE host the
 * fixed BFIs and the tagged access execute natively instead. IDLE: nothing (the site's code is fixed).
 * Kept separate from mte_tag_plugin.c (the self-contained tag-memory emulator) because it pulls in the
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

void mte_emulate_retag(struct simulation_state *sim_state, const struct emul_row *row,
		       emul_phase_t phase) {
	if (EMUL_PHASE_IDLE == phase) {
		return;
	}
	/* An MTE_RETAG row only exists when MTE sealing is active, so tag memory is seeded; a missing one
	 * is a bug, not a tolerable state — fail loud, as the op dispatch does. */
	if (NULL == sim_state || !mte_tagmem_active()) {
		fprintf(stderr, "[CE FATAL] mte_emulate_retag: sim_state=%p tagmem_active=%d -- aborting\n",
			(void *)sim_state, mte_tagmem_active());
		abort();
	}
	uint32_t *shadow = (uint32_t *)simulation.sim_input.code;   // the logger reads encodings from here
	uint32_t access = shadow[(row->offset + row->size) / 4];    // the access, at offset+size
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
	uint8_t cell = (uint8_t)(mte_tagmem_tag_at(ea) & 0xFu);     // tag of the granule the access touches
	// The BFI sets the BASE register's tag, but the effective address's tag is base + index (a
	// register-offset access carries the index's tag into the EA). Target the EA tag: the base tag that
	// makes EA tag == cell is base_tag + (cell - ea_tag). For an immediate-offset access ea_tag ==
	// base_tag, so this reduces to cell.
	uint8_t base_tag = (uint8_t)((cpu_state_read_base_reg(cpu, get_rn(access)) >> 56) & 0xFu);
	uint8_t ea_tag = (uint8_t)((ea >> 56) & 0xFu);
	uint8_t applied = (uint8_t)((base_tag + cell - ea_tag) & 0xFu);
	if (mte_tag_log_enabled()) {
		fprintf(stderr, "MTERETAG side=CE off=0x%x ea=0x%lx gran=%ld cell=%u base_tag=%u ea_tag=%u applied=%u\n",
			row->offset, (unsigned long)ea, ce_granule(ea), cell, base_tag, ea_tag, applied);
	}
	// Inject the applied tag into the applied lane (genuine) and the re-converge lane; each is one byte,
	// read by the site's UMOV.b and inserted into the base register's tag [59:56] by BFI.
	if (row->gst_offset >= sizeof(simulation.sim_input.simd) ||
	    row->gst_offset2 >= sizeof(simulation.sim_input.simd)) {
		fprintf(stderr, "[CE FATAL] mte_emulate_retag: gst_offset out of range -- aborting\n");
		abort();
	}
	simulation.sim_input.simd[row->gst_offset]  = applied;
	simulation.sim_input.simd[row->gst_offset2] = applied;
}

