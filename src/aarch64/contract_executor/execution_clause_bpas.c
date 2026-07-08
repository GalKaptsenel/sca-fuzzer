#include "execution_clause_bpas.h"
#include "instruction_encodings.h"   /* is_memory_access, is_store */
#include "mte_tag_plugin.h"          /* mte_is_mem_tag_access */
#include "simulation_input.h"        /* EXEC_CLAUSE_BPAS */
#include "simulation_output.h"       /* instr_trace_entry_t, trace_pop_last, trace_emit_entry */
#include "simulation_execution_clause_hook.h"  /* spec_reload_checkpoint, spec_max_nesting, frame desc */
#include <stdlib.h>
#include <string.h>

/* Two-phase store bypass:
 *   phase A (at a store): snapshot memory, then let the HW execute the store (writeback included);
 *   phase B (next instruction): undo the store's memory write so the speculative window reads the
 *     stale value, checkpointing the POST-store state as the architectural return — so the store
 *     is never re-executed and rollback stays uniform.
 * Whole-memory snapshot/restore => STR / STP / writeback / atomics are all handled by the HW. */
static uint8_t* bpas_preimage     = NULL;   /* snapshot of memory just before a pending store */
static size_t   bpas_preimage_cap = 0;
static int      bpas_pending      = 0;
static uint64_t bpas_pending_nesting;       /* nesting at which the pending store was seen */
static uintptr_t bpas_pending_store_pc;     /* pc of that store (fail-loud check at phase B) */
static uint64_t bpas_index;

/* A bypassed store's trace entry is pulled out of the trace at phase B and re-emitted when its
 * window unwinds. Nested bypasses stack, one deferred entry per open window (LIFO, balanced with the
 * bpas speculation frames), so this is a stack, not a single slot. */
static instr_trace_entry_t* bpas_deferred     = NULL;
static size_t               bpas_deferred_cap = 0;
static size_t               bpas_deferred_top = 0;

static void bpas_on_init(uint64_t index) { bpas_index = index; }

static void bpas_on_reset(void) {
	bpas_pending = 0;
	bpas_deferred_top = 0;
	size_t msz = spec_memory_size();          /* grow only if a sim ever needs more (cold path) */
	if (bpas_preimage_cap < msz) {
		uint8_t* grown = realloc(bpas_preimage, msz);
		if (NULL == grown) __builtin_trap();   // OOM
		bpas_preimage = grown;
		bpas_preimage_cap = msz;
	}
	size_t need = spec_max_nesting() + 1;      /* at most one deferred store per open window */
	if (bpas_deferred_cap < need) {
		instr_trace_entry_t* grown = realloc(bpas_deferred, need * sizeof(instr_trace_entry_t));
		if (NULL == grown) __builtin_trap();   // OOM
		bpas_deferred = grown;
		bpas_deferred_cap = need;
	}
}

static void* bpas_on_instruction(struct simulation_state* sim_state) {
	/* Phase B: apply the pending bypass — but only if no rollback has happened since phase A
	 * (a changed nesting depth means our snapshot belongs to a now-discarded context). */
	if (bpas_pending) {
		if (spec_nesting() == bpas_pending_nesting) {
			spec_push_frame(sim_state, sim_state->cpu_state.pc, bpas_index); /* post-store = arch return */
			memcpy(sim_state->memory, bpas_preimage, spec_memory_size());    /* undo store -> stale */
			/* Relocate the bypassed store's trace entry to this window's unwind (bpas_on_rollback), so
			 * in trace order it lands after the loads that read the stale value — which must taint the
			 * input — and before the architectural continuation, which still sees the committed store.
			 * The guard above passing means no rollback since phase A, so the store is the last entry. */
			if (bpas_deferred_top >= bpas_deferred_cap) __builtin_trap();
			instr_trace_entry_t store = trace_pop_last();
			if (store.cpu.pc != bpas_pending_store_pc) __builtin_trap();  /* must be the bypassed store */
			bpas_deferred[bpas_deferred_top++] = store;
		}
		bpas_pending = 0;
	}

	/* Phase A: snapshot before a store, let the HW execute it, undo it on the next instruction. */
	uint32_t insn = *(uint32_t*)sim_state->cpu_state.pc;
	if (spec_nesting() < spec_max_nesting() && is_memory_access(insn) && is_store(insn)
	    && !mte_is_mem_tag_access(insn)) {
		memcpy(bpas_preimage, sim_state->memory, spec_memory_size());
		bpas_pending = 1;
		bpas_pending_nesting = spec_nesting();
		bpas_pending_store_pc = sim_state->cpu_state.pc;
	}
	return NULL;
}

/* Window unwind: emit the bypassed store now — after the stale loads (already logged), before the
 * architectural continuation — then restore the checkpoint exactly like the engine's default path. */
static void bpas_on_rollback(struct simulation_state* sim_state,
                             const struct execution_checkpoint_desc* frame) {
	if (0 == bpas_deferred_top) __builtin_trap();   /* one deferred store per open bpas window */
	trace_emit_entry(&bpas_deferred[--bpas_deferred_top]);
	spec_reload_checkpoint(sim_state, frame);
}

/* Write a bypassed store's committed value(s) back into live memory: its `after` at its effective
 * address, pair-aware. (sim_state->memory is simulation_memory, which kaddr2uaddr indexes.) */
static void bpas_commit_store(const instr_trace_entry_t* store) {
	const mem_access_t* a = &store->metadata.memory_access;
	memcpy(kaddr2uaddr((void*)(uintptr_t)a->effective_address), &a->after, (size_t)a->element_size);
	if (store->metadata.is_pair) {
		const mem_access_t* b = &store->metadata.memory_access2;
		memcpy(kaddr2uaddr((void*)(uintptr_t)b->effective_address), &b->after, (size_t)b->element_size);
	}
}

/* Barrier handling for store bypass. Two regimes:
 *
 *  - SSBB / PSSBB are store->load *reordering* barriers only. They stop a load after the barrier from
 *    bypassing a store before it, but they do NOT squash speculation: a value already bypassed into a
 *    register before the barrier stays stale, and it can still be transmitted after the barrier. So we
 *    commit every open bypassed store into live memory (oldest -> newest, last write wins), so loads
 *    past the barrier see committed values, while leaving registers and the speculation itself
 *    untouched. The windows stay open and still roll back to the architectural state at their natural end.
 *
 *  - DSB / ISB / SB are stronger (complete prior accesses / flush the pipeline / general speculation
 *    barrier): they end the mispredicted path, so keep the squash — revert through the oldest open
 *    bypass window and everything nested in it.
 *
 * (A load in program order BEFORE the barrier never reads a later store — the CE runs in program order
 * and never forwards from a future store — so that half of the barrier needs no modeling.) */
static uint64_t bpas_on_barrier(struct simulation_state* sim_state) {
	uint32_t insn = *(uint32_t*)sim_state->cpu_state.pc;
	barrier_kind_t kind = barrier_kind(insn);
	if (BARRIER_SSBB == kind || BARRIER_PSSBB == kind) {
		for (size_t i = 0; i < bpas_deferred_top; ++i) bpas_commit_store(&bpas_deferred[i]);
		return SPEC_NO_REVERT;
	}
	if (!barrier_fences_store_bypass(insn)) return SPEC_NO_REVERT;
	return spec_oldest_frame_of_owner(bpas_index);
}

const struct execution_clause_descriptor bpas_execution_clause = {
	.name           = "bpas",
	.clause_bit     = EXEC_CLAUSE_BPAS,
	.on_init        = bpas_on_init,
	.on_reset       = bpas_on_reset,
	.on_instruction = bpas_on_instruction,
	.on_barrier     = bpas_on_barrier,
	.on_rollback    = bpas_on_rollback,
};
