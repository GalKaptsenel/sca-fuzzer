#include "execution_clause_rsb.h"
#include "call_stack.h"                        /* call_stack_peek/pop — the architectural (true) return */
#include "instruction_encodings.h"             /* classify_branch, barrier_fences_control */
#include "simulation_input.h"                  /* EXEC_CLAUSE_RSB */
#include "simulation_execution_clause_hook.h"  /* spec_request_window, spec_nesting, ... */

/* Return-stack-buffer misprediction (Spectre-RSB / ret2spec).
 *
 * The hardware RSB is a small FIXED-DEPTH stack of return addresses: a BL/BLR pushes the address after
 * it, a RET pops. Because it is finite, a call chain deeper than the RSB overwrites its oldest entries,
 * so the matching (deep) RETs pop STALE addresses and the frontend speculatively runs there until the
 * real return — recovered from the architectural stack/X30 — squashes it.
 *
 * We model that against the architectural call stack (call_stack), which is unbounded and therefore
 * always holds the true return. A finite shadow RSB (depth RSB_DEPTH, a plain circular buffer) mirrors
 * the same pushes/pops; indexing it by the call depth modulo RSB_DEPTH makes it agree with call_stack
 * while the chain is shallow and diverge (a stale slot from RSB_DEPTH calls ago) once it overflows. On a
 * RET whose shadow prediction differs from the true return, open a window exploring the prediction and
 * resuming at the true return.
 *
 * Only the architectural instruction stream trains the shadow — a wrong-path call/return never retires,
 * so it must not update the RSB; and because the shadow changes only on the (linear) architectural path,
 * it needs no per-window checkpoint. */

/* Approximate Neoverse N3 RSB depth. The exact value is not yet reverse-engineered (see the BPU RE
 * notes); until it is, this is a plausible placeholder — a smaller value catches shallower ret2spec
 * chains, a larger one is more conservative. */
#define RSB_DEPTH 16

static uintptr_t rsb[RSB_DEPTH];
static uint64_t  rsb_sp;        /* logical call depth (pushes - pops); slot = rsb_sp % RSB_DEPTH */
static uint64_t  rsb_index;

static void rsb_on_init(uint64_t index) { rsb_index = index; }

static void rsb_on_reset(void) {
	rsb_sp = 0;
	for (size_t i = 0; i < RSB_DEPTH; ++i) rsb[i] = 0;
}

static void rsb_on_instruction(struct simulation_state* sim_state) {
	/* Wrong-path calls/returns never retire, so the RSB only tracks the architectural stream. */
	if (0 != spec_nesting()) return;

	uintptr_t pc = sim_state->cpu_state.pc;
	uint32_t insn = *(uint32_t*)pc;
	branch_type_t bt = classify_branch(insn);

	if (BRANCH_BL == bt || BRANCH_BLR == bt) {
		rsb[rsb_sp % RSB_DEPTH] = pc + 4;   /* push (overwrites the oldest slot once depth > RSB_DEPTH) */
		++rsb_sp;
		return;
	}

	if (0xd65f03c0 == insn) {               /* RET (X30); matches the CE's architectural RET emulation */
		if (0 == rsb_sp) return;            /* no outstanding call */
		uintptr_t predicted = rsb[--rsb_sp % RSB_DEPTH];

		uintptr_t arch_return;
		if (!call_stack_peek(&arch_return)) return;   /* the true return handle_ret_hook will take */
		if (predicted == arch_return) return;         /* shallow enough: the RSB predicts correctly */
		if (spec_nesting() >= spec_max_nesting()) return;

		spec_request_window(predicted, arch_return, rsb_index);
	}
}

/* A control-fencing barrier stops the mispredicted return path: revert through the oldest open window. */
static uint64_t rsb_on_barrier(struct simulation_state* sim_state) {
	if (!barrier_fences_control(*(uint32_t*)sim_state->cpu_state.pc)) return SPEC_NO_REVERT;
	return spec_oldest_frame_of_owner(rsb_index);
}

/* This window was opened at a RET. The checkpoint is taken (in the clause dispatch) BEFORE
 * handle_ret_hook pops the architectural call stack, so the default reload restores the PRE-pop stack --
 * undoing the return's architectural pop and corrupting every later return. Re-apply that one pop after
 * the reload so the resumed architectural flow sees the correct post-return call stack. */
static void rsb_on_rollback(struct simulation_state* sim_state,
                            const struct execution_checkpoint_desc* frame) {
	spec_reload_checkpoint(sim_state, frame);
	uintptr_t discarded;
	call_stack_pop(&discarded);
}

const struct execution_clause_descriptor rsb_execution_clause = {
	.name           = "rsb",
	.clause_bit     = EXEC_CLAUSE_RSB,
	.on_init        = rsb_on_init,
	.on_reset       = rsb_on_reset,
	.on_instruction = rsb_on_instruction,
	.on_barrier     = rsb_on_barrier,
	.on_rollback    = rsb_on_rollback,
};
