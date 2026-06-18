#include "mte_tag_plugin.h"
#include "simulation_state.h"
#include <stdint.h>
#include <stdbool.h>

/*
 * MTE instruction emulation for the Contract Executor.
 *
 * The CE runs as a regular EL0 process without MTE-enabled memory, so all
 * BASE-MTE instructions must be intercepted before they fault.  For each:
 *
 *   - Register-result instructions (IRG, GMI, ADDG, SUBG, SUBP, SUBPS):
 *     compute the register result in software and write it back.
 *
 *   - Memory-tag loads (LDG, LDGM): write 0 to the destination register
 *     (CE sandbox has no real tag memory).
 *
 *   - Memory-tag stores (STG, STZG, ST2G, STZ2G, STGM, STZGM, STGP): NOP.
 *     CE does not model tag memory; data correctness for STGP/STZG* is a
 *     known limitation.
 *
 * All handlers return PC+4 so the native instruction is skipped.
 *
 * All BASE-MTE instructions use 64-bit X registers exclusively — no W-register
 * variants exist in the MTE extension.
 */

/* -------------------------------------------------------------------------
 * GPR write helper for instructions whose Rd is a true GPR (reg 31 == XZR,
 * write ignored): GMI, SUBP/SUBPS, LDG/LDGM. SP-capable operands (IRG/ADDG/
 * SUBG Xd, and every Xn/Xm) use cpu_state_*_base_reg (reg 31 == SP) instead.
 * cpu_state.gpr[29-N] == XN;  X30 == lr.
 * ------------------------------------------------------------------------- */

static inline void write_xreg(struct cpu_state *s, uint32_t n, uint64_t v)
{
	if (n == 31) { return; }
	if (n == 30) { s->lr = (uintptr_t)v; return; }
	s->gpr[29 - n] = (uintptr_t)v;
}

/* -------------------------------------------------------------------------
 * MTE instruction classification
 *
 * Encoding patterns verified against aarch64-linux-gnu-as output:
 *
 * Data-tag register instructions (all 64-bit X registers only):
 *   IRG   bits[31:21]=10011010110 bits[15:10]=000100 → mask 0xFFE0FC00, val 0x9AC01000
 *   GMI   bits[31:21]=10011010110 bits[15:10]=000101 → mask 0xFFE0FC00, val 0x9AC01400
 *   SUBP  bits[31:21]=10011010110 bits[15:10]=000000 → mask 0xFFE0FC00, val 0x9AC00000
 *   SUBPS bits[31:21]=10111010110 bits[15:10]=000000 → mask 0xFFE0FC00, val 0xBAC00000
 *   ADDG  bits[31:22]=1001000110  bits[15:14]=00      → mask 0xFFC0C000, val 0x91800000
 *         fields: uimm6 at bits[21:16], uimm4 at bits[13:10]
 *   SUBG  bits[31:22]=1101000110  bits[15:14]=00      → mask 0xFFC0C000, val 0xD1800000
 *         fields: same as ADDG
 *
 * Memory-tag instructions — bits[31:24]=0xD9 (STGP uses 0x69):
 *   Within the 0xD9 family, bits[23:21] select sub-group and bits[11:10]
 *   distinguish multiple-granule forms (mode=00) from offset/pre/post-index:
 *     bits[23:21]=001: STZGM (mode=00) or STG  (mode≠00)
 *     bits[23:21]=011: LDG   (mode=00) or STZG (mode≠00)
 *     bits[23:21]=101: STGM  (mode=00) or ST2G (mode≠00)
 *     bits[23:21]=111: LDGM  (mode=00) or STZ2G(mode≠00)
 * ------------------------------------------------------------------------- */

typedef enum {
	MTE_NONE,
	MTE_IRG, MTE_GMI, MTE_SUBP, MTE_SUBPS, MTE_ADDG, MTE_SUBG,
	MTE_LDG, MTE_LDGM,
	MTE_STG, MTE_STZG, MTE_ST2G, MTE_STZ2G, MTE_STGM, MTE_STZGM, MTE_STGP,
} mte_type_t;

static mte_type_t classify_mte(uint32_t enc)
{
	/* Data-tag register instructions */
	switch (enc & 0xFFE0FC00u) {
		case 0x9AC01000u: return MTE_IRG;
		case 0x9AC01400u: return MTE_GMI;
		case 0x9AC00000u: return MTE_SUBP;
		case 0xBAC00000u: return MTE_SUBPS;
	}
	switch (enc & 0xFFC0C000u) {
		case 0x91800000u: return MTE_ADDG;
		case 0xD1800000u: return MTE_SUBG;
	}

	/* Memory-tag instructions with 0xD9 prefix */
	if ((enc >> 24) == 0xD9u) {
		uint32_t top3 = (enc >> 21) & 0x7u;
		uint32_t mode = (enc >> 10) & 0x3u;
		switch (top3) {
			case 0x1: return (mode == 0) ? MTE_STZGM : MTE_STG;
			case 0x3: return (mode == 0) ? MTE_LDG   : MTE_STZG;
			case 0x5: return (mode == 0) ? MTE_STGM  : MTE_ST2G;
			case 0x7: return (mode == 0) ? MTE_LDGM  : MTE_STZ2G;
		}
	}

	/* STGP: bits[31:24] = 0x69 */
	if ((enc >> 24) == 0x69u) {
		return MTE_STGP;
	}

	return MTE_NONE;
}

/* -------------------------------------------------------------------------
 * Lifecycle (no-op: pure software emulation, no hardware state to manage)
 * ------------------------------------------------------------------------- */

void mte_tag_plugin_init(void) {}
void mte_tag_plugin_cleanup(void) {}

/* -------------------------------------------------------------------------
 * mte_emulator_hook — intercept all BASE-MTE instructions and software-emulate them.
 *
 * Size and extension rules (verified against ARM ISA DDI0487):
 *   - All MTE instructions use 64-bit X registers exclusively.
 *   - ADDG/SUBG: address part (bits[55:0]) and tag part (bits[59:56]) are
 *     updated independently — no carry propagates from address into tag.
 *     bits[63:60] are preserved unchanged.
 *   - SUBP/SUBPS: result is the 56-bit signed difference, sign-extended to
 *     64 bits from bit 55 (not bit 63).  Uses the shift trick to avoid UB.
 *   - IRG: tag bits[59:56] are set to 0 (deterministic, key-free); address
 *     bits[55:0] and attribute bits[63:60] are copied from Rn unchanged.
 *   - GMI, LDG, LDGM: destination written with 0.
 *
 * Returns PC+4 (consumed) for any recognised MTE instruction, NULL otherwise.
 * ------------------------------------------------------------------------- */
void* mte_emulator_hook(struct simulation_state* sim_state)
{
	if (NULL == sim_state) { return NULL; }

	uint32_t enc = *(uint32_t*)sim_state->cpu_state.pc;
	mte_type_t mtype = classify_mte(enc);
	if (mtype == MTE_NONE) { return NULL; }

	struct cpu_state *state = &sim_state->cpu_state;

	uint32_t rd = enc & 0x1Fu;
	uint32_t rn = (enc >> 5) & 0x1Fu;
	uint32_t rm = (enc >> 16) & 0x1Fu;

	switch (mtype) {
		case MTE_IRG: {
			/*
			 * IRG Xd, Xn[, Xm]: insert a random tag in Xd bits[59:56].
			 * CE uses tag=0 (deterministic).  Address bits[55:0] and attribute
			 * bits[63:60] are copied from Xn unchanged.
			 */
			uint64_t xn = cpu_state_read_base_reg(state, rn);
			/* clear bits[59:56]; preserve bits[63:60] and bits[55:0] */
			cpu_state_write_base_reg(state, rd, xn & 0xF0FFFFFFFFFFFFFFu);
			break;
		}
		case MTE_GMI: {
			/*
			 * GMI Xd, Xn, Xm: compute excluded-tag-set mask.
			 * Not relevant for cache-footprint analysis; zero the destination.
			 */
			write_xreg(state, rd, 0);
			break;
		}
		case MTE_SUBP:
		case MTE_SUBPS: {
			/*
			 * SUBP[S] Xd, Xn, Xm: signed 56-bit pointer subtraction.
			 *
			 * ARM spec: result = Xn[55:0] - Xm[55:0], computed as 56-bit
			 * unsigned (wrapping), then sign-extended to 64 bits from bit 55.
			 * SUBPS additionally sets NZCV; we omit the flag update here since
			 * CE does not currently model MTE-specific flag writes.
			 *
			 * Sign extension from bit 55 uses the shift trick to avoid UB:
			 *   << 8 moves bit 55 to bit 63, then arithmetic >> 8 fills
			 *   bits[63:56] with the sign.
			 */
			uint64_t xn56 = cpu_state_read_base_reg(state, rn) & 0x00FFFFFFFFFFFFFFu;
			uint64_t xm56 = cpu_state_read_base_reg(state, rm) & 0x00FFFFFFFFFFFFFFu;
			uint64_t diff56 = (xn56 - xm56) & 0x00FFFFFFFFFFFFFFu;
			uint64_t shifted = diff56 << 8;   /* bit55 → bit63 */
			uint64_t result  = (uint64_t)((int64_t)shifted >> 8);  /* arith sign-fill */
			write_xreg(state, rd, result);
			break;
		}
		case MTE_ADDG: {
			/*
			 * ADDG Xd, Xn, #uimm6, #uimm4:
			 *   address part (bits[55:0])  += uimm6 * 16  (no carry into tag)
			 *   tag part    (bits[59:56])  += uimm4       (mod 16, no carry from address)
			 *   attribute   (bits[63:60])   preserved
			 *
			 * uimm6 at bits[21:16] (6-bit unsigned, 0..63, unit = 16 bytes).
			 * uimm4 at bits[13:10] (4-bit unsigned, 0..15).
			 */
			uint64_t xn    = cpu_state_read_base_reg(state, rn);
			uint64_t uimm6 = (enc >> 16) & 0x3Fu;
			uint64_t uimm4 = (enc >> 10) & 0xFu;
			uint64_t addr56   = ((xn & 0x00FFFFFFFFFFFFFFu) + (uimm6 << 4))
			                    & 0x00FFFFFFFFFFFFFFu;
			uint64_t new_tag  = ((xn >> 56) + uimm4) & 0xFu;
			uint64_t result   = (xn & 0xF000000000000000u) | (new_tag << 56) | addr56;
			cpu_state_write_base_reg(state, rd, result);
			break;
		}
		case MTE_SUBG: {
			/*
			 * SUBG Xd, Xn, #uimm6, #uimm4: symmetric to ADDG but subtracts.
			 * Tag wraps mod 16 via & 0xF (unsigned underflow is well-defined).
			 */
			uint64_t xn    = cpu_state_read_base_reg(state, rn);
			uint64_t uimm6 = (enc >> 16) & 0x3Fu;
			uint64_t uimm4 = (enc >> 10) & 0xFu;
			uint64_t addr56   = ((xn & 0x00FFFFFFFFFFFFFFu) - (uimm6 << 4))
			                    & 0x00FFFFFFFFFFFFFFu;
			uint64_t new_tag  = ((xn >> 56) - uimm4) & 0xFu;
			uint64_t result   = (xn & 0xF000000000000000u) | (new_tag << 56) | addr56;
			cpu_state_write_base_reg(state, rd, result);
			break;
		}
		case MTE_LDG: {
			/*
			 * LDG Xt, [Xn, #imm]: load allocation tag for address Xn+imm into
			 * Xt bits[59:56].  CE sandbox has no real tag memory; return 0.
			 */
			write_xreg(state, rd, 0);
			break;
		}
		case MTE_LDGM: {
			/*
			 * LDGM Xt, [Xn]: load tag mask (16 tags) into Xt.
			 * No real tag memory in CE; return 0.
			 */
			write_xreg(state, rd, 0);
			break;
		}
		case MTE_STG:
		case MTE_STZG:
		case MTE_ST2G:
		case MTE_STZ2G:
		case MTE_STGM:
		case MTE_STZGM:
		case MTE_STGP:
			/*
			 * Tag-store instructions: set tags in memory (STZG, STZ2G, STZGM,
			 * STGP additionally write zeros or data to the granule).
			 * CE does not model tag memory.  The data-write semantics of the
			 * zeroing and pair variants are suppressed -- a known limitation.
			 * Return PC+4 to skip native execution, which faults without
			 * MTE-enabled memory.
			 */
			break;
		default:
			break;
	}

	return (void*)(sim_state->cpu_state.pc + 4);
}
