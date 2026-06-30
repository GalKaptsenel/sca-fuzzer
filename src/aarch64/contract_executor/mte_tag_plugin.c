#include "mte_tag_plugin.h"
#include "simulation_state.h"
#include <stdint.h>
#include <stdbool.h>

/* GMID_EL1.BS — log2 of the LDGM/STGM tag-block size, in granules. Fixed to 4 (a 16-granule,
 * 256-byte block), the known default for ARM Neoverse server, ARM Cortex High-Performance, and ARM
 * Cortex Efficiency cores. (GMID_EL1 is EL1-only, so the CE at EL0 cannot read it; other
 * implementations would have the kernel read GMID_EL1.BS at EL1 and pass it down.) */
#define MTE_GMID_BS             4u
#define MTE_TAG_BLOCK_GRANULES  (1u << MTE_GMID_BS)   /* tags per LDGM/STGM block */
#define MTE_TAG_BLOCK_BYTES     (MTE_TAG_BLOCK_GRANULES * 16u)

/*
 * MTE instruction emulation for the Contract Executor.
 *
 * The CE runs as a regular EL0 process without MTE-enabled memory, so all
 * MTE instructions must be intercepted before they fault.  For each:
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
 * All MTE instructions use 64-bit X registers exclusively — no W-register
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

	/* STGP: bits[31:24] = 0x68 (post-index) or 0x69 (signed-offset / pre-index), L(bit22)=0. LDPSW
	 * shares the top byte but has L=1 -- it is a normal pair load, not a tag store. */
	if (((enc >> 24) == 0x68u || (enc >> 24) == 0x69u) && (((enc >> 22) & 1u) == 0u)) {
		return MTE_STGP;
	}

	return MTE_NONE;
}

/* -------------------------------------------------------------------------
 * Lifecycle (no-op: pure software emulation, no hardware state to manage)
 * ------------------------------------------------------------------------- */

void mte_tag_plugin_init(void) {}
void mte_tag_plugin_cleanup(void) {}

/* ---- per-input MTE tag memory (one allocation tag per 16B granule of the sandbox) -------------
 * The CE models the allocation tags the test case observes: seeded from the input's MTE_TAGS
 * section, read by LDG, and updated dynamically by the STG* family. Speculative updates are rolled
 * back by the speculation engine, which snapshots/restores this array at each checkpoint via
 * mte_tagmem_snapshot/restore (see simulation_execution_clause_hook.c). Granules are indexed by the
 * address bits [55:0] relative to the sandbox base (the tag/attribute bits are masked off). */
static uint8_t*  g_tagmem      = NULL;
static size_t    g_tagmem_n    = 0;
static uintptr_t g_tagmem_base = 0;

void mte_tagmem_init(uintptr_t base, const uint8_t* initial, size_t count) {
	free(g_tagmem);
	g_tagmem = NULL;
	g_tagmem_n = 0;
	g_tagmem_base = base & 0x00FFFFFFFFFFFFFFu;
	/* No initial tagging supplied -> not MTE-test mode -> no tag memory at all (no fallback). */
	if (NULL == initial || 0 == count) {
		return;
	}
	g_tagmem = malloc(count);
	if (NULL == g_tagmem) {
		return;
	}
	g_tagmem_n = count;
	memcpy(g_tagmem, initial, count);
}

void mte_tagmem_free(void) {
	free(g_tagmem);
	g_tagmem = NULL;
	g_tagmem_n = 0;
}

size_t mte_tagmem_bytes(void) {
	return g_tagmem_n;
}

void mte_tagmem_snapshot(uint8_t* dst) {
	if (NULL != g_tagmem) {
		memcpy(dst, g_tagmem, g_tagmem_n);
	}
}

void mte_tagmem_restore(const uint8_t* src) {
	if (NULL != g_tagmem) {
		memcpy(g_tagmem, src, g_tagmem_n);
	}
}

/* Granule index of kernel-space address `addr`, or -1 if outside the tagged span. */
static long ce_granule(uintptr_t addr) {
	uintptr_t a = addr & 0x00FFFFFFFFFFFFFFu;
	if (NULL == g_tagmem || a < g_tagmem_base) {
		return -1;
	}
	size_t g = (size_t)((a - g_tagmem_base) / 16);
	return (g < g_tagmem_n) ? (long)g : -1;
}

static uint8_t ce_tag_at(uintptr_t addr) {
	long g = ce_granule(addr);
	return (0 <= g) ? (g_tagmem[g] & 0xF) : 0;
}

static void ce_tag_set(uintptr_t addr, uint8_t tag) {
	long g = ce_granule(addr);
	if (0 <= g) {
		g_tagmem[g] = tag & 0xF;
	}
}

/* Zero the 16-byte data of the granule at kernel-space address `addr` (the STZG/STZ2G/STZGM data
 * effect). sim_state->memory[0] maps to the sandbox base, so the host offset is granule_index * 16;
 * a no-op outside the modeled sandbox span. */
static void ce_data_zero_granule(struct simulation_state* s, uintptr_t addr) {
	long g = ce_granule(addr);
	if (0 <= g) {
		memset(s->memory + (size_t)g * 16, 0, 16);
	}
}

/* Store a 16-byte register pair (x1 low, x2 high) at the granule (the STGP data effect). */
static void ce_data_store_pair(struct simulation_state* s, uintptr_t addr, uint64_t x1, uint64_t x2) {
	long g = ce_granule(addr);
	if (0 <= g) {
		memcpy(s->memory + (size_t)g * 16,     &x1, sizeof(x1));
		memcpy(s->memory + (size_t)g * 16 + 8, &x2, sizeof(x2));
	}
}

/* MTE-test mode is active iff per-input tag memory was seeded. */
int mte_tagmem_active(void) {
	return NULL != g_tagmem;
}

/* Allocation tag of the granule holding kernel-space address `addr` (for the after-access tag
 * correction in the fixup path); 0 outside the tagged span / when no tag memory exists. */
uint8_t mte_tagmem_tag_at(uintptr_t addr) {
	return ce_tag_at(addr);
}

/* STG-family / LDG immediate offset: SignExtend(bits[20:12]) * 16 (the granule size). */
static int64_t mte_mem_offset(uint32_t enc) {
	int64_t s9 = (int64_t)((int32_t)((enc >> 12) & 0x1FFu) << 23) >> 23;
	return s9 * 16;
}

/* -------------------------------------------------------------------------
 * mte_emulator_hook — intercept all MTE instructions and software-emulate them.
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
/* True if `enc` is an MTE memory-tag instruction (the LDG/STG/STGP family). These are software-
 * emulated by mte_emulator_hook and skipped from native execution, so base_hook_c must NOT apply its
 * kaddr<->uaddr translation to their base register (the plugin addresses sandbox memory itself, from
 * the architectural kaddr). The data-tag register ops (IRG/ADDG/...) are not memory accesses, so they
 * never reach that path. */
int mte_is_mem_tag_access(uint32_t enc) {
	switch (classify_mte(enc)) {
		case MTE_LDG: case MTE_LDGM:
		case MTE_STG: case MTE_STZG: case MTE_ST2G: case MTE_STZ2G:
		case MTE_STGM: case MTE_STZGM: case MTE_STGP:
			return 1;
		default:
			return 0;
	}
}

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
			 * IRG Xd, Xn, Xm: choose Xd's tag, excluding the tags in Xm's low 16 bits (and
			 * GCR_EL1.Exclude, which the CE assumes 0). The CE has no RNG and the contract must be
			 * reproducible, so it picks the lowest non-excluded tag (tag 0 if all are excluded).
			 * Address bits[55:0] and attribute bits[63:60] are copied from Xn unchanged.
			 */
			uint64_t xn = cpu_state_read_base_reg(state, rn);
			uint32_t exclude = (uint32_t)(cpu_state_read_gpr_zr(state, rm) & 0xFFFFu);
			uint32_t tag = 0;
			while (tag < 16 && (exclude & (1u << tag))) {
				++tag;
			}
			if (16 == tag) {
				tag = 0;
			}
			cpu_state_write_base_reg(state, rd,
			                         (xn & 0xF0FFFFFFFFFFFFFFu) | ((uint64_t)tag << 56));
			break;
		}
		case MTE_GMI: {
			/*
			 * GMI Xd, Xn, Xm: insert Xn's logical tag into the exclude mask Xm.
			 * Xd = Xm | (1 << Xn[59:56]).
			 */
			uint32_t tag = (uint32_t)((cpu_state_read_base_reg(state, rn) >> 56) & 0xFu);
			write_xreg(state, rd, cpu_state_read_gpr_zr(state, rm) | (1ull << tag));
			break;
		}
		case MTE_SUBP:
		case MTE_SUBPS: {
			/*
			 * SUBP[S] Xd, Xn, Xm: result = SignExtend(Xn[55:0]) - SignExtend(Xm[55:0]).
			 * The shift trick (<<8 then arithmetic >>8) moves bit 55 to bit 63 and fills
			 * bits[63:56] with the sign, avoiding UB.
			 */
			uint64_t xn56 = cpu_state_read_base_reg(state, rn) & 0x00FFFFFFFFFFFFFFu;
			uint64_t xm56 = cpu_state_read_base_reg(state, rm) & 0x00FFFFFFFFFFFFFFu;
			uint64_t op1 = (uint64_t)((int64_t)(xn56 << 8) >> 8);
			uint64_t op2 = (uint64_t)((int64_t)(xm56 << 8) >> 8);
			uint64_t result = op1 - op2;
			write_xreg(state, rd, result);
			if (mtype == MTE_SUBPS) {
				/* NZCV from AddWithCarry(op1, NOT op2, 1): C = no borrow, V = signed overflow. */
				uint32_t N = (uint32_t)(result >> 63);
				uint32_t Z = (result == 0);
				uint32_t C = (op1 >= op2);
				uint32_t V = (uint32_t)(((op1 ^ op2) & (op1 ^ result)) >> 63);
				state->nzcv = ((uint64_t)N << 31) | ((uint64_t)Z << 30)
				            | ((uint64_t)C << 29) | ((uint64_t)V << 28);
			}
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
			 * LDG Xt, [Xn, #simm]: load the addressed granule's allocation tag into Xt[59:56],
			 * leaving the other bits of Xt unchanged. simm = SignExtend(bits[20:12]) * 16.
			 */
			uintptr_t addr = cpu_state_read_base_reg(state, rn) + (uintptr_t)mte_mem_offset(enc);
			uint64_t old = cpu_state_read_gpr_zr(state, rd);
			write_xreg(state, rd, (old & ~(0xFull << 56)) | ((uint64_t)ce_tag_at(addr) << 56));
			break;
		}
		case MTE_LDGM: {
			/*
			 * LDGM Xt, [Xn]: load the allocation tags of the block containing Xn into Xt, one tag
			 * per 4-bit field (MTE_TAG_BLOCK_GRANULES tags, block-aligned).
			 */
			uintptr_t block = cpu_state_read_base_reg(state, rn) & ~((uintptr_t)(MTE_TAG_BLOCK_BYTES - 1));
			uint64_t xt = 0;
			for (uint32_t g = 0; g < MTE_TAG_BLOCK_GRANULES; ++g) {
				xt |= (uint64_t)ce_tag_at(block + (uintptr_t)g * 16) << (4 * g);
			}
			write_xreg(state, rd, xt);
			break;
		}
		case MTE_STG:
		case MTE_STZG:
		case MTE_ST2G:
		case MTE_STZ2G: {
			/*
			 * Store Allocation Tag: write Xt's logical tag (Xt[59:56]) to the addressed granule;
			 * the 2G forms tag two consecutive granules. The STZ* forms also zero the granule data.
			 * Pre/post-index writes Xn back. Speculative tag and data updates roll back via the
			 * speculation engine's checkpoint (tag memory + sandbox data).
			 */
			int64_t off  = mte_mem_offset(enc);
			uint32_t mode = (enc >> 10) & 0x3u;
			int two = (MTE_ST2G == mtype || MTE_STZ2G == mtype);
			int zero = (MTE_STZG == mtype || MTE_STZ2G == mtype);
			uintptr_t basev = cpu_state_read_base_reg(state, rn);
			uintptr_t addr  = (0x1u == mode) ? basev : basev + (uintptr_t)off;   /* post uses base */
			uint8_t tag = (uint8_t)((cpu_state_read_base_reg(state, rd) >> 56) & 0xFu);
			ce_tag_set(addr, tag);
			if (zero) {
				ce_data_zero_granule(sim_state, addr);
			}
			if (two) {
				ce_tag_set(addr + 16, tag);
				if (zero) {
					ce_data_zero_granule(sim_state, addr + 16);
				}
			}
			if (0x1u == mode || 0x3u == mode) {   /* post/pre-index writeback */
				cpu_state_write_base_reg(state, rn, basev + (uintptr_t)off);
			}
			break;
		}
		case MTE_STGM:
		case MTE_STZGM: {
			/*
			 * STGM/STZGM Xt, [Xn]: write the block's allocation tags from Xt's 4-bit fields, one
			 * per granule (MTE_TAG_BLOCK_GRANULES tags, block-aligned). STZGM also zeros the block's
			 * data.
			 */
			uintptr_t block = cpu_state_read_base_reg(state, rn) & ~((uintptr_t)(MTE_TAG_BLOCK_BYTES - 1));
			uint64_t xt = cpu_state_read_base_reg(state, rd);
			for (uint32_t g = 0; g < MTE_TAG_BLOCK_GRANULES; ++g) {
				ce_tag_set(block + (uintptr_t)g * 16, (uint8_t)((xt >> (4 * g)) & 0xFu));
				if (MTE_STZGM == mtype) {
					ce_data_zero_granule(sim_state, block + (uintptr_t)g * 16);
				}
			}
			break;
		}
		case MTE_STGP: {
			/*
			 * STGP Xt1, Xt2, [Xn{, #simm}]: store the register pair (16 bytes) AND set the addressed
			 * granule's tag from Xn's logical tag. simm7 (bits[21:15]) sign-extended * 16; index mode
			 * bits[24:23]: 01 post-index, 10 signed offset, 11 pre-index (writeback).
			 */
			int64_t s7  = (int64_t)((int32_t)((enc >> 15) & 0x7Fu) << 25) >> 25;
			int64_t off = s7 * 16;
			uint32_t idx = (enc >> 23) & 0x3u;
			uint32_t rt2 = (enc >> 10) & 0x1Fu;
			uintptr_t basev = cpu_state_read_base_reg(state, rn);
			uintptr_t addr  = (0x1u == idx) ? basev : basev + (uintptr_t)off;   /* post uses base */
			ce_tag_set(addr, (uint8_t)((basev >> 56) & 0xFu));
			ce_data_store_pair(sim_state, addr, cpu_state_read_gpr_zr(state, rd),
			                   cpu_state_read_gpr_zr(state, rt2));
			if (0x1u == idx || 0x3u == idx) {   /* post/pre-index writeback */
				cpu_state_write_base_reg(state, rn, basev + (uintptr_t)off);
			}
			break;
		}
		default:
			break;
	}

	return (void*)(sim_state->cpu_state.pc + 4);
}
