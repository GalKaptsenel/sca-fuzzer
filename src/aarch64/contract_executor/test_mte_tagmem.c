/*
 * Unit tests for the CE's MTE tag-memory emulation (mte_tag_plugin.c): every STG and LDG variant
 * reads/writes the per-input tag memory correctly, the multi-granule block is GMID_EL1.BS=4 (16
 * granules), and the speculation snapshot/restore rolls tag writes back. Pure software (no device).
 *
 * Build: make test_mte_tagmem   Run: ./test_mte_tagmem
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mte_tag_plugin.h"
#include "simulation_state.h"

/* mte_tag_plugin.c declares `extern struct simulation simulation;` via its header but never uses it
 * on the tag path; provide the symbol so the test links standalone. */
struct simulation simulation;

static int g_run = 0;
static int g_fail = 0;
#define CHECK(cond, ...) do { \
	g_run++; \
	if (!(cond)) { g_fail++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

#define BASE   0x100000ull
#define NGRAN  64u                 /* granules of tag memory (1 KB of sandbox) */

static struct simulation_state SS;
static uint32_t INSTR;

static void set_reg(uint32_t n, uint64_t v) { cpu_state_write_base_reg(&SS.cpu_state, n, (uintptr_t)v); }
static uint64_t get_reg(uint32_t n)          { return (uint64_t)cpu_state_read_base_reg(&SS.cpu_state, n); }

static void run(uint32_t enc) {
	INSTR = enc;
	SS.cpu_state.pc = (uintptr_t)&INSTR;
	mte_emulator_hook(&SS);
}

/* STG-family / LDG / *GM encoder (0xD9 group). top3=bits[23:21], mode=bits[11:10]
 * (offset=2, post-index=1, pre-index=3; the *GM and LDG forms use mode=0). */
static uint32_t e(uint32_t top3, uint32_t mode, uint32_t xt, uint32_t xn, int32_t imm) {
	uint32_t s9 = ((uint32_t)(imm / 16)) & 0x1FFu;
	return (0xD9u << 24) | (top3 << 21) | (s9 << 12) | (mode << 10) | (xn << 5) | xt;
}
static uint32_t e_stgp(uint32_t xt1, uint32_t xt2, uint32_t xn, int32_t imm) {
	uint32_t s7 = ((uint32_t)(imm / 16)) & 0x7Fu;   /* offset form (prefix 0x69, idx bits = 10) */
	return (0x69u << 24) | (s7 << 15) | (xt2 << 10) | (xn << 5) | xt1;
}

/* Data-tag register-result instructions. */
static uint32_t e_irg (uint32_t xd, uint32_t xn, uint32_t xm) { return 0x9AC01000u | (xm << 16) | (xn << 5) | xd; }
static uint32_t e_gmi (uint32_t xd, uint32_t xn, uint32_t xm) { return 0x9AC01400u | (xm << 16) | (xn << 5) | xd; }
static uint32_t e_subp(uint32_t xd, uint32_t xn, uint32_t xm) { return 0x9AC00000u | (xm << 16) | (xn << 5) | xd; }
static uint32_t e_subps(uint32_t xd, uint32_t xn, uint32_t xm){ return 0xBAC00000u | (xm << 16) | (xn << 5) | xd; }
static uint32_t e_addg(uint32_t xd, uint32_t xn, uint32_t u6, uint32_t u4) {
	return 0x91800000u | ((u6 & 0x3Fu) << 16) | ((u4 & 0xFu) << 10) | (xn << 5) | xd;
}
static uint32_t e_subg(uint32_t xd, uint32_t xn, uint32_t u6, uint32_t u4) {
	return 0xD1800000u | ((u6 & 0x3Fu) << 16) | ((u4 & 0xFu) << 10) | (xn << 5) | xd;
}
#define LDG(xt,xn,imm)    e(3,0,(xt),(xn),(imm))
#define STG(xt,xn,imm)    e(1,2,(xt),(xn),(imm))
#define STG_PRE(xt,xn,imm)  e(1,3,(xt),(xn),(imm))
#define STG_POST(xt,xn,imm) e(1,1,(xt),(xn),(imm))
#define STZG(xt,xn,imm)   e(3,2,(xt),(xn),(imm))
#define ST2G(xt,xn,imm)   e(5,2,(xt),(xn),(imm))
#define STZ2G(xt,xn,imm)  e(7,2,(xt),(xn),(imm))
#define STGM(xt,xn)       e(5,0,(xt),(xn),0)
#define STZGM(xt,xn)      e(1,0,(xt),(xn),0)
#define LDGM(xt,xn)       e(7,0,(xt),(xn),0)

static uint8_t g_initial[NGRAN];
static uint8_t g_data[NGRAN * 16];   /* the simulated sandbox data memory (memory[0] == BASE) */

static void reset_tagmem(void) {
	for (uint32_t i = 0; i < NGRAN; ++i) {
		g_initial[i] = (uint8_t)(i & 0xF);   /* granule i seeded with tag i%16 */
	}
	mte_tagmem_init(BASE, g_initial, NGRAN);
	memset(&SS, 0, sizeof(SS));
	memset(g_data, 0xFF, sizeof(g_data));    /* non-zero so STZ* zeroing is observable */
	SS.memory = g_data;
}

static uint64_t data_u64(uint32_t granule, uint32_t byte_off) {
	uint64_t v;
	memcpy(&v, g_data + (size_t)granule * 16 + byte_off, sizeof(v));
	return v;
}

static uint8_t tag_of(uint32_t granule) {
	uint8_t buf[NGRAN];
	mte_tagmem_snapshot(buf);
	return buf[granule] & 0xF;
}

int main(void) {
	/* ---- LDG reads the seeded tag, preserving Xt's other bits ---- */
	reset_tagmem();
	set_reg(1, BASE + 5 * 16);
	set_reg(2, 0xDEADBEEF00000000ull);
	run(LDG(2, 1, 0));
	CHECK(((get_reg(2) >> 56) & 0xF) == 5, "LDG tag: got %u want 5", (unsigned)((get_reg(2) >> 56) & 0xF));
	CHECK((get_reg(2) & 0x00FFFFFFFFFFFFFFull) == 0x00ADBEEF00000000ull, "LDG must preserve non-tag bits");

	/* ---- STG writes Xt's tag; LDG reads it back ---- */
	reset_tagmem();
	set_reg(1, BASE + 7 * 16);
	set_reg(2, (uint64_t)0xA << 56);
	run(STG(2, 1, 0));
	CHECK(tag_of(7) == 0xA, "STG: granule 7 tag got %u want 0xA", tag_of(7));
	set_reg(3, 0);
	run(LDG(3, 1, 0));
	CHECK(((get_reg(3) >> 56) & 0xF) == 0xA, "STG->LDG round-trip");

	/* ---- STG with an immediate offset ---- */
	reset_tagmem();
	set_reg(1, BASE);
	set_reg(2, (uint64_t)0x3 << 56);
	run(STG(2, 1, 9 * 16));
	CHECK(tag_of(9) == 0x3, "STG #offset: granule 9");

	/* ---- ST2G / STZ2G tag two consecutive granules ---- */
	reset_tagmem();
	set_reg(1, BASE + 3 * 16);
	set_reg(2, (uint64_t)0xC << 56);
	run(ST2G(2, 1, 0));
	CHECK(tag_of(3) == 0xC && tag_of(4) == 0xC, "ST2G: granules 3,4");

	/* ---- STZG sets the tag AND zeroes the granule data; STZ2G does two granules ---- */
	reset_tagmem();
	set_reg(1, BASE + 11 * 16);
	set_reg(2, (uint64_t)0x6 << 56);
	run(STZG(2, 1, 0));
	CHECK(tag_of(11) == 0x6, "STZG: granule 11 tag");
	CHECK(data_u64(11, 0) == 0 && data_u64(11, 8) == 0, "STZG: granule 11 data zeroed");
	CHECK(data_u64(10, 0) == 0xFFFFFFFFFFFFFFFFull, "STZG: neighbour granule untouched");

	reset_tagmem();
	set_reg(1, BASE + 14 * 16);
	set_reg(2, (uint64_t)0x2 << 56);
	run(STZ2G(2, 1, 0));
	CHECK(tag_of(14) == 0x2 && tag_of(15) == 0x2, "STZ2G: tags 14,15");
	CHECK(data_u64(14, 0) == 0 && data_u64(15, 0) == 0, "STZ2G: granules 14,15 data zeroed");

	/* ---- pre/post-index writeback updates Xn ---- */
	reset_tagmem();
	set_reg(1, BASE);
	set_reg(2, (uint64_t)0x4 << 56);
	run(STG_PRE(2, 1, 2 * 16));   /* addr = Xn + 32, Xn := Xn + 32 */
	CHECK(tag_of(2) == 0x4, "STG pre-index: tag at Xn+offset");
	CHECK(get_reg(1) == BASE + 2 * 16, "STG pre-index: Xn writeback");

	reset_tagmem();
	set_reg(1, BASE + 6 * 16);
	set_reg(2, (uint64_t)0x5 << 56);
	run(STG_POST(2, 1, 16));      /* addr = Xn (granule 6), Xn := Xn + 16 */
	CHECK(tag_of(6) == 0x5, "STG post-index: tag at base");
	CHECK(get_reg(1) == BASE + 7 * 16, "STG post-index: Xn writeback");

	/* ---- STGM writes a 16-granule (GMID_EL1.BS=4) block from Xt's nibbles; the next block is
	 *      untouched -- which also verifies the block size is exactly 16 granules ---- */
	reset_tagmem();
	set_reg(1, BASE);                       /* block 0 = granules 0..15 */
	set_reg(2, 0xFEDCBA9876543210ull);      /* nibble g -> granule g */
	run(STGM(2, 1));
	int stgm_ok = 1;
	for (uint32_t g = 0; g < 16; ++g) {
		if (tag_of(g) != ((0xFEDCBA9876543210ull >> (4 * g)) & 0xF)) { stgm_ok = 0; }
	}
	CHECK(stgm_ok, "STGM: 16 nibbles -> 16 granules");
	CHECK(tag_of(16) == (16 & 0xF), "STGM block size is 16 granules (granule 16 untouched)");

	/* ---- LDGM reads the block back into Xt's nibbles ---- */
	set_reg(3, 0);
	run(LDGM(3, 1));
	CHECK(get_reg(3) == 0xFEDCBA9876543210ull, "LDGM: block -> Xt nibbles got %#018lx",
	      (unsigned long)get_reg(3));

	/* ---- STGP stores the register pair AND sets the granule's tag from Xn's tag ---- */
	reset_tagmem();
	set_reg(1, ((uint64_t)0x9 << 56) | (BASE + 2 * 16));   /* Xn tag = 9 */
	set_reg(4, 0x1111111111111111ull);                     /* Xt1 */
	set_reg(5, 0x2222222222222222ull);                     /* Xt2 */
	run(e_stgp(4, 5, 1, 0));
	CHECK(tag_of(2) == 0x9, "STGP: granule tag from Xn (got %u want 9)", tag_of(2));
	CHECK(data_u64(2, 0) == 0x1111111111111111ull, "STGP: Xt1 stored");
	CHECK(data_u64(2, 8) == 0x2222222222222222ull, "STGP: Xt2 stored");

	/* ---- speculation rollback: snapshot, mutate, restore reverts the tag ---- */
	reset_tagmem();
	uint8_t snap[NGRAN];
	mte_tagmem_snapshot(snap);
	set_reg(1, BASE + 20 * 16);
	set_reg(2, (uint64_t)0xF << 56);
	run(STG(2, 1, 0));
	CHECK(tag_of(20) == 0xF, "rollback: pre-restore write visible");
	mte_tagmem_restore(snap);
	CHECK(tag_of(20) == (20 & 0xF), "rollback: restore reverts the speculative STG");

	/* ---- IRG picks the lowest tag not excluded by Xm; address bits preserved ---- */
	memset(&SS, 0, sizeof(SS));
	set_reg(1, ((uint64_t)0x5 << 56) | 0x2000ull);   /* Xn: tag 5, addr 0x2000 */
	set_reg(3, 0x0);                                  /* Xm exclude = none */
	run(e_irg(2, 1, 3));
	CHECK(((get_reg(2) >> 56) & 0xF) == 0, "IRG exclude=0 -> tag 0");
	CHECK((get_reg(2) & 0x00FFFFFFFFFFFFFFull) == 0x2000ull, "IRG preserves address bits");
	set_reg(3, 0x1);   /* exclude tag 0 */
	run(e_irg(2, 1, 3));
	CHECK(((get_reg(2) >> 56) & 0xF) == 1, "IRG exclude={0} -> tag 1");
	set_reg(3, 0x7);   /* exclude tags 0,1,2 */
	run(e_irg(2, 1, 3));
	CHECK(((get_reg(2) >> 56) & 0xF) == 3, "IRG exclude={0,1,2} -> tag 3");

	/* ---- GMI inserts Xn's tag into the exclude mask Xm ---- */
	memset(&SS, 0, sizeof(SS));
	set_reg(1, (uint64_t)0x3 << 56);   /* Xn tag = 3 */
	set_reg(3, 0x2);                    /* Xm = bit1 set */
	run(e_gmi(2, 1, 3));
	CHECK(get_reg(2) == (0x2ull | (1ull << 3)), "GMI: Xm | (1<<Xn.tag) = 0xA, got %#lx",
	      (unsigned long)get_reg(2));

	/* ---- ADDG / SUBG update address and tag independently (no carry between them) ---- */
	memset(&SS, 0, sizeof(SS));
	set_reg(1, ((uint64_t)0x1 << 56) | 0x1000ull);
	run(e_addg(2, 1, 2, 3));   /* addr += 2*16, tag += 3 */
	CHECK(get_reg(2) == (((uint64_t)0x4 << 56) | 0x1020ull), "ADDG: tag 4, addr 0x1020, got %#lx",
	      (unsigned long)get_reg(2));
	set_reg(1, ((uint64_t)0x8 << 56) | 0x1000ull);
	run(e_subg(2, 1, 1, 3));   /* addr -= 16, tag -= 3 */
	CHECK(get_reg(2) == (((uint64_t)0x5 << 56) | 0x0FF0ull), "SUBG: tag 5, addr 0x0FF0, got %#lx",
	      (unsigned long)get_reg(2));

	/* ---- SUBP / SUBPS: 56-bit signed pointer difference ---- */
	memset(&SS, 0, sizeof(SS));
	set_reg(1, 0x1000ull);
	set_reg(3, 0x0F00ull);
	run(e_subp(2, 1, 3));
	CHECK(get_reg(2) == 0x100ull, "SUBP: 0x1000-0x0F00 = 0x100, got %#lx", (unsigned long)get_reg(2));
	run(e_subps(2, 1, 3));
	CHECK(get_reg(2) == 0x100ull, "SUBPS result");
	CHECK(SS.cpu_state.nzcv == (1ull << 29), "SUBPS flags: C set only (op1>=op2), got %#lx",
	      (unsigned long)SS.cpu_state.nzcv);

	/* ---- no MTE mode (no initial tags) -> no tag memory, LDG reads 0 ---- */
	mte_tagmem_init(BASE, NULL, 0);
	CHECK(mte_tagmem_bytes() == 0, "no-MTE-mode: no tag memory");
	memset(&SS, 0, sizeof(SS));
	set_reg(1, BASE);
	set_reg(2, 0);
	run(LDG(2, 1, 0));
	CHECK(((get_reg(2) >> 56) & 0xF) == 0, "no-MTE-mode: LDG reads 0");
	mte_tagmem_free();

	/* mte_is_mem_tag_access: every memory-tag form is recognised (base_hook skips its kaddr<->uaddr
	 * translation for them); data-tag register ops and ordinary loads/stores are not. */
	CHECK(mte_is_mem_tag_access(LDG(1, 1, 0))   == 1, "is_mem_tag: LDG");
	CHECK(mte_is_mem_tag_access(LDGM(1, 1))     == 1, "is_mem_tag: LDGM");
	CHECK(mte_is_mem_tag_access(STG(0, 0, 0))   == 1, "is_mem_tag: STG");
	CHECK(mte_is_mem_tag_access(STG_PRE(0, 0, 16))  == 1, "is_mem_tag: STG (pre-index)");
	CHECK(mte_is_mem_tag_access(STG_POST(0, 0, 16)) == 1, "is_mem_tag: STG (post-index)");
	CHECK(mte_is_mem_tag_access(STZG(0, 0, 0))  == 1, "is_mem_tag: STZG");
	CHECK(mte_is_mem_tag_access(ST2G(0, 0, 0))  == 1, "is_mem_tag: ST2G");
	CHECK(mte_is_mem_tag_access(STZ2G(0, 0, 0)) == 1, "is_mem_tag: STZ2G");
	CHECK(mte_is_mem_tag_access(STGM(0, 0))     == 1, "is_mem_tag: STGM");
	CHECK(mte_is_mem_tag_access(STZGM(0, 0))    == 1, "is_mem_tag: STZGM");
	CHECK(mte_is_mem_tag_access(e_stgp(0, 1, 0, 0)) == 1, "is_mem_tag: STGP");
	/* LDPSW shares STGP's top byte but has L(bit22)=1: a pair load, not a tag store */
	CHECK(mte_is_mem_tag_access(e_stgp(0, 1, 0, 0) | (1u << 22)) == 0, "is_mem_tag: LDPSW excluded");
	CHECK(mte_is_mem_tag_access(e_addg(2, 1, 2, 3)) == 0, "is_mem_tag: ADDG is not a memory access");
	CHECK(mte_is_mem_tag_access(e_subg(2, 1, 1, 3)) == 0, "is_mem_tag: SUBG is not a memory access");
	CHECK(mte_is_mem_tag_access(e_irg(2, 1, 3))     == 0, "is_mem_tag: IRG is not a memory access");
	CHECK(mte_is_mem_tag_access(e_gmi(2, 1, 3))     == 0, "is_mem_tag: GMI is not a memory access");
	CHECK(mte_is_mem_tag_access(e_subp(2, 1, 3))    == 0, "is_mem_tag: SUBP is not a memory access");
	CHECK(mte_is_mem_tag_access(0xF9400001u) == 0, "is_mem_tag: ordinary LDR x1,[x0] excluded");
	CHECK(mte_is_mem_tag_access(0xF9000001u) == 0, "is_mem_tag: ordinary STR x1,[x0] excluded");
	CHECK(mte_is_mem_tag_access(0xA9400400u) == 0, "is_mem_tag: ordinary LDP excluded");
	CHECK(mte_is_mem_tag_access(0xD503201Fu) == 0, "is_mem_tag: NOP excluded");

	printf("%d checks, %d failed\n", g_run, g_fail);
	return g_fail ? 1 : 0;
}
