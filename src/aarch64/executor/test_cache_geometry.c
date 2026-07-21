/* Host-compiled unit test for the cache-geometry decode. Compiles the REAL decode (cache_geometry.c),
 * not a copy, and feeds it known CLIDR_EL1 / CCSIDR_EL1 values. Oracle: the ARM ARM register layout and
 * hand-computed size = line * assoc * sets. No kernel, no hardware. */
#include <stdio.h>
#include "cache_geometry.h"

static int failed = 0;
#define CHECK(cond) do { \
	if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, #cond); ++failed; } \
} while (0)

/* Build the register fields per the ARM ARM. CCSIDR_EL1: LineSize[2:0] = log2(bytes)-4; non-CCIDX
 * Assoc[12:3]/Sets[27:13]; CCIDX Assoc[23:3]/Sets[55:32]. CLIDR_EL1: CtypeN at bits [3N-1:3N-3]. */
#define LINELOG(b)      ((b)==16?0 : (b)==32?1 : (b)==64?2 : (b)==128?3 : 4)
#define CCSIDR(sets, ways, b)   (((unsigned long long)((sets)-1) << 13) | ((unsigned long long)((ways)-1) << 3) | LINELOG(b))
#define CCSIDR_X(sets, ways, b) (((unsigned long long)((sets)-1) << 32) | ((unsigned long long)((ways)-1) << 3) | LINELOG(b))
#define CLIDR(level, ctype)     ((unsigned long long)((ctype) & 0x7) << (3 * ((level) - 1)))

int main(void) {
	struct cache_geometry g;

	/* Cortex-X3 L1D: 64 KB, 4-way, 256 sets, 64 B line; separate I+D (ctype 3), non-CCIDX. */
	g = decode_cache_geometry(CLIDR(1, 3), CCSIDR(256, 4, 64), 0, 1, CACHE_TYPE_DATA);
	CHECK(g.line == 64); CHECK(g.assoc == 4); CHECK(g.sets == 256);
	CHECK(g.size == 64u * 1024u); CHECK(g.type == CACHE_TYPE_DATA); CHECK(g.valid);

	/* Cortex-A510 L1D: 32 KB, 4-way, 128 sets -- the same binary must decode this too. */
	g = decode_cache_geometry(CLIDR(1, 3), CCSIDR(128, 4, 64), 0, 1, CACHE_TYPE_DATA);
	CHECK(g.size == 32u * 1024u); CHECK(g.sets == 128); CHECK(g.assoc == 4);

	/* CCIDX layout (Assoc[23:3], Sets[55:32]) must yield the same 64 KB geometry. */
	g = decode_cache_geometry(CLIDR(1, 3), CCSIDR_X(256, 4, 64), 1, 1, CACHE_TYPE_DATA);
	CHECK(g.assoc == 4); CHECK(g.sets == 256); CHECK(g.size == 64u * 1024u);

	/* Non-64 B line: 32 B. */
	g = decode_cache_geometry(CLIDR(1, 3), CCSIDR(256, 4, 32), 0, 1, CACHE_TYPE_DATA);
	CHECK(g.line == 32); CHECK(g.size == 32u * 1024u);

	/* Unified L1 (ctype 4): UNIFIED regardless of the requested side. */
	g = decode_cache_geometry(CLIDR(1, 4), CCSIDR(256, 4, 64), 0, 1, CACHE_TYPE_DATA);
	CHECK(g.type == CACHE_TYPE_UNIFIED);

	/* Separate cache: the requested side is echoed. */
	g = decode_cache_geometry(CLIDR(1, 3), CCSIDR(256, 4, 64), 0, 1, CACHE_TYPE_INSTRUCTION);
	CHECK(g.type == CACHE_TYPE_INSTRUCTION);

	/* Data-only cache (ctype 2): DATA regardless of request. */
	g = decode_cache_geometry(CLIDR(1, 2), CCSIDR(256, 4, 64), 0, 1, CACHE_TYPE_INSTRUCTION);
	CHECK(g.type == CACHE_TYPE_DATA);

	/* No cache at this level (ctype 0): NONE, invalid. */
	g = decode_cache_geometry(CLIDR(1, 0), CCSIDR(256, 4, 64), 0, 1, CACHE_TYPE_DATA);
	CHECK(g.type == CACHE_TYPE_NONE); CHECK(!g.valid);

	/* Level indexing: the L2 Ctype lives at CLIDR bits [5:3], not [2:0]. */
	g = decode_cache_geometry(CLIDR(2, 4) | CLIDR(1, 3), CCSIDR(1024, 8, 64), 0, 2, CACHE_TYPE_DATA);
	CHECK(g.type == CACHE_TYPE_UNIFIED); CHECK(g.assoc == 8); CHECK(g.sets == 1024);

	/* Unrelated high bits must not bleed into the non-CCIDX Assoc/Sets fields. */
	g = decode_cache_geometry(CLIDR(1, 3), CCSIDR(256, 4, 64) | (0xffffffffULL << 32), 0, 1, CACHE_TYPE_DATA);
	CHECK(g.sets == 256); CHECK(g.assoc == 4);

	printf(failed ? "cache_geometry: %d check(s) FAILED\n" : "cache_geometry: all checks passed\n", failed);
	return failed ? 1 : 0;
}
