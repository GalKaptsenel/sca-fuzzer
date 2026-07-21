#include "cache_geometry.h"

struct cache_geometry decode_cache_geometry(unsigned long long clidr, unsigned long long ccsidr,
                                            int ccidx, unsigned int level, enum cache_type requested) {
	struct cache_geometry g;
	unsigned int ctype = (unsigned int)((clidr >> (3 * (level - 1))) & 0x7);

	g.level = level;
	g.cpu = -1;
	switch (ctype) {
	case 4:  g.type = CACHE_TYPE_UNIFIED;     break;
	case 3:  g.type = requested;              break;
	case 2:  g.type = CACHE_TYPE_DATA;        break;
	case 1:  g.type = CACHE_TYPE_INSTRUCTION; break;
	default: g.type = CACHE_TYPE_NONE;        break;
	}
	g.line  = 16u << (ccsidr & 0x7);
	g.assoc = (unsigned int)(ccidx ? ((ccsidr >> 3) & 0x1fffffULL) : ((ccsidr >> 3) & 0x3ffULL)) + 1u;
	g.sets  = (unsigned int)(ccidx ? ((ccsidr >> 32) & 0xffffffULL) : ((ccsidr >> 13) & 0x7fffULL)) + 1u;
	g.size  = g.line * g.assoc * g.sets;
	g.valid = (g.type != CACHE_TYPE_NONE);
	return g;
}
