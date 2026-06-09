#include "utils.h"

bool get_bit(uint32_t val, int n) {
	return (val >> n) & 1;
}
