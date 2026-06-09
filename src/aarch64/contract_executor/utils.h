#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdbool.h>

/* Extract bit `n` of `val`. */
bool get_bit(uint32_t val, int n);

#endif // UTILS_H
