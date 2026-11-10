#include "main.h"

void *invalidate_bpu_entries(void) {
    static size_t current_view = 0;
    size_t idx = current_view % MAX_MEASUREMENT_VIEWS;
    ++current_view;
    return executor.measurement_code_views[idx];
}

void flush_bpu_phr(void) {
    /* Official ARM Spectre-BHB mitigation sequence (see ARM-EPM-048486 v4).
     * PHR on Neoverse N3: 75 records × 4 bits/record = 300 bits (shift register).
     * Each taken branch shifts in a new 4-bit footprint. 75 taken branches fully
     * overwrite all 300 bits of old history.
     *
     * K=38 iterations: "b 2f" always taken (38 updates). "b.ne 1b" taken when
     * x0 ≠ 0 after subs (37 times).  Total = 38+37 = 75 taken-branch updates. */
    asm volatile (
        "mov x0, #38\n"
        "1:\n"
        "b 2f\n"
        "2:\n"
        "subs x0, x0, #1\n"
        "b.ne 1b\n"
        "dsb nsh\n"
        "isb\n"
        :
        :
        : "x0", "cc"
    );
}

