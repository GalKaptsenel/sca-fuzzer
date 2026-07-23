#ifndef CE_QARMA_H
#define CE_QARMA_H

#include <stdint.h>
#include <stdbool.h>

/* Architected ARM pointer-auth (QARMA3/QARMA5), ported from QEMU target/arm/tcg/pauth_helper.c.
 * Bit-exact against real hardware for the architected algorithm. */

/* Target PAC parameters. iterations = 2 (QARMA3) or 4 (QARMA5); tsz = 64 - VA_size; tbi0/tbi1 =
 * top-byte-ignore for the low (TTBR0) / high (TTBR1) VA half, selected per pointer by bit 55;
 * pauth2 = FEAT_PAuth2 (EnhancedPAC2, i.e. APA/APA3 >= 3). */
struct pac_profile {
    int iterations;
    int tsz;
    int tbi0;
    int tbi1;
    bool pauth2;
};

/* The raw QARMA MAC (before pointer-field insertion). key0 = key_hi, key1 = key_lo. */
uint64_t qarma_computepac(uint64_t data, uint64_t modifier,
                          uint64_t key_lo, uint64_t key_hi, int iterations);

/* Sign a pointer: insert the PAC into the field bits per `p` (the AddPAC pseudocode). */
uint64_t qarma_addpac(uint64_t ptr, uint64_t modifier,
                      uint64_t key_lo, uint64_t key_hi, struct pac_profile p);

/* Strip the PAC field back to the canonical pointer (XPAC). */
uint64_t qarma_strip(uint64_t ptr, struct pac_profile p);

#endif /* CE_QARMA_H */
