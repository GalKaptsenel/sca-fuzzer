#ifndef CE_QARMA_H
#define CE_QARMA_H

#include <stdint.h>
#include <stdbool.h>

/* Architected ARM pointer-auth (QARMA3/QARMA5): ComputePAC ported from QEMU, AddPAC/Strip per the ARM
 * ARM shared pseudocode (aarch64/functions/pac). Bit-exact against real hardware. */

/* The EL1 PAC state AddPAC/Strip/ComputePAC read. iterations / generic_iterations = 2 (QARMA3) or 4
 * (QARMA5) for address / generic (PACGA) auth, generic 0 = no modeled generic algorithm; level = the
 * APA/APA3 feature value (1 PAuth, 2 EPAC, 3 PAuth2, 4 FPAC, 5 FPACCOMBINE); t0sz/t1sz, tbi, tbid =
 * TCR_EL1 fields; constpacfield = FEAT_CONSTPACFIELD. */
struct pac_profile {
    int iterations;
    int generic_iterations;
    int level;
    int t0sz;
    int t1sz;
    int tbi0;
    int tbi1;
    int tbid0;
    int tbid1;
    bool constpacfield;
};

/* The raw QARMA MAC (before pointer-field insertion). key0 = key_hi, key1 = key_lo. */
uint64_t qarma_computepac(uint64_t data, uint64_t modifier,
                          uint64_t key_lo, uint64_t key_hi, int iterations);

/* ARM AddPAC: insert the PAC of `ptr` (with good extension bits) into its PAC field. */
uint64_t qarma_addpac(uint64_t ptr, uint64_t modifier,
                      uint64_t key_lo, uint64_t key_hi, struct pac_profile p, int is_instr);

/* ARM Strip (XPAC): replace the PAC field with copies of ptr<55>. */
uint64_t qarma_strip(uint64_t ptr, struct pac_profile p, int is_instr);

#endif /* CE_QARMA_H */
