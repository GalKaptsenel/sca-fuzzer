#include "pac_sign_plugin.h"
#include "simulation_state.h"
#include "simulation_execution_clause_hook.h"

#include "userapi/executor_pac_api.h"
#include "qarma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PACGA_MASK   0xFFE0FC00U
#define PACGA_BASE   0x9AC03000U

#define PAC2_MASK    0xFFFFFC00U
#define PACIA_BASE   0xDAC10000U
#define PACIB_BASE   0xDAC10400U
#define PACDA_BASE   0xDAC10800U
#define PACDB_BASE   0xDAC10C00U

#define XPAC_MASK    0xFFFFFFE0U
#define XPACI_BASE   0xDAC143E0U
#define XPACD_BASE   0xDAC147E0U

#define PACZ_MASK    0xFFFFFFE0U
#define PACIZA_BASE  0xDAC123E0U
#define PACIZB_BASE  0xDAC127E0U
#define PACDZA_BASE  0xDAC12BE0U
#define PACDZB_BASE  0xDAC12FE0U

typedef enum {
    PAC_NONE,
    PAC_IA,  PAC_IB,  PAC_DA,  PAC_DB,
    PAC_IZA, PAC_IZB, PAC_DZA, PAC_DZB
} pac_type_t;

#define AUTH2_MASK   0xFFFFFC00U
#define AUTIA_BASE   0xDAC11000U
#define AUTIB_BASE   0xDAC11400U
#define AUTDA_BASE   0xDAC11800U
#define AUTDB_BASE   0xDAC11C00U

#define AUTHZ_MASK   0xFFFFFFE0U
#define AUTIZA_BASE  0xDAC133E0U
#define AUTIZB_BASE  0xDAC137E0U
#define AUTDZA_BASE  0xDAC13BE0U
#define AUTDZB_BASE  0xDAC13FE0U

typedef enum {
    AUTH_NONE,
    AUTH_IA,  AUTH_IB,  AUTH_DA,  AUTH_DB,
    AUTH_IZA, AUTH_IZB, AUTH_DZA, AUTH_DZB
} auth_type_t;

static pac_type_t classify_pac(uint32_t inst, uint32_t *rd_out, uint32_t *rn_out)
{
    uint32_t rd = inst & 0x1F;
    uint32_t rn = (inst >> 5) & 0x1F;

    switch (inst & PAC2_MASK) {
        case PACIA_BASE: { *rd_out = rd; *rn_out = rn; return PAC_IA; }
        case PACIB_BASE: { *rd_out = rd; *rn_out = rn; return PAC_IB; }
        case PACDA_BASE: { *rd_out = rd; *rn_out = rn; return PAC_DA; }
        case PACDB_BASE: { *rd_out = rd; *rn_out = rn; return PAC_DB; }
    }
    switch (inst & PACZ_MASK) {
        case PACIZA_BASE: { *rd_out = rd; *rn_out = 31; return PAC_IZA; }
        case PACIZB_BASE: { *rd_out = rd; *rn_out = 31; return PAC_IZB; }
        case PACDZA_BASE: { *rd_out = rd; *rn_out = 31; return PAC_DZA; }
        case PACDZB_BASE: { *rd_out = rd; *rn_out = 31; return PAC_DZB; }
    }
    return PAC_NONE;
}

static auth_type_t classify_auth(uint32_t inst, uint32_t *rd_out, uint32_t *rn_out)
{
    uint32_t rd = inst & 0x1F;
    uint32_t rn = (inst >> 5) & 0x1F;

    switch (inst & AUTH2_MASK) {
        case AUTIA_BASE: { *rd_out = rd; *rn_out = rn; return AUTH_IA; }
        case AUTIB_BASE: { *rd_out = rd; *rn_out = rn; return AUTH_IB; }
        case AUTDA_BASE: { *rd_out = rd; *rn_out = rn; return AUTH_DA; }
        case AUTDB_BASE: { *rd_out = rd; *rn_out = rn; return AUTH_DB; }
    }
    switch (inst & AUTHZ_MASK) {
        case AUTIZA_BASE: { *rd_out = rd; *rn_out = 31; return AUTH_IZA; }
        case AUTIZB_BASE: { *rd_out = rd; *rn_out = 31; return AUTH_IZB; }
        case AUTDZA_BASE: { *rd_out = rd; *rn_out = 31; return AUTH_DZA; }
        case AUTDZB_BASE: { *rd_out = rd; *rn_out = 31; return AUTH_DZB; }
    }
    return AUTH_NONE;
}

/* gpr[0]=x29, gpr[1]=x28, ..., gpr[29]=x0; x30=lr, x31=XZR */
static inline uint64_t read_xreg(const struct cpu_state *s, uint32_t n)
{
    if (n == 31) { return 0; }
    if (n == 30) { return (uint64_t)s->lr; }
    return (uint64_t)s->gpr[29 - n];
}

static inline void write_xreg(struct cpu_state *s, uint32_t n, uint64_t v)
{
    if (n == 31) { return; }
    if (n == 30) { s->lr = (uintptr_t)v; return; }
    s->gpr[29 - n] = (uintptr_t)v;
}

static const char *auth_mnemonic(auth_type_t atype)
{
    switch (atype) {
        case AUTH_IA:  return "autia";
        case AUTH_IB:  return "autib";
        case AUTH_DA:  return "autda";
        case AUTH_DB:  return "autdb";
        case AUTH_IZA: return "autiza";
        case AUTH_IZB: return "autizb";
        case AUTH_DZA: return "autdza";
        case AUTH_DZB: return "autdzb";
        default:       return NULL;
    }
}

/* Per-input PAC keys (from the input's PAC_KEYS section) the software model signs under. */
static struct pac_keys g_pac_keys;
static bool g_pac_keys_present = false;

void pac_keys_init(const uint64_t* keys, bool present)
{
    g_pac_keys_present = present;
    if (present) {
        memcpy(&g_pac_keys, keys, sizeof(g_pac_keys));
    }
}

void pac_sign_plugin_init(void) { pac_profile_set(2, 16, 1, true); }   /* QARMA3, VA 48 */
void pac_sign_plugin_cleanup(void) {}

/* The runner's PAC profile, established by pac_profile_set (pac_sign_plugin_init sets the default).
 *   QARMA3: iterations = 2, VA 48 -> tsz = 16.
 *   QARMA5: iterations = 4, VA 39 -> tsz = 25.
 * tbi = top-byte-ignore; pauth2 = FEAT_PAuth2 (APA/APA3 >= 3). */
static struct pac_profile g_pac_profile;

void pac_profile_set(int iterations, int tsz, int tbi, bool pauth2)
{
    g_pac_profile.iterations = iterations;
    g_pac_profile.tsz = tsz;
    g_pac_profile.tbi = tbi;
    g_pac_profile.pauth2 = pauth2;
}

static void key_for(pac_type_t ptype, uint64_t* lo, uint64_t* hi)
{
    switch (ptype) {
        case PAC_IA: case PAC_IZA: *lo = g_pac_keys.apia_lo; *hi = g_pac_keys.apia_hi; break;
        case PAC_IB: case PAC_IZB: *lo = g_pac_keys.apib_lo; *hi = g_pac_keys.apib_hi; break;
        case PAC_DA: case PAC_DZA: *lo = g_pac_keys.apda_lo; *hi = g_pac_keys.apda_hi; break;
        case PAC_DB: case PAC_DZB: *lo = g_pac_keys.apdb_lo; *hi = g_pac_keys.apdb_hi; break;
        default:                   *lo = 0; *hi = 0; break;
    }
}

static uint64_t sw_pac_xpac(uint64_t ptr)
{
    return qarma_strip(ptr, g_pac_profile);
}

static uint64_t sw_pac_sign(pac_type_t ptype, uint64_t ptr, uint64_t ctx)
{
    uint64_t lo, hi;
    key_for(ptype, &lo, &hi);
    return qarma_addpac(ptr, ctx, lo, hi, g_pac_profile);
}

/* PACGA writes a 32-bit generic MAC into bits [63:32] of the destination. */
static uint64_t sw_pac_ga(uint64_t xn, uint64_t xm)
{
    return qarma_computepac(xn, xm, g_pac_keys.apga_lo, g_pac_keys.apga_hi,
                            g_pac_profile.iterations) & 0xFFFFFFFF00000000ull;
}

/* The PAC sign op corresponding to an auth op (same key/key-letter; parallel enums). */
static pac_type_t auth_to_pac(auth_type_t a)
{
    switch (a) {
        case AUTH_IA:  return PAC_IA;
        case AUTH_IB:  return PAC_IB;
        case AUTH_DA:  return PAC_DA;
        case AUTH_DB:  return PAC_DB;
        case AUTH_IZA: return PAC_IZA;
        case AUTH_IZB: return PAC_IZB;
        case AUTH_DZA: return PAC_DZA;
        case AUTH_DZB: return PAC_DZB;
        default:       return PAC_NONE;
    }
}

/* Model an architectural AUT* without ever executing one: a successful AUT* yields the canonical
 * pointer (XPAC), and re-signing that with the same key+context reproduces the input iff it was
 * correctly signed. A real AUT* is never issued, so a wrong signature cannot FPAC-reset the box.
 * Genuine architectural auths are always correctly signed; a mismatch means a forged/mis-signed
 * AUT* reached the architectural path (a generator/seal bug). That must not be papered over with a
 * canonical result -- abort so the Python side surfaces it and stops the run. */
static uint64_t model_auth(auth_type_t atype, uint64_t ptr, uint64_t ctx, uintptr_t pc)
{
    uint64_t canonical = sw_pac_xpac(ptr);
    uint64_t resigned  = sw_pac_sign(auth_to_pac(atype), canonical, ctx);
    if (resigned != ptr) {
        fprintf(stderr, "[CE FATAL] %s at pc=%#lx would FPAC (ptr=%#018lx ctx=%#018lx): "
                "forged architectural auth -- aborting\n",
                auth_mnemonic(atype), (unsigned long)pc, (unsigned long)ptr, (unsigned long)ctx);
        abort();
    }
    return canonical;
}

/* ---------------------------------------------------------------------------
 * Hooks
 *
 * Each hook fires before the corresponding instruction in the simulation.
 * We call the kernel to execute the PAC/AUT operation at EL1 with the correct
 * keys, write the result back into the simulated register file, then return
 * PC+4 to skip the original instruction.
 *
 * auth_verify_hook splits on speculation depth (is_in_speculation()):
 *   - Architectural path (nesting==0): call kernel AUT — correct architectural
 *     behavior, matches what the hardware TC executes.
 *   - Speculative path (nesting>0): run XPAC instead of AUT.  AUT in
 *     speculation would allow the speculative trace to observe whether a
 *     pointer is correctly signed (success vs. corrupted address), leaking PAC
 *     key information.  XPAC strips the tag unconditionally — NI-correct.
 * --------------------------------------------------------------------------- */

void *pac_sign_hook(struct simulation_state *sim_state)
{
    if (NULL == sim_state) { return NULL; }

    uint32_t inst = *(uint32_t *)sim_state->cpu_state.pc;

    if ((inst & PACGA_MASK) == PACGA_BASE) {
        uint32_t rd = inst & 0x1F;
        uint32_t rn = (inst >> 5)  & 0x1F;
        uint32_t rm = (inst >> 16) & 0x1F;
        uint64_t xn = read_xreg(&sim_state->cpu_state, rn);
        uint64_t xm = read_xreg(&sim_state->cpu_state, rm);
        write_xreg(&sim_state->cpu_state, rd, sw_pac_ga(xn, xm));
        return (void*)(sim_state->cpu_state.pc + 4);
    }

    uint32_t rd, rn;
    pac_type_t ptype = classify_pac(inst, &rd, &rn);
    if (ptype == PAC_NONE) { return NULL; }

    uint64_t ptr = read_xreg(&sim_state->cpu_state, rd);
    uint64_t ctx = read_xreg(&sim_state->cpu_state, rn);

    uint64_t signed_val = sw_pac_sign(ptype, ptr, ctx);
    write_xreg(&sim_state->cpu_state, rd, signed_val);

    return (void*)(sim_state->cpu_state.pc + 4);
}

void *auth_verify_hook(struct simulation_state *sim_state)
{
    if (NULL == sim_state) { return NULL; }

    uint32_t inst = *(uint32_t *)sim_state->cpu_state.pc;
    uint32_t rd, rn;
    auth_type_t atype = classify_auth(inst, &rd, &rn);
    if (atype == AUTH_NONE) { return NULL; }

    uint64_t ptr = read_xreg(&sim_state->cpu_state, rd);
    uint64_t ctx = read_xreg(&sim_state->cpu_state, rn);

    uint64_t result;
    int spec = is_in_speculation();

    if (spec) {
        result = sw_pac_xpac(ptr);
    } else {
        result = model_auth(atype, ptr, ctx, sim_state->cpu_state.pc);
    }
    write_xreg(&sim_state->cpu_state, rd, result);

    return (void*)(sim_state->cpu_state.pc + 4);
}

void *xpac_hook(struct simulation_state *sim_state)
{
    if (NULL == sim_state) { return NULL; }

    uint32_t inst = *(uint32_t *)sim_state->cpu_state.pc;
    uint32_t masked = inst & XPAC_MASK;
    if (masked != XPACI_BASE && masked != XPACD_BASE) { return NULL; }

    uint32_t rd = inst & 0x1F;
    uint64_t ptr = read_xreg(&sim_state->cpu_state, rd);

    write_xreg(&sim_state->cpu_state, rd, sw_pac_xpac(ptr));

    return (void*)(sim_state->cpu_state.pc + 4);
}
