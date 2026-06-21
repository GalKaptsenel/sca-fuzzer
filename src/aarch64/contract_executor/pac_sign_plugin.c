#include "pac_sign_plugin.h"
#include "simulation_state.h"
#include "simulation_execution_clause_hook.h"

#include "userapi/executor_pac_api.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define EXECUTOR_DEV "/dev/executor"

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

static const char *pac_mnemonic(pac_type_t ptype)
{
    switch (ptype) {
        case PAC_IA:  return "pacia";
        case PAC_IB:  return "pacib";
        case PAC_DA:  return "pacda";
        case PAC_DB:  return "pacdb";
        case PAC_IZA: return "paciza";
        case PAC_IZB: return "pacizb";
        case PAC_DZA: return "pacdza";
        case PAC_DZB: return "pacdzb";
        default:      return NULL;
    }
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

static int g_executor_fd = -1;

void pac_sign_plugin_init(void)
{
    g_executor_fd = open(EXECUTOR_DEV, O_RDWR);
    if (g_executor_fd < 0) {
        perror("[CE FATAL] pac_sign_plugin: open " EXECUTOR_DEV);
        abort();
    }

}

void pac_sign_plugin_cleanup(void)
{
    if (g_executor_fd >= 0) {
        close(g_executor_fd);
        g_executor_fd = -1;
    }
}

static uint64_t kernel_pac_xpac(auth_type_t atype, uint64_t ptr)
{
    const char *m;
    switch (atype) {
        case AUTH_IA: case AUTH_IB: case AUTH_IZA: case AUTH_IZB: m = "xpaci"; break;
        case AUTH_DA: case AUTH_DB: case AUTH_DZA: case AUTH_DZB: m = "xpacd"; break;
        default:
            fprintf(stderr, "[CE FATAL] kernel_pac_xpac: unrecognised auth_type_t %d\n", atype);
            abort();
    }
    struct pac_sign_req req = { .ptr = ptr, .ctx = 0, .result = 0 };
    strncpy(req.mnemonic, m, sizeof(req.mnemonic) - 1);
    if (ioctl(g_executor_fd, REVISOR_PAC_XPAC, &req) < 0) {
        perror("[CE FATAL] pac_sign_plugin: REVISOR_PAC_XPAC failed");
        abort();
    }
    return req.result;
}

static uint64_t kernel_pac_sign(pac_type_t ptype, uint64_t ptr, uint64_t ctx)
{
    const char *m = pac_mnemonic(ptype);
    if (!m) {
        fprintf(stderr, "[CE FATAL] kernel_pac_sign: unrecognised pac_type_t %d\n", ptype);
        abort();
    }
    struct pac_sign_req req = { .ptr = ptr, .ctx = ctx, .result = 0 };
    strncpy(req.mnemonic, m, sizeof(req.mnemonic) - 1);
    if (ioctl(g_executor_fd, REVISOR_PAC_SIGN, &req) < 0) {
        perror("[CE FATAL] pac_sign_plugin: REVISOR_PAC_SIGN failed");
        abort();
    }
    return req.result;
}

static uint64_t kernel_pac_auth(auth_type_t atype, uint64_t ptr, uint64_t ctx)
{
    const char *m = auth_mnemonic(atype);
    if (!m) {
        fprintf(stderr, "[CE FATAL] kernel_pac_auth: unrecognised auth_type_t %d\n", atype);
        abort();
    }
    struct pac_sign_req req = { .ptr = ptr, .ctx = ctx, .result = 0 };
    strncpy(req.mnemonic, m, sizeof(req.mnemonic) - 1);
    if (ioctl(g_executor_fd, REVISOR_PAC_AUTH, &req) < 0) {
        perror("[CE FATAL] pac_sign_plugin: REVISOR_PAC_AUTH failed");
        abort();
    }
    return req.result;
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
        struct pac_sign_req req = { .ptr = xn, .ctx = xm, .result = 0 };
        strncpy(req.mnemonic, "pacga", sizeof(req.mnemonic) - 1);
        if (ioctl(g_executor_fd, REVISOR_PAC_SIGN, &req) < 0) {
            perror("[CE FATAL] pac_sign_hook: REVISOR_PAC_SIGN(pacga) failed");
            abort();
        }
        write_xreg(&sim_state->cpu_state, rd, req.result);
        return (void*)(sim_state->cpu_state.pc + 4);
    }

    uint32_t rd, rn;
    pac_type_t ptype = classify_pac(inst, &rd, &rn);
    if (ptype == PAC_NONE) { return NULL; }

    uint64_t ptr = read_xreg(&sim_state->cpu_state, rd);
    uint64_t ctx = read_xreg(&sim_state->cpu_state, rn);

    uint64_t signed_val = kernel_pac_sign(ptype, ptr, ctx);
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
        result = kernel_pac_xpac(atype, ptr);
    } else {
        result = kernel_pac_auth(atype, ptr, ctx);
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

    const char *m = (masked == XPACI_BASE) ? "xpaci" : "xpacd";
    struct pac_sign_req req = { .ptr = ptr, .ctx = 0, .result = 0 };
    strncpy(req.mnemonic, m, sizeof(req.mnemonic) - 1);
    if (ioctl(g_executor_fd, REVISOR_PAC_XPAC, &req) < 0) {
        perror("[CE FATAL] xpac_hook: REVISOR_PAC_XPAC failed");
        abort();
    }
    write_xreg(&sim_state->cpu_state, rd, req.result);

    return (void*)(sim_state->cpu_state.pc + 4);
}
