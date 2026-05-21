#include "pac_sign_plugin.h"
#include "simulation_state.h"
#include "simulation_execution_clause_hook.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Shared ioctl definitions — must match chardevice.h in the kernel module.
 * We redefine them here to avoid pulling in kernel headers.
 * --------------------------------------------------------------------------- */

#define REVISOR_IOC_MAGIC                   'r'
#define REVISOR_SWAP_PAC_KEYS_CONSTANT       12
#define REVISOR_GET_EXEC_PAC_KEYS_CONSTANT   13

struct pac_keys {
    uint64_t apia_lo, apia_hi;
    uint64_t apib_lo, apib_hi;
    uint64_t apda_lo, apda_hi;
    uint64_t apdb_lo, apdb_hi;
    uint64_t apga_lo, apga_hi;
};

struct pac_keys_swap_req {
    struct pac_keys in_keys;
    struct pac_keys out_keys;
};

struct pac_exec_keys_info {
    struct pac_keys keys;
    uint8_t         use_swap;
};

#define REVISOR_SWAP_PAC_KEYS     _IOWR(REVISOR_IOC_MAGIC, REVISOR_SWAP_PAC_KEYS_CONSTANT, struct pac_keys_swap_req)
#define REVISOR_GET_EXEC_PAC_KEYS _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_EXEC_PAC_KEYS_CONSTANT, struct pac_exec_keys_info)

#define EXECUTOR_DEV "/dev/executor"

/* ---------------------------------------------------------------------------
 * PAC sign instruction encoding detection
 *
 * 2-operand (PACIA/PACIB/PACDA/PACDB):
 *   bits[31:10] fixed per variant, bits[9:5] = Rn, bits[4:0] = Rd
 *
 * Zero-context (PACIZA/PACIZB/PACDZA/PACDZB):
 *   bits[31:5] fixed (Rn=XZR encoded), bits[4:0] = Rd
 * --------------------------------------------------------------------------- */

#define PAC2_MASK    0xFFFFFC00U
#define PACIA_BASE   0xDAC10000U
#define PACIB_BASE   0xDAC10400U
#define PACDA_BASE   0xDAC10800U
#define PACDB_BASE   0xDAC10C00U

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

/* ---------------------------------------------------------------------------
 * PAC auth instruction encoding
 * --------------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------------
 * Register access helpers
 *
 * cpu_state.gpr[] layout: gpr[0]=x29, gpr[1]=x28, ..., gpr[29]=x0
 * so gpr[29 - N] == xN.  x30 == lr.  x31 == XZR (always 0, not writable).
 * --------------------------------------------------------------------------- */

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

/* ---------------------------------------------------------------------------
 * EL0 inline asm PAC/AUTH/XPAC operations
 *
 * PAC sign and XPAC strip operations are always safe — they never fault.
 *
 * IMPORTANT: This CPU implements FEAT_FPAC (faulting PAC auth).  A failed
 * AUT* instruction in EL0 raises a synchronous Pointer Authentication Fault
 * (delivered as SIGILL).  auth_verify_hook splits on speculation depth
 * (is_in_speculation()):
 *   - Architectural path (nesting==0): run AUT — this is the correct
 *     architectural behavior and matches what the hardware executes.
 *   - Speculative path (nesting>0): run XPAC instead of AUT.  Using AUT in
 *     speculation would allow the speculative trace to observe whether a
 *     pointer is correctly signed or not (success vs. corrupted address),
 *     which would leak PAC key information.  XPAC strips the tag
 *     unconditionally, ensuring the speculative path carries no signal about
 *     key correctness — which is the NI-contract-correct behavior.
 * --------------------------------------------------------------------------- */

static inline uint64_t el0_pacia(uint64_t ptr, uint64_t ctx)  { __asm__ volatile("pacia  %0, %1" : "+r"(ptr) : "r"(ctx)); return ptr; }
static inline uint64_t el0_pacib(uint64_t ptr, uint64_t ctx)  { __asm__ volatile("pacib  %0, %1" : "+r"(ptr) : "r"(ctx)); return ptr; }
static inline uint64_t el0_pacda(uint64_t ptr, uint64_t ctx)  { __asm__ volatile("pacda  %0, %1" : "+r"(ptr) : "r"(ctx)); return ptr; }
static inline uint64_t el0_pacdb(uint64_t ptr, uint64_t ctx)  { __asm__ volatile("pacdb  %0, %1" : "+r"(ptr) : "r"(ctx)); return ptr; }
static inline uint64_t el0_paciza(uint64_t ptr)                { __asm__ volatile("paciza %0"     : "+r"(ptr));             return ptr; }
static inline uint64_t el0_pacizb(uint64_t ptr)                { __asm__ volatile("pacizb %0"     : "+r"(ptr));             return ptr; }
static inline uint64_t el0_pacdza(uint64_t ptr)                { __asm__ volatile("pacdza %0"     : "+r"(ptr));             return ptr; }
static inline uint64_t el0_pacdzb(uint64_t ptr)                { __asm__ volatile("pacdzb %0"     : "+r"(ptr));             return ptr; }

static inline uint64_t el0_autia(uint64_t ptr, uint64_t ctx)  { __asm__ volatile("autia  %0, %1" : "+r"(ptr) : "r"(ctx)); return ptr; }
static inline uint64_t el0_autib(uint64_t ptr, uint64_t ctx)  { __asm__ volatile("autib  %0, %1" : "+r"(ptr) : "r"(ctx)); return ptr; }
static inline uint64_t el0_autda(uint64_t ptr, uint64_t ctx)  { __asm__ volatile("autda  %0, %1" : "+r"(ptr) : "r"(ctx)); return ptr; }
static inline uint64_t el0_autdb(uint64_t ptr, uint64_t ctx)  { __asm__ volatile("autdb  %0, %1" : "+r"(ptr) : "r"(ctx)); return ptr; }
static inline uint64_t el0_autiza(uint64_t ptr)                { __asm__ volatile("autiza %0"     : "+r"(ptr));             return ptr; }
static inline uint64_t el0_autizb(uint64_t ptr)                { __asm__ volatile("autizb %0"     : "+r"(ptr));             return ptr; }
static inline uint64_t el0_autdza(uint64_t ptr)                { __asm__ volatile("autdza %0"     : "+r"(ptr));             return ptr; }
static inline uint64_t el0_autdzb(uint64_t ptr)                { __asm__ volatile("autdzb %0"     : "+r"(ptr));             return ptr; }

static inline uint64_t el0_xpaci(uint64_t ptr)  { __asm__ volatile("xpaci %0" : "+r"(ptr)); return ptr; }
static inline uint64_t el0_xpacd(uint64_t ptr)  { __asm__ volatile("xpacd %0" : "+r"(ptr)); return ptr; }

static uint64_t do_pac_el0(pac_type_t ptype, uint64_t ptr, uint64_t ctx)
{
    switch (ptype) {
        case PAC_IA:  { return el0_pacia(ptr, ctx); }
        case PAC_IB:  { return el0_pacib(ptr, ctx); }
        case PAC_DA:  { return el0_pacda(ptr, ctx); }
        case PAC_DB:  { return el0_pacdb(ptr, ctx); }
        case PAC_IZA: { return el0_paciza(ptr); }
        case PAC_IZB: { return el0_pacizb(ptr); }
        case PAC_DZA: { return el0_pacdza(ptr); }
        case PAC_DZB: { return el0_pacdzb(ptr); }
        default: {
            fprintf(stderr, "[CE FATAL] do_pac_el0: unrecognised pac_type_t %d\n", ptype);
            abort();
        }
    }
}

/* Architectural auth: pointer is correctly signed, direct AUT is safe. */
static uint64_t do_auth_el0(auth_type_t atype, uint64_t ptr, uint64_t ctx)
{
    switch (atype) {
        case AUTH_IA:  { return el0_autia(ptr, ctx); }
        case AUTH_IB:  { return el0_autib(ptr, ctx); }
        case AUTH_DA:  { return el0_autda(ptr, ctx); }
        case AUTH_DB:  { return el0_autdb(ptr, ctx); }
        case AUTH_IZA: { return el0_autiza(ptr); }
        case AUTH_IZB: { return el0_autizb(ptr); }
        case AUTH_DZA: { return el0_autdza(ptr); }
        case AUTH_DZB: { return el0_autdzb(ptr); }
        default: {
            fprintf(stderr, "[CE FATAL] do_auth_el0: unrecognised auth_type_t %d\n", atype);
            abort();
        }
    }
}

/* Speculative strip-only: XPAC unconditionally removes the PAC tag without
 * observing whether the pointer was correctly signed.  Key-free and never
 * faults on FEAT_FPAC hardware. */
static uint64_t do_xpac_el0(auth_type_t atype, uint64_t ptr)
{
    switch (atype) {
        case AUTH_IA: case AUTH_IB: case AUTH_IZA: case AUTH_IZB: { return el0_xpaci(ptr); }
        case AUTH_DA: case AUTH_DB: case AUTH_DZA: case AUTH_DZB: { return el0_xpacd(ptr); }
        default: {
            fprintf(stderr, "[CE FATAL] do_xpac_el0: unrecognised auth_type_t %d\n", atype);
            abort();
        }
    }
}

/* ---------------------------------------------------------------------------
 * Executor key management
 *
 * On init we call REVISOR_GET_EXEC_PAC_KEYS once:
 *   use_swap=0 → executor uses the process's default keys; no swap needed.
 *   use_swap=1 → executor was configured with custom keys; we must swap to
 *                those keys around each PAC/AUT operation and restore after.
 * --------------------------------------------------------------------------- */

static int             g_executor_fd = -1;
static struct pac_keys g_exec_keys;
static int             g_use_swap = 0;

static void swap_to_exec_keys(struct pac_keys *saved)
{
    struct pac_keys_swap_req req;
    req.in_keys = g_exec_keys;
    memset(&req.out_keys, 0, sizeof(req.out_keys));
    if (ioctl(g_executor_fd, REVISOR_SWAP_PAC_KEYS, &req) < 0) {
        perror("[CE FATAL] pac_sign_plugin: REVISOR_SWAP_PAC_KEYS failed");
        abort();
    }
    *saved = req.out_keys;
}

static void restore_keys(const struct pac_keys *saved)
{
    struct pac_keys_swap_req req;
    req.in_keys = *saved;
    memset(&req.out_keys, 0, sizeof(req.out_keys));
    if (ioctl(g_executor_fd, REVISOR_SWAP_PAC_KEYS, &req) < 0) {
        perror("[CE FATAL] pac_sign_plugin: REVISOR_SWAP_PAC_KEYS (restore) failed");
        abort();
    }
}

void pac_sign_plugin_init(void)
{
    g_executor_fd = open(EXECUTOR_DEV, O_RDWR);
    if (g_executor_fd < 0) {
        perror("pac_sign_plugin: open " EXECUTOR_DEV);
        return;
    }

    struct pac_exec_keys_info info;
    memset(&info, 0, sizeof(info));
    if (ioctl(g_executor_fd, REVISOR_GET_EXEC_PAC_KEYS, &info) < 0) {
        perror("[CE FATAL] pac_sign_plugin: REVISOR_GET_EXEC_PAC_KEYS failed");
        abort();
    }
    g_exec_keys = info.keys;
    g_use_swap  = info.use_swap;
}

void pac_sign_plugin_cleanup(void)
{
    if (g_executor_fd >= 0) {
        close(g_executor_fd);
        g_executor_fd = -1;
    }
}

/* ---------------------------------------------------------------------------
 * Hooks
 *
 * Each hook fires before the corresponding instruction in the simulation.
 * We execute the PAC/AUT operation in EL0 with the correct keys, write the
 * result back into the simulated register file, then return PC+4 to skip the
 * original instruction so it does not execute again with stale register values.
 * --------------------------------------------------------------------------- */

void *pac_sign_hook(struct simulation_state *sim_state)
{
    if (!sim_state) { return NULL; }

    uint32_t inst = *(uint32_t *)sim_state->cpu_state.pc;
    uint32_t rd, rn;
    pac_type_t ptype = classify_pac(inst, &rd, &rn);
    if (ptype == PAC_NONE) { return NULL; }

    uint64_t ptr = read_xreg(&sim_state->cpu_state, rd);
    uint64_t ctx = read_xreg(&sim_state->cpu_state, rn);

    uint64_t signed_val;
    if (g_use_swap) {
        struct pac_keys saved;
        swap_to_exec_keys(&saved);
        signed_val = do_pac_el0(ptype, ptr, ctx);
        restore_keys(&saved);
    } else {
        signed_val = do_pac_el0(ptype, ptr, ctx);
    }

    write_xreg(&sim_state->cpu_state, rd, signed_val);

    /* Skip to PC+4 — do not NOP the instruction so speculative re-visits still fire the hook */
    return (void*)(sim_state->cpu_state.pc + 4);
}

void *auth_verify_hook(struct simulation_state *sim_state)
{
    if (!sim_state) { return NULL; }

    uint32_t inst = *(uint32_t *)sim_state->cpu_state.pc;
    uint32_t rd, rn;
    auth_type_t atype = classify_auth(inst, &rd, &rn);
    if (atype == AUTH_NONE) { return NULL; }

    uint64_t ptr = read_xreg(&sim_state->cpu_state, rd);
    uint64_t ctx = read_xreg(&sim_state->cpu_state, rn);

    uint64_t result;
    if (is_in_speculation()) {
        result = do_xpac_el0(atype, ptr);
    } else {
        if (g_use_swap) {
            struct pac_keys saved;
            swap_to_exec_keys(&saved);
            result = do_auth_el0(atype, ptr, ctx);
            restore_keys(&saved);
        } else {
            result = do_auth_el0(atype, ptr, ctx);
        }
    }

    write_xreg(&sim_state->cpu_state, rd, result);

    /* Skip to PC+4 — do not NOP so speculative re-visits still fire the hook */
    return (void*)(sim_state->cpu_state.pc + 4);
}
