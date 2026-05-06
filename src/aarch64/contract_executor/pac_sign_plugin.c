#include "pac_sign_plugin.h"
#include "simulation_state.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <stdint.h>

/* ---------------------------------------------------------------------------
 * Shared ioctl definitions — must match chardevice.h in the kernel module.
 * We redefine them here to avoid pulling in kernel headers.
 * --------------------------------------------------------------------------- */

#define REVISOR_IOC_MAGIC       'r'
#define REVISOR_PAC_SIGN_CONSTANT 12
#define REVISOR_PAC_AUTH_CONSTANT 13

struct pac_sign_req {
    uint64_t ptr;
    uint64_t ctx;
    char     mnemonic[16];
    uint64_t result;
};

#define REVISOR_PAC_SIGN  _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_SIGN_CONSTANT, struct pac_sign_req)
#define REVISOR_PAC_AUTH  _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_AUTH_CONSTANT, struct pac_sign_req)

#define EXECUTOR_DEV "/dev/executor"

/* ---------------------------------------------------------------------------
 * PAC instruction encoding detection
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

#define NOP_ENCODING 0xD503201FU

/* ---------------------------------------------------------------------------
 * PAC sign instruction encoding
 * --------------------------------------------------------------------------- */

typedef enum {
    PAC_NONE,
    PAC_IA,  PAC_IB,  PAC_DA,  PAC_DB,
    PAC_IZA, PAC_IZB, PAC_DZA, PAC_DZB
} pac_type_t;

/* ---------------------------------------------------------------------------
 * PAC auth instruction encoding
 * AUTH instructions share the same mask as sign; bit 12 is set relative to
 * the sign variant (e.g. AUTIA_BASE = PACIA_BASE | 0x1000).
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
        case PACIA_BASE: *rd_out = rd; *rn_out = rn; return PAC_IA;
        case PACIB_BASE: *rd_out = rd; *rn_out = rn; return PAC_IB;
        case PACDA_BASE: *rd_out = rd; *rn_out = rn; return PAC_DA;
        case PACDB_BASE: *rd_out = rd; *rn_out = rn; return PAC_DB;
    }
    switch (inst & PACZ_MASK) {
        case PACIZA_BASE: *rd_out = rd; *rn_out = 31; return PAC_IZA;
        case PACIZB_BASE: *rd_out = rd; *rn_out = 31; return PAC_IZB;
        case PACDZA_BASE: *rd_out = rd; *rn_out = 31; return PAC_DZA;
        case PACDZB_BASE: *rd_out = rd; *rn_out = 31; return PAC_DZB;
    }
    return PAC_NONE;
}

static const char *pac_mnemonic(pac_type_t t)
{
    switch (t) {
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

static auth_type_t classify_auth(uint32_t inst, uint32_t *rd_out, uint32_t *rn_out)
{
    uint32_t rd = inst & 0x1F;
    uint32_t rn = (inst >> 5) & 0x1F;

    switch (inst & AUTH2_MASK) {
        case AUTIA_BASE: *rd_out = rd; *rn_out = rn; return AUTH_IA;
        case AUTIB_BASE: *rd_out = rd; *rn_out = rn; return AUTH_IB;
        case AUTDA_BASE: *rd_out = rd; *rn_out = rn; return AUTH_DA;
        case AUTDB_BASE: *rd_out = rd; *rn_out = rn; return AUTH_DB;
    }
    switch (inst & AUTHZ_MASK) {
        case AUTIZA_BASE: *rd_out = rd; *rn_out = 31; return AUTH_IZA;
        case AUTIZB_BASE: *rd_out = rd; *rn_out = 31; return AUTH_IZB;
        case AUTDZA_BASE: *rd_out = rd; *rn_out = 31; return AUTH_DZA;
        case AUTDZB_BASE: *rd_out = rd; *rn_out = 31; return AUTH_DZB;
    }
    return AUTH_NONE;
}

static const char *auth_mnemonic(auth_type_t t)
{
    switch (t) {
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

/* ---------------------------------------------------------------------------
 * Register access helpers
 *
 * cpu_state.gpr[] layout: gpr[0]=x29, gpr[1]=x28, ..., gpr[29]=x0
 * so gpr[29 - N] == xN.  x30 == lr.  x31 == XZR (always 0, not writable).
 * --------------------------------------------------------------------------- */

static inline uint64_t read_xreg(const struct cpu_state *s, uint32_t n)
{
    if (n == 31) return 0;
    if (n == 30) return (uint64_t)s->lr;
    return (uint64_t)s->gpr[29 - n];
}

static inline void write_xreg(struct cpu_state *s, uint32_t n, uint64_t v)
{
    if (n == 31) return;
    if (n == 30) { s->lr = (uintptr_t)v; return; }
    s->gpr[29 - n] = (uintptr_t)v;
}

/* ---------------------------------------------------------------------------
 * Persistent device fd — opened once in pac_sign_plugin_init(), closed in
 * pac_sign_plugin_cleanup().  A single ioctl() is one blocking syscall:
 * the kernel signs the pointer and writes the result before returning,
 * so the caller sees the result the instant ioctl() returns.
 * --------------------------------------------------------------------------- */

static int g_executor_fd = -1;

void pac_sign_plugin_init(void)
{
    g_executor_fd = open(EXECUTOR_DEV, O_RDWR);
    if (g_executor_fd < 0)
        perror("pac_sign_plugin: open " EXECUTOR_DEV);
}

void pac_sign_plugin_cleanup(void)
{
    if (g_executor_fd >= 0) {
        close(g_executor_fd);
        g_executor_fd = -1;
    }
}

static uint64_t kernel_sign(uint64_t ptr, uint64_t ctx, const char *mnemonic)
{
    struct pac_sign_req req;
    req.ptr    = ptr;
    req.ctx    = ctx;
    req.result = ptr; /* safe fallback */
    memset(req.mnemonic, 0, sizeof(req.mnemonic));
    strncpy(req.mnemonic, mnemonic, sizeof(req.mnemonic) - 1);

    if (g_executor_fd < 0) {
        fprintf(stderr, "pac_sign_plugin: device not open\n");
        return ptr;
    }

    if (ioctl(g_executor_fd, REVISOR_PAC_SIGN, &req) < 0) {
        perror("pac_sign_plugin: ioctl REVISOR_PAC_SIGN");
        return ptr;
    }

    return req.result;
}

/* ---------------------------------------------------------------------------
 * The hook
 *
 * Fires before every instruction.  For PAC signing instructions:
 *   1. Read pre-PAC register values (unsigned pointer + context).
 *   2. Ask kernel to sign with its live EL1 keys (one blocking ioctl).
 *   3. Write the kernel-signed value back into Xd.
 *   4. Patch the instruction to NOP so the real PAC doesn't re-sign it.
 *
 * The NOP is written into sim_code.code while it is in restored form.
 * base_hook_c then copies sim_code back into sim_input (the "original" save
 * buffer), so pc_to_orig_instruction() returns NOP when it restores the
 * instruction at this PC before execution.  The patch is ephemeral:
 * sim_input is reloaded fresh from the IPC stream for every request.
 * --------------------------------------------------------------------------- */

void *pac_sign_hook(struct simulation_state *sim_state)
{
    if (!sim_state) return NULL;

    uint32_t inst = *(uint32_t *)sim_state->cpu_state.pc;
    uint32_t rd, rn;
    pac_type_t ptype = classify_pac(inst, &rd, &rn);
    if (ptype == PAC_NONE) return NULL;

    uint64_t ptr = read_xreg(&sim_state->cpu_state, rd);
    uint64_t ctx = read_xreg(&sim_state->cpu_state, rn);

    uint64_t signed_val = kernel_sign(ptr, ctx, pac_mnemonic(ptype));

    write_xreg(&sim_state->cpu_state, rd, signed_val);

    /* Skip to PC+4 — do not NOP the instruction so speculative re-visits still fire the hook */
    return (void*)(sim_state->cpu_state.pc + 4);
}

static uint64_t kernel_auth(uint64_t ptr, uint64_t ctx, const char *mnemonic)
{
    struct pac_sign_req req;
    req.ptr    = ptr;
    req.ctx    = ctx;
    req.result = ptr; /* safe fallback: return pointer as-is on failure */
    memset(req.mnemonic, 0, sizeof(req.mnemonic));
    strncpy(req.mnemonic, mnemonic, sizeof(req.mnemonic) - 1);

    if (g_executor_fd < 0) {
        fprintf(stderr, "pac_sign_plugin: device not open\n");
        return ptr;
    }

    if (ioctl(g_executor_fd, REVISOR_PAC_AUTH, &req) < 0) {
        perror("pac_sign_plugin: ioctl REVISOR_PAC_AUTH");
        return ptr;
    }

    return req.result;
}

void *auth_verify_hook(struct simulation_state *sim_state)
{
    if (!sim_state) return NULL;

    uint32_t inst = *(uint32_t *)sim_state->cpu_state.pc;
    uint32_t rd, rn;
    auth_type_t atype = classify_auth(inst, &rd, &rn);
    if (atype == AUTH_NONE) return NULL;

    uint64_t ptr = read_xreg(&sim_state->cpu_state, rd);
    uint64_t ctx = read_xreg(&sim_state->cpu_state, rn);

    uint64_t clean_val = kernel_auth(ptr, ctx, auth_mnemonic(atype));

    write_xreg(&sim_state->cpu_state, rd, clean_val);

    /* Skip to PC+4 — do not NOP so speculative re-visits still fire the hook */
    return (void*)(sim_state->cpu_state.pc + 4);
}
