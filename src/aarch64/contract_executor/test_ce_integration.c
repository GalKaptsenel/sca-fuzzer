/*
 * test_ce_integration.c — Black-box integration tests for ./contract_executor
 *
 * Approach:
 *   fork() + pipe() + exec() ./contract_executor, write a crafted
 *   simulation_input initialization to its stdin, read the contract_trace_t output
 *   from its stdout, and verify trace entries match expected observations.
 *
 * Main.c wires X29 = requested_mem_base_virt before entering simulation,
 * so code that uses X29 as a base register is the simplest way to write
 * self-contained test programs.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <libgen.h>
#include <limits.h>

#include "simulation_input.h"
#include "simulation_output.h"
#include "instruction_encodings.h"
#include "stream_ipc.h"
#include "userapi/executor_input_format.h"

/* ---- test infrastructure ------------------------------------------------ */

static int g_tests_run    = 0;
static int g_tests_failed = 0;

#define EXPECT(cond) do { \
    ++g_tests_run; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d  %s\n", __func__, __LINE__, #cond); \
        ++g_tests_failed; \
    } \
} while (0)

#define EXPECT_EQ(a, b) do { \
    ++g_tests_run; \
    unsigned long long _a = (unsigned long long)(uintptr_t)(a); \
    unsigned long long _b = (unsigned long long)(uintptr_t)(b); \
    if (_a != _b) { \
        fprintf(stderr, "FAIL %s:%d  %s == %s  got 0x%llx vs 0x%llx\n", \
            __func__, __LINE__, #a, #b, _a, _b); \
        ++g_tests_failed; \
    } \
} while (0)

/* ---- constants ---------------------------------------------------------- */

/* Absolute path to contract_executor next to THIS test binary, so the test runs from any cwd. */
static const char* ce_binary_path(void) {
    static char path[PATH_MAX];
    if (path[0]) return path;                       // cached
    char exe[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) { return "./contract_executor"; }   // fallback
    exe[n] = '\0';
    snprintf(path, sizeof(path), "%s/contract_executor", dirname(exe));
    return path;
}
#define KBASE      UINT64_C(0xFFFF000080000000)
#define MEM_SIZE   4096u
#define REGS_COUNT 8u   /* the 8 GPR input slots: X0..X5, NZCV(slot6), SP(slot7) */
#define MAX_ENTRIES 256u

/* ---- instruction helpers ------------------------------------------------ */

/* LDR X<rt>, [X<rn>] — unsigned offset 0, 64-bit */
static uint32_t enc_ldr_reg(int rt, int rn) {
    return 0xF9400000u | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* STR X<rt>, [X<rn>] — unsigned offset 0, 64-bit */
static uint32_t enc_str_reg(int rt, int rn) {
    return 0xF9000000u | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* STR X<rt>, [X<rn>], #<imm9> — post-index, 64-bit (writes [Xn], then Xn += imm9) */
static uint32_t enc_str_postidx(int rt, int rn, int imm9) {
    uint32_t i = (uint32_t)(int32_t)imm9 & 0x1FF;
    return 0xF8000400u | (i << 12) | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* STP X<rt>, X<rt2>, [X<rn>] — signed offset 0, 64-bit (writes two 8-byte elements) */
static uint32_t enc_stp(int rt, int rt2, int rn) {
    return 0xA9000000u | ((uint32_t)rt2 << 10) | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* LDR X<rt>, [X<rn>, #<off>] — unsigned offset, 64-bit; off must be mult of 8 */
static uint32_t enc_ldr_unsigned(int rt, int rn, int off) {
    uint32_t pimm12 = (uint32_t)(off / 8);
    return 0xF9400000u | (pimm12 << 10) | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* LDR X<rt>, [X<rn>, #<imm9>]! — pre-index, 64-bit; -256..255 */
static uint32_t enc_ldr_preidx(int rt, int rn, int imm9) {
    uint32_t i = (uint32_t)(int32_t)imm9 & 0x1FF;
    return 0xF8400C00u | (i << 12) | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* LDR X<rt>, [X<rn>], #<imm9> — post-index, 64-bit */
static uint32_t enc_ldr_postidx(int rt, int rn, int imm9) {
    uint32_t i = (uint32_t)(int32_t)imm9 & 0x1FF;
    return 0xF8400400u | (i << 12) | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* LDP X<rt>, X<rt2>, [X<rn>] — signed offset 0, 64-bit */
static uint32_t enc_ldp_signed(int rt, int rt2, int rn) {
    return 0xA9400000u | ((uint32_t)rt2 << 10) | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* NOP */
static uint32_t enc_nop(void)   { return 0xD503201Fu; }
static uint32_t enc_ssbb(void)   { return 0xD503309Fu; }  /* store-bypass barrier (DSB #0)          */
static uint32_t enc_pssbb(void)  { return 0xD503349Fu; }  /* physical store-bypass barrier (DSB #4)  */
static uint32_t enc_dmb_sy(void) { return 0xD5033FBFu; }  /* ordering only — fences no speculation  */
static uint32_t enc_sb(void)     { return 0xD50330FFu; }  /* full speculation barrier               */
static uint32_t enc_isb(void)    { return 0xD5033FDFu; }  /* context sync — fences control          */
static uint32_t enc_dsb_sy(void) { return 0xD5033F9Fu; }  /* full-system DSB — fences control       */
static uint32_t enc_dsb_ish(void){ return 0xD5033B9Fu; }  /* narrower DSB — store-bypass only        */
/* LDRB W<rt>, [X<rn>] — unsigned offset 0, 8-bit */
static uint32_t enc_ldrb(int rt, int rn) {
    return 0x39400000u | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* LDRH W<rt>, [X<rn>] — unsigned offset 0, 16-bit */
static uint32_t enc_ldrh(int rt, int rn) {
    return 0x79400000u | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* LDR W<rt>, [X<rn>] — unsigned offset 0, 32-bit */
static uint32_t enc_ldr32(int rt, int rn) {
    return 0xB9400000u | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* STRB W<rt>, [X<rn>] — unsigned offset 0, 8-bit */
static uint32_t enc_strb(int rt, int rn) {
    return 0x39000000u | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* LDR X<rt>, [X<rn>, X<rm>] — register offset, 64-bit, LSL#0 */
static uint32_t enc_ldr_regoff(int rt, int rn, int rm) {
    return 0xF8606800u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rt;
}
/* B.<cond> <label> — label is signed byte offset from this instruction (multiple of 4) */
static uint32_t enc_b_cond(uint32_t cond, int offset) {
    uint32_t imm19 = (uint32_t)(offset / 4) & 0x7FFFFu;
    return 0x54000000u | (imm19 << 5) | cond;
}
/* CBZ X<rt>, <label> — 64-bit */
static uint32_t enc_cbz(int rt, int offset) {
    uint32_t imm19 = (uint32_t)(offset / 4) & 0x7FFFFu;
    return 0xB4000000u | (imm19 << 5) | (uint32_t)rt;
}
/* CBNZ X<rt>, <label> — 64-bit */
static uint32_t enc_cbnz(int rt, int offset) {
    uint32_t imm19 = (uint32_t)(offset / 4) & 0x7FFFFu;
    return 0xB5000000u | (imm19 << 5) | (uint32_t)rt;
}
/* TBZ X<rt>, #<bit>, <label> */
static uint32_t enc_tbz(int rt, int bit, int offset) {
    uint32_t imm14 = (uint32_t)(offset / 4) & 0x3FFFu;
    uint32_t b5  = (uint32_t)(bit >> 5) & 1u;
    uint32_t b40 = (uint32_t)(bit & 0x1F);
    return 0x36000000u | (b5 << 31) | (b40 << 19) | (imm14 << 5) | (uint32_t)rt;
}
/* TBNZ X<rt>, #<bit>, <label> */
static uint32_t enc_tbnz(int rt, int bit, int offset) {
    uint32_t imm14 = (uint32_t)(offset / 4) & 0x3FFFu;
    uint32_t b5  = (uint32_t)(bit >> 5) & 1u;
    uint32_t b40 = (uint32_t)(bit & 0x1F);
    return 0x37000000u | (b5 << 31) | (b40 << 19) | (imm14 << 5) | (uint32_t)rt;
}

/* ---- IPC frame builder -------------------------------------------------- */

/*
 * Build the message written to CE's stdin:
 *   [struct header (8 B)] [struct input_header (128 B)] [code] [input initialization]
 * The input initialization (executor_input_format) carries memory as MAIN(=mem) + an empty FAULTY,
 * and the GPR section (regs). regs_override: 8-element array for X0..X5, NZCV(slot6), SP(slot7);
 * NULL = all zero.
 */
static size_t build_ce_input(
        uint8_t *buf, size_t bufsz,
        const uint32_t *code, size_t n_words,
        const uint8_t  *mem,  size_t mem_sz,
        uint64_t kbase_addr,
        uint64_t max_nesting,
        uint64_t max_instr,
        uint64_t contract,
        uint64_t branch_predictor,
        const uint64_t *regs_override,
        const uint8_t  *mte_tags, size_t mte_len)
{
    uint64_t regs[REGS_COUNT];
    if (regs_override) {
        memcpy(regs, regs_override, sizeof(regs));
    } else {
        memset(regs, 0, sizeof(regs));
    }

    size_t code_sz  = n_words * 4;
    size_t regs_sz  = sizeof(regs);

    /* input initialization: main = mem, empty faulty, gpr = regs, and (optionally) MTE tags. */
    const uint64_t n_sec = (NULL != mte_tags) ? 4 : 3;
    size_t init_hdr = sizeof(struct revisor_input_header) + n_sec * sizeof(struct revisor_input_section);
    size_t init_len = init_hdr + mem_sz + regs_sz + ((NULL != mte_tags) ? mte_len : 0);
    size_t payload  = sizeof(struct input_header) + code_sz + init_len;

    if (bufsz < sizeof(struct header) + payload) return 0;

    uint8_t *p = buf;

    /* IPC header */
    struct header ipc = { .length = (uint32_t)payload, .type = 0 };
    memcpy(p, &ipc, sizeof(ipc)); p += sizeof(ipc);

    /* CE envelope */
    struct input_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic    = RVZRCE_MAGIC;
    hdr.version  = RVZRCE_VERSION;
    hdr.arch     = RVZR_ARCH_AARCH64;
    hdr.flags    = RVZR_FLAG_HAS_CODE | RVZR_FLAG_HAS_INPUT;
    hdr.config.flags                    = CONFIG_FLAG_REQ_MEM_BASE_VIRT;
    hdr.config.max_misspred_branch_nesting = max_nesting;
    hdr.config.max_misspred_instructions   = max_instr;
    hdr.config.requested_mem_base_virt  = kbase_addr;
    hdr.config.execution_clauses            = contract;
    hdr.config.branch_predictor             = branch_predictor;
    hdr.code_size       = (uint64_t)code_sz;
    hdr.input_init_size = (uint64_t)init_len;
    memcpy(p, &hdr, sizeof(hdr)); p += sizeof(hdr);

    /* code */
    memcpy(p, code, code_sz); p += code_sz;

    /* input initialization */
    uint8_t *init = p;
    struct revisor_input_header *ih = (struct revisor_input_header *)init;
    struct revisor_input_section *tab = (struct revisor_input_section *)(init + sizeof(*ih));
    size_t off = init_hdr;
    uint64_t i = 0;
    tab[i].type = REVISOR_SEC_MEMORY_MAIN;   tab[i].flags = 0; tab[i].offset = off; tab[i].length = mem_sz;
    memcpy(init + off, mem, mem_sz); off += mem_sz; i++;
    tab[i].type = REVISOR_SEC_MEMORY_FAULTY; tab[i].flags = 0; tab[i].offset = off; tab[i].length = 0; i++;
    tab[i].type = REVISOR_SEC_GPR;           tab[i].flags = 0; tab[i].offset = off; tab[i].length = regs_sz;
    memcpy(init + off, regs, regs_sz); off += regs_sz; i++;
    if (NULL != mte_tags) {
        tab[i].type = REVISOR_SEC_MTE_TAGS;  tab[i].flags = 0; tab[i].offset = off; tab[i].length = mte_len;
        memcpy(init + off, mte_tags, mte_len); off += mte_len; i++;
    }
    ih->magic = REVISOR_INPUT_MAGIC; ih->version = REVISOR_INPUT_VERSION;
    ih->header_len = init_hdr; ih->n_sections = n_sec; ih->flags = 0; ih->total_len = init_len;
    p += init_len;

    return (size_t)(p - buf);
}

/* ---- output parser ------------------------------------------------------ */

typedef struct {
    int              n_entries;
    uint64_t         truncated;
    instr_trace_entry_t entries[MAX_ENTRIES];
} ce_result_t;

static bool read_all(int fd, void *dst, size_t n) {
    uint8_t *p = dst;
    while (n > 0) {
        ssize_t r = read(fd, p, n);
        if (r <= 0) return false;
        p += r; n -= (size_t)r;
    }
    return true;
}

/* Read and parse the contract_trace_t output produced on CE's stdout. */
static bool parse_ce_output(int fd, ce_result_t *out) {
    out->n_entries = 0;

    struct header ipc;
    if (!read_all(fd, &ipc, sizeof(ipc))) return false;
    if (ipc.length < sizeof(contract_trace_t))  return false;

    contract_trace_t ct;
    if (!read_all(fd, &ct, sizeof(ct))) return false;
    out->truncated = ct.truncated;

    size_t n = ct.entry_count;
    if (n > MAX_ENTRIES) n = MAX_ENTRIES;

    for (size_t i = 0; i < n; ++i) {
        if (!read_all(fd, &out->entries[i], sizeof(instr_trace_entry_t)))
            return false;
    }
    out->n_entries = (int)n;
    return true;
}

/* ---- CE runner ---------------------------------------------------------- */

/*
 * Fork + exec CE, pipe input/output, return parsed trace.
 * regs_override: optional 8-element array (X0..X5, NZCV(slot6), SP(slot7)); NULL = zeros.
 */
static bool run_ce_full(
        const uint32_t *code, size_t n_words,
        const uint8_t  *mem,  size_t mem_sz,
        uint64_t kbase_addr,
        uint64_t max_nesting,
        uint64_t max_instr,
        uint64_t contract,
        uint64_t branch_predictor,
        const uint64_t *regs_override,
        const uint8_t  *mte_tags, size_t mte_len,
        ce_result_t *result)
{
    static uint8_t in_buf[1 << 22];   /* 4 MB — never stack-allocate */

    size_t in_len = build_ce_input(in_buf, sizeof(in_buf),
                                   code, n_words, mem, mem_sz,
                                   kbase_addr, max_nesting, max_instr, contract,
                                   branch_predictor, regs_override, mte_tags, mte_len);
    if (in_len == 0) return false;

    int to_ce[2], from_ce[2];
    if (pipe(to_ce)   < 0) return false;
    if (pipe(from_ce) < 0) { close(to_ce[0]); close(to_ce[1]); return false; }

    pid_t pid = fork();
    if (pid < 0) {
        close(to_ce[0]); close(to_ce[1]);
        close(from_ce[0]); close(from_ce[1]);
        return false;
    }

    if (pid == 0) {
        /* Child: redirect stdin/stdout to pipes */
        dup2(to_ce[0],   STDIN_FILENO);
        dup2(from_ce[1], STDOUT_FILENO);
        close(to_ce[0]); close(to_ce[1]);
        close(from_ce[0]); close(from_ce[1]);

        /* Redirect stderr to /dev/null so CE log spam doesn't clutter test output */
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) dup2(devnull, STDERR_FILENO);

        execl(ce_binary_path(), ce_binary_path(), NULL);
        _exit(127);
    }

    /* Parent: close child-side ends */
    close(to_ce[0]);
    close(from_ce[1]);

    /* Write input then close write-end so CE sees EOF on stdin after one frame */
    const uint8_t *wp = in_buf;
    size_t wrem = in_len;
    while (wrem > 0) {
        ssize_t w = write(to_ce[1], wp, wrem);
        if (w <= 0) break;
        wp += w; wrem -= (size_t)w;
    }
    close(to_ce[1]);

    bool ok = parse_ce_output(from_ce[0], result);
    close(from_ce[0]);

    int status;
    waitpid(pid, &status, 0);


    return ok;
}

/* Run, auto-selecting the predictor the way the Python mapping does (BPU => Neoverse-N3). */
static bool run_ce(
        const uint32_t *code, size_t n_words,
        const uint8_t  *mem,  size_t mem_sz,
        uint64_t kbase_addr,
        uint64_t max_nesting,
        uint64_t contract,
        const uint64_t *regs_override,
        ce_result_t *result)
{
    uint64_t bp = (contract & EXEC_CLAUSE_BPU) ? BRANCH_PREDICTOR_NEOVERSE_N3
                                               : BRANCH_PREDICTOR_NONE;
    return run_ce_full(code, n_words, mem, mem_sz, kbase_addr, max_nesting, 0, contract, bp,
                       regs_override, NULL, 0, result);
}

/* Convenience: run with all-zero regs. */
static bool run_ce_simple(
        const uint32_t *code, size_t n_words,
        const uint8_t  *mem,  size_t mem_sz,
        uint64_t kbase_addr,
        uint64_t max_nesting,
        uint64_t contract,
        ce_result_t *result)
{
    return run_ce(code, n_words, mem, mem_sz, kbase_addr, max_nesting, contract, NULL, result);
}

/* PAC sign/auth/strip route through /dev/executor (EL1 keys); such tests SKIP without it. */
static bool executor_dev_present(void) {
    return access("/dev/executor", R_OK) == 0;
}

#define SKIP_IF_NO_EXECUTOR() do {                                              \
    if (!executor_dev_present()) {                                              \
        printf("SKIP %s: /dev/executor absent (load revizor-executor.ko)\n",   \
               __func__);                                                       \
        return;                                                                 \
    }                                                                          \
} while (0)

/* Helper: find the Nth memory-access entry (0-indexed). Returns NULL if not found. */
static instr_trace_entry_t *find_mem_entry(ce_result_t *res, int n) {
    int found = 0;
    for (int i = 0; i < res->n_entries; ++i) {
        if (res->entries[i].metadata.has_memory_access) {
            if (found == n) return &res->entries[i];
            ++found;
        }
    }
    return NULL;
}

/* Helper: count memory-access entries. */
static int count_mem_entries(const ce_result_t *res) {
    int c = 0;
    for (int i = 0; i < res->n_entries; ++i) {
        if (res->entries[i].metadata.has_memory_access) ++c;
    }
    return c;
}

/* ---- GROUP 1: Simple LDR X0, [X29] ------------------------------------ */

static void test_integration_ldr_base(void) {
    uint32_t code[] = { enc_ldr_reg(0, 29) };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t expect_val = UINT64_C(0xDEADBEEF12345678);
    memcpy(mem, &expect_val, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(res.truncated, (uint64_t)0);   // a short trace must not be flagged truncated

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.is_write,          (uint64_t)0);
    EXPECT_EQ(e->metadata.memory_access.element_size,       (uint64_t)8);
    EXPECT_EQ(e->metadata.memory_access.effective_address,  KBASE);
    EXPECT_EQ(e->metadata.memory_access.before,             expect_val);
    EXPECT_EQ(e->metadata.memory_access.after,              expect_val);
    EXPECT_EQ(e->metadata.speculation_nesting,              (uint64_t)0);
}

/* ---- GROUP 2: Simple STR X0, [X29] ------------------------------------ */

static void test_integration_str_base(void) {
    uint32_t code[] = { enc_str_reg(0, 29) };  /* STR X0, [X29] — X0=0 */

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t init_val = UINT64_C(0xAAAAAAAAAAAAAAAA);
    memcpy(mem, &init_val, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.is_write,         (uint64_t)1);
    EXPECT_EQ(e->metadata.memory_access.element_size,      (uint64_t)8);
    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(e->metadata.memory_access.before,            init_val);
    EXPECT_EQ(e->metadata.memory_access.after,             (uint64_t)0);
    EXPECT_EQ(e->metadata.speculation_nesting,             (uint64_t)0);
}

/* ---- GROUP 3: LDR X0, [X29, #16] — unsigned offset ------------------- */

static void test_integration_ldr_unsigned_offset(void) {
    uint32_t code[] = { enc_ldr_unsigned(0, 29, 16) };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t expect_val = UINT64_C(0x0123456789ABCDEF);
    memcpy(mem + 16, &expect_val, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.is_write,          (uint64_t)0);
    EXPECT_EQ(e->metadata.memory_access.element_size,       (uint64_t)8);
    EXPECT_EQ(e->metadata.memory_access.effective_address,  KBASE + 16);
    EXPECT_EQ(e->metadata.memory_access.before,             expect_val);
}

/* ---- GROUP 4: LDR X0, [X29], #8 — post-index ------------------------- */

static void test_integration_ldr_postidx(void) {
    uint32_t code[] = {
        enc_ldr_postidx(0, 29, 8),  /* LDR X0, [X29], #8 */
        enc_ldr_reg(1, 29),         /* LDR X1, [X29]     */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t val0 = UINT64_C(0x1111111111111111);
    uint64_t val8 = UINT64_C(0x2222222222222222);
    memcpy(mem,     &val0, 8);
    memcpy(mem + 8, &val8, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 2, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
    instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
    EXPECT(ma0 != NULL); EXPECT(ma1 != NULL);
    if (!ma0 || !ma1) return;

    EXPECT_EQ(ma0->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(ma0->metadata.memory_access.element_size,      (uint64_t)8);
    EXPECT_EQ(ma0->metadata.memory_access.before,            val0);
    EXPECT_EQ(ma1->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma1->metadata.memory_access.before,            val8);
}

/* ---- GROUP 5: LDR X0, [X29, #8]! — pre-index ------------------------- */

static void test_integration_ldr_preidx(void) {
    uint32_t code[] = {
        enc_ldr_preidx(0, 29, 8),   /* LDR X0, [X29, #8]! */
        enc_ldr_reg(1, 29),         /* LDR X1, [X29]      */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t val0 = UINT64_C(0x3333333333333333);
    uint64_t val8 = UINT64_C(0x4444444444444444);
    memcpy(mem,     &val0, 8);
    memcpy(mem + 8, &val8, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 2, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
    instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
    EXPECT(ma0 != NULL); EXPECT(ma1 != NULL);
    if (!ma0 || !ma1) return;

    /* Pre-index: EA is already kbase + 8 */
    EXPECT_EQ(ma0->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma0->metadata.memory_access.before,            val8);
    EXPECT_EQ(ma1->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma1->metadata.memory_access.before,            val8);
}

/* ---- GROUP 6: LDP X0, X1, [X29] — pair load -------------------------- */

static void test_integration_ldp(void) {
    uint32_t code[] = { enc_ldp_signed(0, 1, 29) };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0xAAAABBBBCCCCDDDD);
    uint64_t v8 = UINT64_C(0xEEEEFFFF00001111);
    memcpy(mem,     &v0, 8);
    memcpy(mem + 8, &v8, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.is_write,          (uint64_t)0);
    EXPECT_EQ(e->metadata.memory_access.element_size,       (uint64_t)8);
    EXPECT_EQ(e->metadata.memory_access.effective_address,  KBASE);
    EXPECT_EQ(e->metadata.memory_access.before,             v0);
}

/* ---- GROUP 7: kaddr transparency — observation uses kaddr not uaddr --- */

static void test_integration_kaddr_transparency(void) {
    struct { uint32_t off; uint64_t val; } cases[] = {
        { 0,   UINT64_C(0x0000000000000001) },
        { 8,   UINT64_C(0x0000000000000002) },
        { 128, UINT64_C(0x0000000000000003) },
        { 256, UINT64_C(0x0000000000000004) },
    };

    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        uint32_t off = cases[i].off;
        uint64_t val = cases[i].val;

        uint32_t code[] = { enc_ldr_unsigned(0, 29, (int)off) };

        uint8_t mem[MEM_SIZE];
        memset(mem, 0, sizeof(mem));
        memcpy(mem + off, &val, 8);

        ce_result_t res;
        if (!run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
            ++g_tests_run; ++g_tests_failed;
            fprintf(stderr, "FAIL %s[%zu]: run_ce failed\n", __func__, i);
            continue;
        }

        instr_trace_entry_t *e = find_mem_entry(&res, 0);
        EXPECT(e != NULL);
        if (!e) continue;
        EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE + off);
        EXPECT_EQ(e->metadata.memory_access.before, val);
    }
}

/* ---- GROUP 8: multiple sequential loads — trace ordering -------------- */

static void test_integration_sequential_loads(void) {
    uint32_t code[] = {
        enc_ldr_unsigned(0, 29,  0),
        enc_ldr_unsigned(1, 29,  8),
        enc_ldr_unsigned(2, 29, 16),
        enc_ldr_unsigned(3, 29, 24),
    };

    uint64_t expected[4] = {
        UINT64_C(0xAAAAAAAA00000000),
        UINT64_C(0xBBBBBBBB11111111),
        UINT64_C(0xCCCCCCCC22222222),
        UINT64_C(0xDDDDDDDD33333333),
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    for (int i = 0; i < 4; ++i) memcpy(mem + i * 8, &expected[i], 8);

    ce_result_t res;
    if (!run_ce_simple(code, 4, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 4);

    for (int i = 0; i < 4; ++i) {
        instr_trace_entry_t *e = find_mem_entry(&res, i);
        EXPECT(e != NULL);
        if (!e) continue;
        EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE + (uint64_t)i * 8);
        EXPECT_EQ(e->metadata.memory_access.before,            expected[i]);
        EXPECT_EQ(e->metadata.speculation_nesting,             (uint64_t)0);
        if (i > 0) {
            instr_trace_entry_t *prev = find_mem_entry(&res, i - 1);
            EXPECT(prev && e->metadata.instr_index > prev->metadata.instr_index);
        }
    }
}

/* ---- GROUP 9: store then verify — write-back is visible to next load -- */

static void test_integration_store_then_load(void) {
    uint32_t code[] = {
        enc_str_reg(0, 29),         /* STR X0, [X29]  (X0=0) */
        enc_ldr_unsigned(1, 29, 0), /* LDR X1, [X29]  */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t init_val = UINT64_C(0xBEEFBEEFBEEFBEEF);
    memcpy(mem, &init_val, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 2, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
    instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
    EXPECT(ma0 != NULL); EXPECT(ma1 != NULL);
    if (!ma0 || !ma1) return;

    EXPECT_EQ(ma0->metadata.memory_access.is_write,  (uint64_t)1);
    EXPECT_EQ(ma0->metadata.memory_access.before,     init_val);
    EXPECT_EQ(ma0->metadata.memory_access.after,      (uint64_t)0);
    EXPECT_EQ(ma1->metadata.memory_access.is_write,  (uint64_t)0);
    EXPECT_EQ(ma1->metadata.memory_access.before,     (uint64_t)0);
}

/* ---- GROUP 10: NOP has has_memory_access=0 ----------------------------- */

static void test_integration_nop_no_mem(void) {
    /*
     * Code: NOP; LDR X0,[X29]
     * Two trace entries total: NOP entry (has_memory_access=0) and LDR entry (=1).
     */
    uint32_t code[] = { enc_nop(), enc_ldr_reg(0, 29) };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t val = UINT64_C(0xCAFEBABECAFEBABE);
    memcpy(mem, &val, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 2, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    /* Expect at least 2 trace entries total (NOP + LDR) */
    EXPECT(res.n_entries >= 2);

    /* Find NOP entry: first entry with has_memory_access==0 */
    int nop_idx = -1, ldr_idx = -1;
    for (int i = 0; i < res.n_entries; ++i) {
        if (!res.entries[i].metadata.has_memory_access && nop_idx < 0) nop_idx = i;
        if ( res.entries[i].metadata.has_memory_access && ldr_idx < 0) ldr_idx = i;
    }

    EXPECT(nop_idx >= 0);
    EXPECT(ldr_idx >= 0);
    if (nop_idx < 0 || ldr_idx < 0) return;

    /* NOP must come before LDR in trace order */
    EXPECT(nop_idx < ldr_idx);

    /* LDR entry correctness */
    instr_trace_entry_t *e = &res.entries[ldr_idx];
    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(e->metadata.memory_access.before,            val);
    EXPECT_EQ(e->metadata.memory_access.element_size,      (uint64_t)8);
}

/* ---- GROUP 11: access size — LDRB → element_size=1 ------------------- */

static void test_integration_access_size_byte(void) {
    /* before/after record the accessed-size value (element_size bytes), not the surrounding word. */
    uint32_t code[] = { enc_ldrb(0, 29) };  /* LDRB W0,[X29] */

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t word = UINT64_C(0x0102030405060708);
    memcpy(mem, &word, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.element_size,      (uint64_t)1);
    EXPECT_EQ(e->metadata.memory_access.is_write,          (uint64_t)0);
    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(e->metadata.memory_access.before,            word & 0xFF);
}

/* ---- GROUP 12: access size — LDRH → element_size=2 ------------------- */

static void test_integration_access_size_halfword(void) {
    uint32_t code[] = { enc_ldrh(0, 29) };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t word = UINT64_C(0xAABBCCDDEEFF0011);
    memcpy(mem, &word, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.element_size,      (uint64_t)2);
    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(e->metadata.memory_access.before,            word & 0xFFFF);
}

/* ---- GROUP 13: access size — LDR W → element_size=4 ------------------ */

static void test_integration_access_size_word(void) {
    uint32_t code[] = { enc_ldr32(0, 29) };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t word = UINT64_C(0x1122334455667788);
    memcpy(mem, &word, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.element_size,      (uint64_t)4);
    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(e->metadata.memory_access.before,            word & 0xFFFFFFFF);
}

/* ---- GROUP 14: STRB partial write — after field shows byte merge ------ */

static void test_integration_strb_partial_write(void) {
    /*
     * STRB W0,[X29] with X0=0x42 (set via regs input_init slot 0 = X0).
     * mem[0..7] = 0xAABBCCDDEEFF0011; access-size = 1 byte, so before=0x11, after=0x42.
     */
    uint32_t code[] = { enc_strb(0, 29) };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t init_word = UINT64_C(0xAABBCCDDEEFF0011);
    memcpy(mem, &init_word, 8);

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = 0x42;  /* X0 = 0x42 */

    ce_result_t res;
    if (!run_ce(code, 1, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.element_size,      (uint64_t)1);
    EXPECT_EQ(e->metadata.memory_access.is_write,          (uint64_t)1);
    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(e->metadata.memory_access.before,            init_word & 0xFF);
    EXPECT_EQ(e->metadata.memory_access.after,             (uint64_t)0x42);
}

/* ---- GROUP 15: non-X29 base register from regs input_init ------------------- */

static void test_integration_non_x29_base(void) {
    /*
     * Set X1 = kbase via regs input_init (slot 1 = X1).
     * Code: LDR X0,[X1]
     * EA must be kbase, even though we didn't use X29.
     */
    uint32_t code[] = { enc_ldr_reg(0, 1) };  /* LDR X0,[X1] */

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t val = UINT64_C(0xFEEDFACEFEEDFACE);
    memcpy(mem, &val, 8);

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = KBASE;  /* X1 = kbase */

    ce_result_t res;
    if (!run_ce(code, 1, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(e->metadata.memory_access.element_size,      (uint64_t)8);
    EXPECT_EQ(e->metadata.memory_access.before,            val);
    EXPECT_EQ(e->metadata.memory_access.is_write,          (uint64_t)0);
}

/* ---- GROUP 16: NZCV propagated into trace cpu_state ------------------- */

static void test_integration_nzcv_in_cpu_state(void) {
    /*
     * Set initial NZCV = Z=1 (PSTATE: 0x40000000) via regs input_init slot 6.
     * Code: LDR X0,[X29]
     * The trace entry's cpu_state.nzcv should have Z=1.
     */
    uint32_t code[] = { enc_ldr_reg(0, 29) };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));

    /* PSTATE format: N[31] Z[30] C[29] V[28]. Z=1 → 0x40000000 */
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[6] = UINT64_C(0x40000000);  /* slot 6 = NZCV, Z=1 */

    ce_result_t res;
    if (!run_ce(code, 1, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    /* Z bit must be set */
    EXPECT((e->cpu.nzcv >> 30) & 1);
    /* N, C, V must be clear */
    EXPECT(!((e->cpu.nzcv >> 31) & 1));
    EXPECT(!((e->cpu.nzcv >> 29) & 1));
    EXPECT(!((e->cpu.nzcv >> 28) & 1));
}

/* ---- GROUP 17: register-offset addressing — EA = kbase + X1 ----------- */

static void test_integration_register_offset(void) {
    /*
     * Set X1 = 16 via regs input_init. Code: LDR X0,[X29,X1]
     * EA = kbase + 16.
     */
    uint32_t code[] = { enc_ldr_regoff(0, 29, 1) };  /* LDR X0,[X29,X1] */

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t val = UINT64_C(0x123456789ABCDEF0);
    memcpy(mem + 16, &val, 8);

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = 16;  /* X1 = 16 (byte offset) */

    ce_result_t res;
    if (!run_ce(code, 1, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE + 16);
    EXPECT_EQ(e->metadata.memory_access.element_size,      (uint64_t)8);
    EXPECT_EQ(e->metadata.memory_access.before,            val);
}

/* ---- GROUP 17b: register-offset offset-cancel — non-canonical base is preserved (9d8cfdd) --- */

static void test_integration_regoff_noncanonical_base_preserved(void) {
    /*
     * The seal's SUB Rn,Rn,Rm leaves X3 a non-canonical small value (3) while EA = X3+X4 stays in
     * the sandbox. apply_fixups' delta restore must leave X3 == 3; the old kbase-forcing round-trip
     * gave 0xFF00000000000003, which later corrupted an AUT* context.
     */
    uint32_t code[] = {
        enc_ldr_regoff(0, 3, 4),   /* LDR X0,[X3,X4]; X3=3, EA = KBASE+16 */
        enc_ldr_reg(1, 29),        /* second entry so X3's restore is observable */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t val = UINT64_C(0x123456789ABCDEF0);
    memcpy(mem + 16, &val, 8);

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[3] = 3;                 /* X3 = non-canonical base */
    regs[4] = KBASE + 13;        /* EA = X3 + X4 = KBASE + 16 */

    ce_result_t res;
    if (!run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e0 = find_mem_entry(&res, 0);
    EXPECT(e0 != NULL);
    if (!e0) return;
    EXPECT_EQ(e0->metadata.memory_access.effective_address, KBASE + 16);
    EXPECT_EQ(e0->metadata.memory_access.before,            val);

    instr_trace_entry_t *e1 = find_mem_entry(&res, 1);
    EXPECT(e1 != NULL);
    if (!e1) return;
    EXPECT_EQ(e1->cpu.gpr[3], (uint64_t)3);   /* preserved, not 0xFF00000000000003 */
}

/* ---- GROUP 18: load value visible in dest reg of next trace entry ----- */

static void test_integration_load_updates_dest_reg(void) {
    /*
     * Code: LDR X0,[X29]; LDR X1,[X29]
     * mem[0..7] = 0xDEADC0DE
     *
     * The second trace entry is captured AFTER the first LDR executes.
     * So cpu_state.gpr[0] in entry[1] must equal the loaded value.
     */
    uint32_t code[] = {
        enc_ldr_reg(0, 29),   /* LDR X0,[X29] */
        enc_ldr_reg(1, 29),   /* LDR X1,[X29] */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t loaded_val = UINT64_C(0x00000000DEADC0DE);
    memcpy(mem, &loaded_val, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 2, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 2);

    instr_trace_entry_t *e1 = find_mem_entry(&res, 1);  /* second LDR */
    EXPECT(e1 != NULL);
    if (!e1) return;

    /* cpu_state captured before LDR X1 executes = after LDR X0 executed */
    EXPECT_EQ(e1->cpu.gpr[0], loaded_val);
}

/* ---- GROUP 19: post-index writeback reflected in next cpu_state.gpr[29] */

static void test_integration_postidx_writeback_cpu_state(void) {
    /*
     * Code: LDR X0,[X29],#8; LDR X1,[X29]
     * After the first instruction, X29 is updated to kbase+8.
     * apply_fixups in the hook for LDR X1 converts uaddr+8 back to kbase+8.
     * So the second trace entry's cpu.gpr[29] must equal kbase+8.
     */
    uint32_t code[] = {
        enc_ldr_postidx(0, 29, 8),
        enc_ldr_reg(1, 29),
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0xAAAAAAAAAAAAAAAA);
    uint64_t v8 = UINT64_C(0xBBBBBBBBBBBBBBBB);
    memcpy(mem,     &v0, 8);
    memcpy(mem + 8, &v8, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 2, mem, sizeof(mem), KBASE, 0, 0u, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    instr_trace_entry_t *e1 = find_mem_entry(&res, 1);  /* LDR X1,[X29] */
    EXPECT(e1 != NULL);
    if (!e1) return;

    /* gpr[29] = X29 must show kbase+8, not the uaddr */
    EXPECT_EQ(e1->cpu.gpr[29], KBASE + 8);
    EXPECT_EQ(e1->metadata.memory_access.effective_address, KBASE + 8);
}

/* ---- GROUP 20c: window_id — unique per speculative excursion ----------- */

static void test_integration_window_id_unique_per_excursion(void) {
    /*
     * Two CBZ branches, then an architectural load. With X0=0 both CBZs are TAKEN, so each skips its
     * own LDR architecturally and mispredicts into a SEPARATE depth-1 window; the trailing LDR X3 runs
     * architecturally. The two speculative loads share nesting=1 but MUST carry DIFFERENT window ids
     * (checkpoint slots are reused across excursions; window ids are not). This is what lets the taint
     * tracker tell a dead sibling flow from the live one. Architectural entries carry window_id 0.
     */
    uint32_t code[] = {
        enc_cbz(0, +8),               /* CBZ X0,+8  taken -> skip LDR X1; window A explores fall-through */
        enc_ldr_reg(1, 29),           /* LDR X1,[X29]        (speculative only) */
        enc_cbz(0, +8),               /* CBZ X0,+8  taken -> skip LDR X2; window B explores fall-through */
        enc_ldr_unsigned(2, 29, 8),   /* LDR X2,[X29,#8]     (speculative only) */
        enc_ldr_unsigned(3, 29, 16),  /* LDR X3,[X29,#16]    (architectural) */
    };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));

    ce_result_t res;
    if (!run_ce_simple(code, 5, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    /* Order-independent checks: architectural entries carry window_id 0, speculative entries carry a
     * non-zero id, and (the fix) two speculative entries at the SAME nesting come from distinct
     * excursions and so carry DIFFERENT window ids — which a reused checkpoint slot could never show. */
    int n = count_mem_entries(&res);
    int arch_seen = 0, arch_window_ok = 1, spec_ids_nonzero = 1, same_depth_distinct = 0;
    for (int i = 0; i < n; ++i) {
        instr_trace_entry_t *ei = find_mem_entry(&res, i);
        if (0 == ei->metadata.speculation_nesting) {
            arch_seen = 1;
            if (0 != ei->metadata.window_id) arch_window_ok = 0;
            continue;
        }
        if (0 == ei->metadata.window_id) spec_ids_nonzero = 0;
        for (int j = i + 1; j < n; ++j) {
            instr_trace_entry_t *ej = find_mem_entry(&res, j);
            if (ej->metadata.speculation_nesting == ei->metadata.speculation_nesting &&
                ej->metadata.window_id != ei->metadata.window_id) {
                same_depth_distinct = 1;
            }
        }
    }

    EXPECT(arch_seen);                 /* the trailing LDR X3 ran architecturally */
    EXPECT(arch_window_ok);            /* architectural entries have window_id 0 */
    EXPECT(spec_ids_nonzero);          /* speculative entries have a non-zero window id */
    EXPECT(same_depth_distinct);       /* sibling flows at one depth get distinct window ids */
}

/* ---- GROUP 20: ALWAYS_MISPREDICT CBZ taken — spec NOT-TAKEN first ------ */

static void test_integration_always_mispredict_cbz(void) {
    /*
     * Code: CBZ X0, +8; LDR X1,[X29]; LDR X2,[X29,#8]
     * X0=0 → arch TAKEN (jump to LDR X2).
     * ALWAYS_MISPREDICT → spec NOT-TAKEN (fall through to LDR X1 first).
     *
     * Expected memory-access entries in order:
     *   [0] LDR X1  nesting=1   EA=kbase     (spec: not-taken fall-through)
     *   [1] LDR X2  nesting=1   EA=kbase+8   (still on spec path)
     *   [2] LDR X2  nesting=0   EA=kbase+8   (arch taken after checkpoint restore)
     */
    uint32_t code[] = {
        enc_cbz(0, +8),              /* CBZ X0, +8  (skip next insn if X0==0) */
        enc_ldr_reg(1, 29),          /* LDR X1,[X29]      offset 0 */
        enc_ldr_unsigned(2, 29, 8),  /* LDR X2,[X29,#8]   offset 8 */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0x1111111111111111);
    uint64_t v8 = UINT64_C(0x2222222222222222);
    memcpy(mem,     &v0, 8);
    memcpy(mem + 8, &v8, 8);

    /* X0=0, NZCV=0, max_nesting=1 */
    ce_result_t res;
    if (!run_ce_simple(code, 3, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 3);

    instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
    instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
    instr_trace_entry_t *ma2 = find_mem_entry(&res, 2);
    EXPECT(ma0 != NULL); EXPECT(ma1 != NULL); EXPECT(ma2 != NULL);
    if (!ma0 || !ma1 || !ma2) return;

    /* Speculative: LDR X1 at kbase, nesting=1 */
    EXPECT_EQ(ma0->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(ma0->metadata.memory_access.before,            v0);
    EXPECT_EQ(ma0->metadata.speculation_nesting,             (uint64_t)1);

    /* Speculative: LDR X2 at kbase+8, nesting=1 */
    EXPECT_EQ(ma1->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma1->metadata.memory_access.before,            v8);
    EXPECT_EQ(ma1->metadata.speculation_nesting,             (uint64_t)1);

    /* Architectural: LDR X2 at kbase+8, nesting=0 */
    EXPECT_EQ(ma2->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma2->metadata.memory_access.before,            v8);
    EXPECT_EQ(ma2->metadata.speculation_nesting,             (uint64_t)0);
}

/* ---- GROUP 20b: per-window instruction cap (max_misspred_instructions) ---- */

static void test_integration_spec_window_cap(void) {
    /* Same TC as GROUP 20 (uncapped: 2 speculative + 1 arch access). With max_misspred_instructions=1
     * the window runs one instruction (LDR X1) then is cut before the second (LDR X2), so
     * only one speculative access survives. */
    uint32_t code[] = {
        enc_cbz(0, +8),
        enc_ldr_reg(1, 29),
        enc_ldr_unsigned(2, 29, 8),
    };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));

    ce_result_t res;
    if (!run_ce_full(code, 3, mem, sizeof(mem), KBASE, 1, /*max_instr=*/1,
                     EXEC_CLAUSE_COND, BRANCH_PREDICTOR_NONE, NULL, NULL, 0, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 2);   /* one speculative + one architectural (was 3 uncapped) */
    instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
    instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
    EXPECT(ma0 != NULL); EXPECT(ma1 != NULL);
    if (!ma0 || !ma1) return;
    EXPECT_EQ(ma0->metadata.speculation_nesting, (uint64_t)1);   /* spec LDR X1 survives the cap */
    EXPECT_EQ(ma1->metadata.speculation_nesting, (uint64_t)0);   /* arch LDR X2 after the window cut */
}

/* ---- GROUP 21: ALWAYS_MISPREDICT CBNZ not-taken — spec TAKEN first ---- */

static void test_integration_always_mispredict_cbnz(void) {
    /*
     * Code: CBNZ X0, +8; LDR X1,[X29]; LDR X2,[X29,#8]
     * X0=0 → CBNZ arch NOT-TAKEN (fall through to LDR X1).
     * ALWAYS_MISPREDICT → spec TAKEN (jump to LDR X2).
     *
     * Expected memory-access entries:
     *   [0] LDR X2  nesting=1   (spec: taken path)
     *   [1] LDR X1  nesting=0   (arch: not-taken fall-through)
     *   [2] LDR X2  nesting=0
     */
    uint32_t code[] = {
        enc_cbnz(0, +8),             /* CBNZ X0, +8 */
        enc_ldr_reg(1, 29),          /* LDR X1,[X29]    */
        enc_ldr_unsigned(2, 29, 8),  /* LDR X2,[X29,#8] */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0x3333333333333333);
    uint64_t v8 = UINT64_C(0x4444444444444444);
    memcpy(mem,     &v0, 8);
    memcpy(mem + 8, &v8, 8);

    ce_result_t res;
    if (!run_ce_simple(code, 3, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 3);

    instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
    instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
    instr_trace_entry_t *ma2 = find_mem_entry(&res, 2);
    EXPECT(ma0 != NULL); EXPECT(ma1 != NULL); EXPECT(ma2 != NULL);
    if (!ma0 || !ma1 || !ma2) return;

    /* Spec: taken → LDR X2 */
    EXPECT_EQ(ma0->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma0->metadata.speculation_nesting,             (uint64_t)1);

    /* Arch not-taken: LDR X1, then LDR X2 */
    EXPECT_EQ(ma1->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(ma1->metadata.speculation_nesting,             (uint64_t)0);
    EXPECT_EQ(ma2->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma2->metadata.speculation_nesting,             (uint64_t)0);
}

/* ---- GROUP 22: TBZ bit32 — verifies b5 extraction fix ----------------- */

static void test_integration_tbz_bit32(void) {
    /*
     * Code: TBZ X0, #32, +8; LDR X1,[X29]; LDR X2,[X29,#8]
     *
     * Sub-case A: X0 has bit 32 SET → TBZ NOT-TAKEN (bit is not zero).
     *   ALWAYS_MISPREDICT → spec TAKEN (LDR X2 at nesting=1),
     *   then arch NOT-TAKEN: LDR X1 (nesting=0), LDR X2 (nesting=0).
     *
     * Sub-case B: X0 has bit 32 CLEAR (X0=0) → TBZ TAKEN.
     *   ALWAYS_MISPREDICT → spec NOT-TAKEN: LDR X1 (nesting=1), LDR X2 (nesting=1),
     *   then arch TAKEN: LDR X2 (nesting=0).
     *
     * A bug in b5 extraction (using insn bit 31 instead of insn bit 31 for the
     * 6-bit test-bit index) would misidentify the bit being tested, causing the
     * spec/arch split to be on the wrong path.
     */
    uint32_t code[] = {
        enc_tbz(0, 32, +8),          /* TBZ X0, #32, +8 */
        enc_ldr_reg(1, 29),          /* LDR X1,[X29]    */
        enc_ldr_unsigned(2, 29, 8),  /* LDR X2,[X29,#8] */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0xAAAAAAAAAAAAAAAA);
    uint64_t v8 = UINT64_C(0xBBBBBBBBBBBBBBBB);
    memcpy(mem,     &v0, 8);
    memcpy(mem + 8, &v8, 8);

    /* --- sub-case A: bit 32 SET → TBZ condition fails → NOT-TAKEN arch --- */
    {
        uint64_t regs[REGS_COUNT] = { 0 };
        regs[0] = UINT64_C(1) << 32;  /* bit 32 set */

        ce_result_t res;
        if (!run_ce(code, 3, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND, regs, &res)) {
            ++g_tests_run; ++g_tests_failed;
            fprintf(stderr, "FAIL %s(A): run_ce failed\n", __func__);
        } else {
            EXPECT_EQ(count_mem_entries(&res), 3);
            instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
            instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
            instr_trace_entry_t *ma2 = find_mem_entry(&res, 2);
            EXPECT(ma0 != NULL); EXPECT(ma1 != NULL); EXPECT(ma2 != NULL);
            if (ma0 && ma1 && ma2) {
                /* Spec TAKEN: LDR X2 */
                EXPECT_EQ(ma0->metadata.memory_access.effective_address, KBASE + 8);
                EXPECT_EQ(ma0->metadata.speculation_nesting,             (uint64_t)1);
                /* Arch NOT-TAKEN: LDR X1, LDR X2 */
                EXPECT_EQ(ma1->metadata.memory_access.effective_address, KBASE);
                EXPECT_EQ(ma1->metadata.speculation_nesting,             (uint64_t)0);
                EXPECT_EQ(ma2->metadata.memory_access.effective_address, KBASE + 8);
                EXPECT_EQ(ma2->metadata.speculation_nesting,             (uint64_t)0);
            }
        }
    }

    /* --- sub-case B: bit 32 CLEAR (X0=0) → TBZ TAKEN arch --- */
    {
        ce_result_t res;
        if (!run_ce_simple(code, 3, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND, &res)) {
            ++g_tests_run; ++g_tests_failed;
            fprintf(stderr, "FAIL %s(B): run_ce failed\n", __func__);
        } else {
            EXPECT_EQ(count_mem_entries(&res), 3);
            instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
            instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
            instr_trace_entry_t *ma2 = find_mem_entry(&res, 2);
            EXPECT(ma0 != NULL); EXPECT(ma1 != NULL); EXPECT(ma2 != NULL);
            if (ma0 && ma1 && ma2) {
                /* Spec NOT-TAKEN: LDR X1, LDR X2 */
                EXPECT_EQ(ma0->metadata.memory_access.effective_address, KBASE);
                EXPECT_EQ(ma0->metadata.speculation_nesting,             (uint64_t)1);
                EXPECT_EQ(ma1->metadata.memory_access.effective_address, KBASE + 8);
                EXPECT_EQ(ma1->metadata.speculation_nesting,             (uint64_t)1);
                /* Arch TAKEN: LDR X2 */
                EXPECT_EQ(ma2->metadata.memory_access.effective_address, KBASE + 8);
                EXPECT_EQ(ma2->metadata.speculation_nesting,             (uint64_t)0);
            }
        }
    }
}

/* ---- GROUP 22b: ALWAYS_MISPREDICT B.cond taken — spec NOT-TAKEN first -- */

static void test_integration_always_mispredict_bcond(void) {
    /*
     * Code: B.EQ +8; LDR X1,[X29]; LDR X2,[X29,#8]
     * NZCV Z=1 → B.EQ arch TAKEN (jump to LDR X2).
     * ALWAYS_MISPREDICT → spec NOT-TAKEN (fall through to LDR X1 first).
     *
     * Expected memory-access entries:
     *   [0] LDR X1  nesting=1   (spec: not-taken fall-through)
     *   [1] LDR X2  nesting=1
     *   [2] LDR X2  nesting=0   (arch taken after checkpoint restore)
     */
    uint32_t code[] = {
        enc_b_cond(0x0, +8),         /* B.EQ +8  (cond 0x0 = EQ, taken iff Z=1) */
        enc_ldr_reg(1, 29),          /* LDR X1,[X29]      offset 0 */
        enc_ldr_unsigned(2, 29, 8),  /* LDR X2,[X29,#8]   offset 8 */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0x5555555555555555);
    uint64_t v8 = UINT64_C(0x6666666666666666);
    memcpy(mem,     &v0, 8);
    memcpy(mem + 8, &v8, 8);

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[6] = UINT64_C(1) << 30;     /* NZCV slot (PSTATE format): Z=1 */

    ce_result_t res;
    if (!run_ce(code, 3, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 3);
    instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
    instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
    instr_trace_entry_t *ma2 = find_mem_entry(&res, 2);
    EXPECT(ma0 != NULL); EXPECT(ma1 != NULL); EXPECT(ma2 != NULL);
    if (!ma0 || !ma1 || !ma2) return;

    /* Spec NOT-TAKEN: LDR X1, LDR X2 */
    EXPECT_EQ(ma0->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(ma0->metadata.speculation_nesting,             (uint64_t)1);
    EXPECT_EQ(ma1->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma1->metadata.speculation_nesting,             (uint64_t)1);
    /* Arch TAKEN: LDR X2 */
    EXPECT_EQ(ma2->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma2->metadata.speculation_nesting,             (uint64_t)0);
}

/* ---- GROUP 22c: ALWAYS_MISPREDICT TBNZ not-taken — spec TAKEN first ---- */

static void test_integration_always_mispredict_tbnz(void) {
    /*
     * Code: TBNZ X0, #5, +8; LDR X1,[X29]; LDR X2,[X29,#8]
     * X0=0 (bit 5 clear) → TBNZ arch NOT-TAKEN (fall through to LDR X1).
     * ALWAYS_MISPREDICT → spec TAKEN (jump to LDR X2 first).
     *
     * Expected memory-access entries:
     *   [0] LDR X2  nesting=1   (spec: taken path)
     *   [1] LDR X1  nesting=0   (arch: not-taken fall-through)
     *   [2] LDR X2  nesting=0
     */
    uint32_t code[] = {
        enc_tbnz(0, 5, +8),          /* TBNZ X0, #5, +8 */
        enc_ldr_reg(1, 29),          /* LDR X1,[X29]    */
        enc_ldr_unsigned(2, 29, 8),  /* LDR X2,[X29,#8] */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0x7777777777777777);
    uint64_t v8 = UINT64_C(0x8888888888888888);
    memcpy(mem,     &v0, 8);
    memcpy(mem + 8, &v8, 8);

    /* X0=0 → bit 5 clear → TBNZ not-taken (all-zero regs) */
    ce_result_t res;
    if (!run_ce_simple(code, 3, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 3);
    instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
    instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
    instr_trace_entry_t *ma2 = find_mem_entry(&res, 2);
    EXPECT(ma0 != NULL); EXPECT(ma1 != NULL); EXPECT(ma2 != NULL);
    if (!ma0 || !ma1 || !ma2) return;

    /* Spec TAKEN: LDR X2 */
    EXPECT_EQ(ma0->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma0->metadata.speculation_nesting,             (uint64_t)1);
    /* Arch NOT-TAKEN: LDR X1, LDR X2 */
    EXPECT_EQ(ma1->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(ma1->metadata.speculation_nesting,             (uint64_t)0);
    EXPECT_EQ(ma2->metadata.memory_access.effective_address, KBASE + 8);
    EXPECT_EQ(ma2->metadata.speculation_nesting,             (uint64_t)0);
}

/* ---- GROUP 23: ARCH_ONLY vs ALWAYS_MISPREDICT — trace count differs --- */

static void test_integration_contract_type_comparison(void) {
    /*
     * Same code: CBZ X0, +8; LDR X1,[X29]; LDR X2,[X29,#8]
     * X0=0 → branch TAKEN (arch: only LDR X2 executes).
     *
     * ARCH_ONLY:           1 memory-access entry (LDR X2)
     * ALWAYS_MISPREDICT:   3 memory-access entries (LDR X1 spec, LDR X2 spec, LDR X2 arch)
     */
    uint32_t code[] = {
        enc_cbz(0, +8),
        enc_ldr_reg(1, 29),
        enc_ldr_unsigned(2, 29, 8),
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));

    ce_result_t arch_res, spec_res;

    bool arch_ok = run_ce_simple(code, 3, mem, sizeof(mem), KBASE, 1, 0u,          &arch_res);
    bool spec_ok = run_ce_simple(code, 3, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND,  &spec_res);

    EXPECT(arch_ok);
    EXPECT(spec_ok);

    if (arch_ok) {
        EXPECT_EQ(count_mem_entries(&arch_res), 1);
        instr_trace_entry_t *e = find_mem_entry(&arch_res, 0);
        EXPECT(e != NULL);
        if (e) EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE + 8);
    }

    if (spec_ok) {
        EXPECT_EQ(count_mem_entries(&spec_res), 3);
    }

    /* The speculative trace must be strictly larger than the arch-only trace */
    if (arch_ok && spec_ok) {
        EXPECT(count_mem_entries(&spec_res) > count_mem_entries(&arch_res));
    }
}

/* ---- GROUP 24: PACIA+AUTIA round-trip on arch path -------------------- */

/* PACIA X<rd>, X<rn> */
static uint32_t enc_pacia(int rd, int rn) {
    return 0xDAC10000u | ((uint32_t)rn << 5) | (uint32_t)rd;
}
/* AUTIA X<rd>, X<rn> */
static uint32_t enc_autia(int rd, int rn) {
    return 0xDAC11000u | ((uint32_t)rn << 5) | (uint32_t)rd;
}

static void test_integration_pac_arch_roundtrip(void) {
    SKIP_IF_NO_EXECUTOR();
    /*
     * X0 = kbase (the pointer to sign), X29 = kbase (set by main.c).
     * Code: PACIA X0,X29  →  AUTIA X0,X29  →  LDR X1,[X29]
     *
     * pac_sign_hook intercepts PACIA and signs kbase with kbase-as-context.
     * auth_verify_hook intercepts AUTIA on the arch path (nesting==0) and
     * runs the real AUT, recovering the clean kbase pointer.
     *
     * After the round-trip X0 must equal kbase again.  LDR X1 produces a
     * trace entry whose cpu.gpr[0] (= X0 captured before LDR executes) must
     * equal kbase.  The EA must also be kbase (X29 is untouched).
     */
    uint32_t code[] = {
        enc_pacia(0, 29),    /* PACIA X0, X29 */
        enc_autia(0, 29),    /* AUTIA X0, X29 */
        enc_ldr_reg(1, 29),  /* LDR  X1, [X29] */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t val = UINT64_C(0xABCDABCDABCDABCD);
    memcpy(mem, &val, 8);

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = KBASE;  /* X0 = kbase (pointer to sign) */

    ce_result_t res;
    if (!run_ce(code, 3, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    /* Only LDR X1 produces a memory-access entry; PACIA and AUTIA do not */
    EXPECT_EQ(count_mem_entries(&res), 1);

    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(e->metadata.memory_access.element_size,      (uint64_t)8);
    EXPECT_EQ(e->metadata.speculation_nesting,             (uint64_t)0);
    /* X0 must be restored to kbase after the PACIA→AUTIA round-trip */
    EXPECT_EQ(e->cpu.gpr[0], KBASE);
}

/* ---- GROUP 25: AUTIA on spec path uses XPAC, not real AUT ------------- */

static void test_integration_pac_spec_xpac(void) {
    SKIP_IF_NO_EXECUTOR();
    /*
     * X0=0, X1=kbase, X29=kbase (main.c), max_nesting=1,
     * ALWAYS_MISPREDICT contract.
     *
     * Code layout (byte offsets):
     *   [0]  CBZ X0, +12   ; X0=0 → arch TAKEN (→ [3] LDR X2)
     *                       ; ALWAYS_MISPREDICT → spec NOT-TAKEN (→ [1])
     *   [4]  PACIA X1, X29 ; spec: sign kbase with kbase context → X1=signed
     *   [8]  AUTIA X1, X29 ; spec: auth_verify_hook sees is_in_speculation()==1
     *                       ;       → do_xpac_el0 strips tag → X1=kbase again
     *   [12] LDR X2, [X29] ; both paths: EA=kbase
     *
     * Expected memory-access entries:
     *   [0] LDR X2 nesting=1  (spec: after PACIA+XPAC, cpu.gpr[1]=kbase)
     *   [1] LDR X2 nesting=0  (arch: checkpoint-restored X1=kbase)
     *
     * If auth_verify_hook incorrectly ran the real AUT on the speculative
     * path, the AUTIA would authenticate a pointer signed with PACIA (so it
     * would actually still succeed here). More importantly, the test
     * validates end-to-end plumbing: the CE must NOT crash, the trace must
     * contain exactly 2 memory entries, and X1 must equal kbase in both.
     */
    uint32_t code[] = {
        enc_cbz(0, 12),      /* CBZ  X0, +12   [0] */
        enc_pacia(1, 29),    /* PACIA X1, X29  [4] */
        enc_autia(1, 29),    /* AUTIA X1, X29  [8] */
        enc_ldr_reg(2, 29),  /* LDR  X2, [X29] [12] */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t val = UINT64_C(0x1234567812345678);
    memcpy(mem, &val, 8);

    uint64_t regs[REGS_COUNT] = { 0 };
    /* regs[0] = 0 (X0=0 so CBZ is TAKEN) */
    regs[1] = KBASE;  /* X1 = kbase (to be signed by PACIA on spec path) */

    ce_result_t res;
    if (!run_ce(code, 4, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 2);

    instr_trace_entry_t *ma0 = find_mem_entry(&res, 0);
    instr_trace_entry_t *ma1 = find_mem_entry(&res, 1);
    EXPECT(ma0 != NULL); EXPECT(ma1 != NULL);
    if (!ma0 || !ma1) return;

    /* Speculative entry: PACIA then XPAC restore X1 to kbase */
    EXPECT_EQ(ma0->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(ma0->metadata.speculation_nesting,             (uint64_t)1);
    EXPECT_EQ(ma0->cpu.gpr[1],                               KBASE);

    /* Arch entry: X1 restored from checkpoint (= original kbase) */
    EXPECT_EQ(ma1->metadata.memory_access.effective_address, KBASE);
    EXPECT_EQ(ma1->metadata.speculation_nesting,             (uint64_t)0);
    EXPECT_EQ(ma1->cpu.gpr[1],                               KBASE);
}

/* ---- GROUP 26: PACIZA+AUTIZA round-trip — zero-context variant --------- */

/* PACIZA X<rd>: sign with XZR context (ctx=0) */
static uint32_t enc_paciza(int rd) { return 0xDAC123E0u | (uint32_t)rd; }
/* AUTIZA X<rd>: auth with XZR context */
static uint32_t enc_autiza(int rd) { return 0xDAC133E0u | (uint32_t)rd; }

static void test_integration_paciza_autiza_roundtrip(void) {
    SKIP_IF_NO_EXECUTOR();
    /*
     * PACIZA X0 signs X0=kbase with context=0 (XZR).
     * AUTIZA X0 authenticates on arch path; must recover kbase exactly.
     * Verifies zero-context dispatch through ioctl 17/18 kernel handlers.
     *
     * Memory layout (3 distinct sentinel values at different offsets):
     *   [+0x00]  0xAA11BB22CC33DD44  ← target of LDR X1,[X29] if pointer OK
     *   [+0x08]  0x5566778899AABBCC  ← would be loaded if pointer off by 8
     *   [+0x10]  0xDEAD000000000001  ← would be loaded if pointer off by 16
     *
     * If AUTIZA fails to restore kbase the LDR hits a wrong EA and the
     * effective_address and before checks below expose the failure.
     */
    static const uint64_t VAL0 = UINT64_C(0xAA11BB22CC33DD44);
    static const uint64_t VAL8 = UINT64_C(0x5566778899AABBCC);
    static const uint64_t VAL16 = UINT64_C(0xDEAD000000000001);

    uint32_t code[] = {
        enc_paciza(0),         /* PACIZA X0                   */
        enc_autiza(0),         /* AUTIZA X0 → must restore X0 */
        enc_ldr_reg(1, 0),     /* LDR X1,[X0] — load via recovered pointer */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    memcpy(mem + 0x00, &VAL0,  8);
    memcpy(mem + 0x08, &VAL8,  8);
    memcpy(mem + 0x10, &VAL16, 8);

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = KBASE;

    ce_result_t res;
    if (!run_ce(code, 3, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 1);
    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    /* Pointer must be fully restored — effective_address must equal kbase */
    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    /* Value at kbase offset 0 must be VAL0 */
    EXPECT_EQ(e->metadata.memory_access.before, VAL0);
    EXPECT_EQ(e->metadata.memory_access.element_size, (uint64_t)8);
    EXPECT_EQ(e->metadata.memory_access.is_write, (uint64_t)0);
    EXPECT_EQ(e->metadata.speculation_nesting, (uint64_t)0);
    /* gpr[0] must still hold the recovered (stripped) kbase */
    EXPECT_EQ(e->cpu.gpr[0], KBASE);
}

/* ---- GROUP 27: PACDA+AUTDA round-trip — D-key variant ------------------ */

/* PACDA X<rd>, X<rn>: data key A */
static uint32_t enc_pacda(int rd, int rn) {
    return 0xDAC10800u | ((uint32_t)rn << 5) | (uint32_t)rd;
}
/* AUTDA X<rd>, X<rn>: data key A */
static uint32_t enc_autda(int rd, int rn) {
    return 0xDAC11800u | ((uint32_t)rn << 5) | (uint32_t)rd;
}

static void test_integration_pacda_autda_roundtrip(void) {
    SKIP_IF_NO_EXECUTOR();
    /*
     * PACDA X0,X29 signs with the DA (data-key A) key, context=kbase (X29).
     * AUTDA X0,X29 authenticates on arch path; must recover X0=kbase exactly.
     * Verifies D-key dispatch through ioctl 17/18 kernel handlers.
     *
     * Memory layout (3 distinct sentinel values at different offsets):
     *   [+0x00]  0xDEADBEEFCAFEBABE  ← target of LDR X1,[X0] if pointer OK
     *   [+0x08]  0x0102030405060708  ← would be loaded if pointer off by 8
     *   [+0x10]  0xFEDCBA9876543210  ← would be loaded if pointer off by 16
     *
     * If AUTDA fails to restore kbase the LDR hits a wrong EA and the
     * effective_address and before checks below expose the failure.
     */
    static const uint64_t VAL0  = UINT64_C(0xDEADBEEFCAFEBABE);
    static const uint64_t VAL8  = UINT64_C(0x0102030405060708);
    static const uint64_t VAL16 = UINT64_C(0xFEDCBA9876543210);

    uint32_t code[] = {
        enc_pacda(0, 29),      /* PACDA X0, X29                    */
        enc_autda(0, 29),      /* AUTDA X0, X29 → must restore X0  */
        enc_ldr_reg(1, 0),     /* LDR X1,[X0] — load via recovered pointer */
    };

    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    memcpy(mem + 0x00, &VAL0,  8);
    memcpy(mem + 0x08, &VAL8,  8);
    memcpy(mem + 0x10, &VAL16, 8);

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = KBASE;

    ce_result_t res;
    if (!run_ce(code, 3, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }

    EXPECT_EQ(count_mem_entries(&res), 1);
    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    /* Pointer must be fully restored — effective_address must equal kbase */
    EXPECT_EQ(e->metadata.memory_access.effective_address, KBASE);
    /* Value at kbase offset 0 must be VAL0 */
    EXPECT_EQ(e->metadata.memory_access.before, VAL0);
    EXPECT_EQ(e->metadata.memory_access.element_size, (uint64_t)8);
    EXPECT_EQ(e->metadata.memory_access.is_write, (uint64_t)0);
    EXPECT_EQ(e->metadata.speculation_nesting, (uint64_t)0);
    /* gpr[0] must still hold the recovered (stripped) kbase */
    EXPECT_EQ(e->cpu.gpr[0], KBASE);
}

/* ---- main -------------------------------------------------------------- */

/* ---- bpas: store-then-load to the same address ----------------------- */
static void test_integration_bpas_store_bypass(void) {
    /*
     * STR X0,[X29] ; LDR X1,[X29].  mem[0]=STALE, X0=NEW.
     * seq:  the load reads the stored (NEW) value -> store + 1 load.
     * bpas: the store is bypassed, so a speculative load reads the STALE value; after the
     *       window the load re-runs architecturally with NEW -> store + 2 loads.
     */
    uint32_t code[] = { enc_str_reg(0, 29), enc_ldr_reg(1, 29) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t stale = UINT64_C(0xDEADDEADDEADDEAD);
    memcpy(mem, &stale, 8);
    uint64_t newv = UINT64_C(0xBEEFBEEFBEEFBEEF);
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = newv;

    ce_result_t seq_res, bpas_res;
    bool seq_ok  = run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u,              regs, &seq_res);
    bool bpas_ok = run_ce(code, 2, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_BPAS, regs, &bpas_res);
    EXPECT(seq_ok);
    EXPECT(bpas_ok);

    if (seq_ok) {
        EXPECT_EQ(count_mem_entries(&seq_res), 2);
        instr_trace_entry_t *ld = find_mem_entry(&seq_res, 1);   /* the load */
        EXPECT(ld != NULL);
        if (ld) EXPECT_EQ(ld->metadata.memory_access.before, newv);   /* arch store-forward */
    }

    if (bpas_ok) {
        EXPECT(count_mem_entries(&bpas_res) > count_mem_entries(&seq_res));
        int saw_stale_spec_load = 0;
        for (int i = 0; i < count_mem_entries(&bpas_res); ++i) {
            instr_trace_entry_t *e = find_mem_entry(&bpas_res, i);
            if (e && !e->metadata.memory_access.is_write &&
                e->metadata.speculation_nesting > 0 &&
                e->metadata.memory_access.before == stale) {
                saw_stale_spec_load = 1;
            }
        }
        EXPECT(saw_stale_spec_load);
    }
}

/* ---- bpas: a store with no following instruction must not corrupt ----- */
static void test_integration_bpas_trailing_store(void) {
    /* The store sets a pending bypass that is never applied (the sim ends). Must not crash;
     * the trace is just the store. Guards the "pending dangles past the window" case. */
    uint32_t code[] = { enc_str_reg(0, 29) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    ce_result_t res;
    bool ok = run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_BPAS, &res);
    EXPECT(ok);
    if (ok) EXPECT_EQ(count_mem_entries(&res), 1);
}

/* ---- unsupported clause combination must be rejected by the CE -------- */
static void test_integration_unsupported_clauses_rejected(void) {
    /* COND|BPU is two branch models — not a supported contract; the CE must reject it. */
    uint32_t code[] = { enc_ldr_reg(0, 29) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    ce_result_t res;
    bool ok = run_ce_simple(code, 1, mem, sizeof(mem), KBASE, 8,
                            EXEC_CLAUSE_COND | EXEC_CLAUSE_BPU, &res);
    EXPECT(!ok);   /* CE traps on the unsupported bitmask */
}

/* ---- bpas preserves a store's register writeback while bypassing its memory write ---- */
static void test_integration_bpas_preserves_writeback(void) {
    /* STR X0,[X1],#8 bypassed: the memory write is undone, but the post-index writeback
     * (X1 += 8) must persist (the HW applied it; only memory is restored). So the following
     * load's EA is X1+8 even on the speculative (stale-memory) path. */
    uint32_t code[] = { enc_str_postidx(0, 1, 8), enc_ldr_reg(2, 1) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = KBASE;                              /* X1 = base */
    ce_result_t res;
    bool ok = run_ce(code, 2, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_BPAS, regs, &res);
    EXPECT(ok);
    if (ok) {
        int saw_writeback = 0;
        for (int i = 0; i < count_mem_entries(&res); ++i) {
            instr_trace_entry_t *e = find_mem_entry(&res, i);
            if (e && !e->metadata.memory_access.is_write &&
                e->metadata.speculation_nesting > 0 &&
                e->metadata.memory_access.effective_address == KBASE + 8) {
                saw_writeback = 1;
            }
        }
        EXPECT(saw_writeback);   /* would be KBASE (not +8) if writeback were lost */
    }
}

/* ---- bpas bypasses a whole store-pair (both elements) ---- */
static void test_integration_bpas_store_pair(void) {
    /* STP X0,X1,[X29] bypassed: the whole-memory restore undoes BOTH elements, so loads from
     * [X29] and [X29+8] read the stale values on the speculative path. */
    uint32_t code[] = { enc_stp(0, 1, 29), enc_ldr_reg(2, 29), enc_ldr_unsigned(3, 29, 8) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t stale0 = UINT64_C(0x1111111111111111), stale8 = UINT64_C(0x2222222222222222);
    memcpy(mem, &stale0, 8);
    memcpy(mem + 8, &stale8, 8);
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = 0xAAAA;                             /* new values the STP would write */
    regs[1] = 0xBBBB;
    ce_result_t res;
    bool ok = run_ce(code, 3, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_BPAS, regs, &res);
    EXPECT(ok);
    if (ok) {
        int saw0 = 0, saw8 = 0;
        for (int i = 0; i < count_mem_entries(&res); ++i) {
            instr_trace_entry_t *e = find_mem_entry(&res, i);
            if (e && !e->metadata.memory_access.is_write && e->metadata.speculation_nesting > 0) {
                if (e->metadata.memory_access.before == stale0) saw0 = 1;
                if (e->metadata.memory_access.before == stale8) saw8 = 1;
            }
        }
        EXPECT(saw0);   /* first element bypassed */
        EXPECT(saw8);   /* second element bypassed too */
    }
}

/* ---- cond-bpas composition: both store-bypass and branch-mispredict windows fire ---- */
static void test_integration_cond_bpas_composition(void) {
    /* A store followed by a conditional branch. Under cond-bpas the trace must be at least as
     * large as either clause alone, and strictly larger than seq — i.e. both speculations run. */
    uint32_t code[] = {
        enc_str_reg(0, 29),          /* STR X0,[X29] */
        enc_cbz(3, +8),              /* CBZ X3, +8   */
        enc_ldr_reg(1, 29),          /* LDR X1,[X29] */
        enc_ldr_unsigned(2, 29, 8),
    };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };           /* X0=0 (store value), X3=0 (CBZ taken) */
    ce_result_t seq_r, cond_r, bpas_r, both_r;
    bool sq = run_ce(code, 4, mem, sizeof(mem), KBASE, 0, 0u, regs, &seq_r);
    bool cd = run_ce(code, 4, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_COND, regs, &cond_r);
    bool bp = run_ce(code, 4, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_BPAS, regs, &bpas_r);
    bool bo = run_ce(code, 4, mem, sizeof(mem), KBASE, 8,
                     EXEC_CLAUSE_COND | EXEC_CLAUSE_BPAS, regs, &both_r);
    EXPECT(sq); EXPECT(cd); EXPECT(bp); EXPECT(bo);
    if (sq && cd && bp && bo) {
        int ns = count_mem_entries(&seq_r),  nc = count_mem_entries(&cond_r);
        int nb = count_mem_entries(&bpas_r), nbo = count_mem_entries(&both_r);
        EXPECT(nbo >= nc);   /* at least what cond explores */
        EXPECT(nbo >= nb);   /* at least what bpas explores */
        EXPECT(nbo > ns);    /* strictly more than no speculation */
    }
}

/* ---- bpas with max_nesting=0 must gate the bypass off entirely (== seq) ---- */
static void test_integration_bpas_max_nesting_zero(void) {
    /* Phase A only fires when spec_nesting() < spec_max_nesting(); with the cap at 0 that is
     * 0 < 0 == false, so no window ever opens. STR;LDR must then behave exactly like seq:
     * the store executes and the load reads the NEW (forwarded) value — no stale spec read. */
    uint32_t code[] = { enc_str_reg(0, 29), enc_ldr_reg(1, 29) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t stale = UINT64_C(0xDEADDEADDEADDEAD);
    memcpy(mem, &stale, 8);
    uint64_t newv = UINT64_C(0xBEEFBEEFBEEFBEEF);
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = newv;

    ce_result_t res;
    bool ok = run_ce(code, 2, mem, sizeof(mem), KBASE, 0, EXEC_CLAUSE_BPAS, regs, &res);
    EXPECT(ok);
    if (ok) {
        EXPECT_EQ(count_mem_entries(&res), 2);   /* store + 1 load, no extra spec load */
        for (int i = 0; i < count_mem_entries(&res); ++i) {
            instr_trace_entry_t *e = find_mem_entry(&res, i);
            if (e) EXPECT_EQ(e->metadata.speculation_nesting, (uint64_t)0);  /* never speculates */
        }
        instr_trace_entry_t *ld = find_mem_entry(&res, 1);
        if (ld) EXPECT_EQ(ld->metadata.memory_access.before, newv);          /* arch store-forward */
    }
}

/* ---- bpas with two consecutive stores to the same address: windows nest ---- */
static void test_integration_bpas_consecutive_stores(void) {
    /*
     * STR X0,[X29] ; STR X1,[X29] ; LDR X2,[X29].  mem[0]=ORIG, X0=V1, X1=V2.
     * store2's phase A snapshots AFTER store1's phase B already restored ORIG, so the second
     * window's pre-image is ORIG. The two bypass windows therefore NEST (store2 inside store1),
     * and the deepest speculative load reads the ORIGINAL pre-both value — proving "multiple
     * stores" compose into nested bypasses rather than clobbering one snapshot.
     */
    uint32_t code[] = { enc_str_reg(0, 29), enc_str_reg(1, 29), enc_ldr_reg(2, 29) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t orig = UINT64_C(0x0123456789ABCDEF);
    memcpy(mem, &orig, 8);
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = UINT64_C(0x1111111111111111);   /* V1 */
    regs[1] = UINT64_C(0x2222222222222222);   /* V2 */

    /* single-store baseline for the "more exploration" comparison */
    uint32_t one[] = { enc_str_reg(0, 29), enc_ldr_reg(2, 29) };
    ce_result_t one_res, res;
    bool one_ok = run_ce(one, 2, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_BPAS, regs, &one_res);
    bool ok     = run_ce(code, 3, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_BPAS, regs, &res);
    EXPECT(one_ok); EXPECT(ok);
    if (ok && one_ok) {
        int saw_orig = 0; uint64_t max_nest = 0;
        for (int i = 0; i < count_mem_entries(&res); ++i) {
            instr_trace_entry_t *e = find_mem_entry(&res, i);
            if (!e) continue;
            if (e->metadata.speculation_nesting > max_nest)
                max_nest = e->metadata.speculation_nesting;
            if (!e->metadata.memory_access.is_write &&
                e->metadata.speculation_nesting > 0 &&
                e->metadata.memory_access.before == orig)
                saw_orig = 1;
        }
        EXPECT(saw_orig);                 /* deepest window reads the original (both stores bypassed) */
        EXPECT(max_nest >= 2);            /* the two store windows nested */
        EXPECT(count_mem_entries(&res) > count_mem_entries(&one_res));  /* more than one store */
    }
}

/* ---- bpas trace ORDER: a bypassed store is logged AFTER the load that reads its stale value ----
 * Regression for the store-bypass taint bug: taint is derived from trace order, so the bypassed
 * store's entry must sit AFTER the speculative load that read stale (so that load taints the input)
 * and BEFORE the post-window load (which still sees the committed store). Pre-fix the store was
 * logged at execution — before the bypassing load — and wrongly masked it. */
static void test_integration_bpas_store_logged_after_bypassing_load(void) {
    uint32_t code[] = { enc_str_reg(0, 29), enc_ldr_reg(1, 29) };   /* STR X0,[X29]; LDR X1,[X29] */
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t stale = UINT64_C(0xDEADDEADDEADDEAD);
    memcpy(mem, &stale, 8);
    uint64_t newv = UINT64_C(0xBEEFBEEFBEEFBEEF);
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = newv;

    ce_result_t res;
    bool ok = run_ce(code, 2, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_BPAS, regs, &res);
    EXPECT(ok);
    if (!ok) return;

    int store_idx = -1, spec_stale_load_idx = -1, post_load_idx = -1;
    for (int i = 0; i < count_mem_entries(&res); ++i) {
        instr_trace_entry_t *e = find_mem_entry(&res, i);
        if (!e) continue;
        const mem_access_t *ma = &e->metadata.memory_access;
        if (ma->is_write && ma->after == newv && store_idx < 0)
            store_idx = i;
        else if (!ma->is_write && e->metadata.speculation_nesting > 0 && ma->before == stale
                 && spec_stale_load_idx < 0)
            spec_stale_load_idx = i;
        else if (!ma->is_write && e->metadata.speculation_nesting == 0 && ma->before == newv
                 && post_load_idx < 0)
            post_load_idx = i;
    }
    EXPECT(store_idx >= 0);
    EXPECT(spec_stale_load_idx >= 0);
    EXPECT(store_idx > spec_stale_load_idx);   /* store re-emitted AFTER the bypassing load  <-- the fix */
    if (post_load_idx >= 0) EXPECT(store_idx < post_load_idx);   /* ...but before the committed re-read */
}

/* ---- bpas trace ORDER, nested: the deepest stale load precedes BOTH store writes ---- */
static void test_integration_bpas_nested_stores_logged_after_stale_load(void) {
    uint32_t code[] = { enc_str_reg(0, 29), enc_str_reg(1, 29), enc_ldr_reg(2, 29) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t orig = UINT64_C(0x0123456789ABCDEF);
    memcpy(mem, &orig, 8);
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = UINT64_C(0x1111111111111111);
    regs[1] = UINT64_C(0x2222222222222222);

    ce_result_t res;
    bool ok = run_ce(code, 3, mem, sizeof(mem), KBASE, 8, EXEC_CLAUSE_BPAS, regs, &res);
    EXPECT(ok);
    if (!ok) return;

    /* Both bypass windows nest, so the DFS visits: bypass-both (reads ORIG) -> inner window unwinds
     * (S2 committed) -> bypass-only-outer -> outer window unwinds (S1 committed) -> ... . The deferred
     * stores must therefore be logged inner-before-outer, both AFTER the load that read the input. */
    uint64_t v1 = UINT64_C(0x1111111111111111), v2 = UINT64_C(0x2222222222222222);
    int orig_load_idx = -1, first_write_idx = -1, s2_idx = -1, s1_idx = -1;
    for (int i = 0; i < count_mem_entries(&res); ++i) {
        instr_trace_entry_t *e = find_mem_entry(&res, i);
        if (!e) continue;
        const mem_access_t *ma = &e->metadata.memory_access;
        if (ma->is_write && first_write_idx < 0)          first_write_idx = i;
        if (ma->is_write && ma->after == v2 && s2_idx < 0) s2_idx = i;   /* inner store, emitted first */
        if (ma->is_write && ma->after == v1 && s1_idx < 0) s1_idx = i;   /* outer store, emitted second */
        if (!ma->is_write && e->metadata.speculation_nesting > 0 && ma->before == orig
            && orig_load_idx < 0)                          orig_load_idx = i;
    }
    EXPECT(orig_load_idx >= 0);        /* both stores bypassed -> a load reads the pre-both (input) value */
    EXPECT(s2_idx >= 0);
    EXPECT(s1_idx >= 0);
    EXPECT(orig_load_idx < first_write_idx);  /* input read logged before any store masks it  <-- the fix */
    EXPECT(s2_idx < s1_idx);                  /* inner window unwinds first: S2 emitted before S1 (LIFO) */
}

/* ---- cond max_nesting cap: depth 2 reachable at cap 2, never exceeds cap 1 ---- */
static void test_integration_cond_nesting_cap(void) {
    /*
     * Two always-mispredicted branches stack: CBZ X0 forks (nest 1), and inside that window
     * CBZ X1 forks again (nest 2). The cap (spec_nesting() < spec_max_nesting()) must bound this:
     *   max_nesting=2 -> a load at nesting 2 exists; max_nesting=1 -> nothing exceeds nesting 1.
     */
    uint32_t code[] = {
        enc_cbz(0, +8),              /* CBZ X0,+8 : arch taken, spec falls through */
        enc_cbz(1, +8),              /* CBZ X1,+8 : second mispredict, deeper window */
        enc_ldr_reg(2, 29),
        enc_ldr_unsigned(3, 29, 8),
    };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };   /* X0=X1=0 → both CBZ arch-taken, spec not-taken */

    ce_result_t r2, r1;
    bool ok2 = run_ce(code, 4, mem, sizeof(mem), KBASE, 2, EXEC_CLAUSE_COND, regs, &r2);
    bool ok1 = run_ce(code, 4, mem, sizeof(mem), KBASE, 1, EXEC_CLAUSE_COND, regs, &r1);
    EXPECT(ok2); EXPECT(ok1);
    if (ok2) {
        uint64_t mx = 0;
        for (int i = 0; i < count_mem_entries(&r2); ++i) {
            instr_trace_entry_t *e = find_mem_entry(&r2, i);
            if (e && e->metadata.speculation_nesting > mx) mx = e->metadata.speculation_nesting;
        }
        EXPECT_EQ(mx, (uint64_t)2);   /* cap 2 reaches depth 2 */
    }
    if (ok1) {
        for (int i = 0; i < count_mem_entries(&r1); ++i) {
            instr_trace_entry_t *e = find_mem_entry(&r1, i);
            if (e) EXPECT(e->metadata.speculation_nesting <= 1);   /* cap 1 never exceeds 1 */
        }
    }
}

/* ======================================================================== *
 *  Combination / nesting-cap flows: each store and each branch is an
 *  independent 2-way speculation fork, so N forks => up to 2^N flows, and
 *  max_misspred_branch_nesting bounds how many can be active at once.
 * ======================================================================== */

static int saw_load_value(ce_result_t *res, uint64_t val) {
    for (int i = 0; i < res->n_entries; ++i) {
        instr_trace_entry_t *e = &res->entries[i];
        if (e->metadata.has_memory_access && !e->metadata.memory_access.is_write
            && e->metadata.memory_access.before == val) return 1;
    }
    return 0;
}
static int saw_spec_load_value(ce_result_t *res, uint64_t val) {
    for (int i = 0; i < res->n_entries; ++i) {
        instr_trace_entry_t *e = &res->entries[i];
        if (e->metadata.has_memory_access && !e->metadata.memory_access.is_write
            && e->metadata.speculation_nesting > 0
            && e->metadata.memory_access.before == val) return 1;
    }
    return 0;
}
static uint64_t trace_max_nesting(ce_result_t *res) {
    uint64_t m = 0;
    for (int i = 0; i < res->n_entries; ++i)
        if (res->entries[i].metadata.speculation_nesting > m)
            m = res->entries[i].metadata.speculation_nesting;
    return m;
}
/* true if some SPECULATIVE trace entry holds `val` in GPR `reg` (used to catch a bypassed register). */
static int saw_spec_reg_value(ce_result_t *res, int reg, uint64_t val) {
    for (int i = 0; i < res->n_entries; ++i) {
        instr_trace_entry_t *e = &res->entries[i];
        if (e->metadata.speculation_nesting > 0 && e->cpu.gpr[reg] == val) return 1;
    }
    return 0;
}
static int count_loads(ce_result_t *res) {
    int c = 0;
    for (int i = 0; i < res->n_entries; ++i)
        if (res->entries[i].metadata.has_memory_access && !res->entries[i].metadata.memory_access.is_write)
            ++c;
    return c;
}

#define ORIG  UINT64_C(0xA0A0A0A0A0A0A0A0)
#define SV1   UINT64_C(0x1111111111111111)
#define SV2   UINT64_C(0x2222222222222222)
#define SV3   UINT64_C(0x3333333333333333)

/* run a bpas/cond gadget at a given nesting cap; mem[0]=ORIG */
static bool run_gadget(uint32_t *code, int nw, uint64_t cap, uint64_t clauses,
                       uint64_t *regs, ce_result_t *res) {
    uint8_t mem[MEM_SIZE]; memset(mem, 0, sizeof(mem));
    memcpy(mem, &(uint64_t){ORIG}, 8);
    return run_ce(code, nw, mem, sizeof(mem), KBASE, cap, clauses, regs, res);
}

/* Same, but with an explicit per-window instruction cap (max_instr). */
static bool run_gadget_capped(uint32_t *code, int nw, uint64_t nest, uint64_t max_instr,
                              uint64_t clauses, uint64_t *regs, ce_result_t *res) {
    uint8_t mem[MEM_SIZE]; memset(mem, 0, sizeof(mem));
    memcpy(mem, &(uint64_t){ORIG}, 8);
    return run_ce_full(code, nw, mem, sizeof(mem), KBASE, nest, max_instr, clauses,
                       BRANCH_PREDICTOR_NONE, regs, NULL, 0, res);
}

/* Regression: the per-window instruction cap must count each window's OWN path, not the global
 * instruction counter. A cond misprediction interposed between two bypassed stores must not age out
 * the enclosing bpas window: both stores stay bypassed and the backbone load reads doubly-stale ORIG.
 * With the old global counter the long wrong-path excursion prematurely closed the outer window, so
 * the second store committed and the load read SV1 — a lost store-bypass flow (non-monotone cond+bpas
 * composition, the root cause of a false violation). */
static void test_integration_cond_bpas_outer_window_survives_excursion(void) {
    uint32_t code[] = {
        enc_str_reg(0, 29),   /* STR x0 -> SV1 (bypassed) */
        enc_cbz(3, +12),      /* CBZ x3: x3=1 arch falls through; cond mispredicts TAKEN -> idx4 (NOPs) */
        enc_str_reg(1, 29),   /* STR x1 -> SV2 (bypassed on the backbone iff the window survives) */
        enc_ldr_reg(4, 29),   /* LDR x4: backbone load — reads doubly-stale ORIG at nest>=2 */
        enc_nop(), enc_nop(), enc_nop(), enc_nop(), enc_nop(),   /* long mispredicted excursion */
    };
    uint64_t regs[REGS_COUNT] = {0}; regs[0]=SV1; regs[1]=SV2; regs[3]=1;
    ce_result_t r;
    bool ok = run_gadget_capped(code, 9, 8, 3, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_spec_load_value(&r, ORIG));   /* both stores bypassed -> load reads ORIG speculatively */
        EXPECT(trace_max_nesting(&r) >= 2);      /* the enclosing window survived to nest the 2nd bypass */
        EXPECT(saw_load_value(&r, SV2));         /* architecturally both stores applied */
    }
}

/* ======================================================================== *
 *  EXEC_CLAUSE_BARRIER: a fencing barrier ends the speculation it stops.
 * ======================================================================== */

#define STALE UINT64_C(0xDEADDEADDEADDEAD)
#define NEWV  UINT64_C(0xBEEFBEEFBEEFBEEF)

/* Run `code` with mem[0]=STALE, X0=NEWV, cap 8. */
static bool run_bypass(uint32_t *code, int nw, uint64_t clauses, ce_result_t *res) {
    uint8_t mem[MEM_SIZE]; memset(mem, 0, sizeof(mem));
    memcpy(mem, &(uint64_t){STALE}, 8);
    uint64_t regs[REGS_COUNT] = { 0 }; regs[0] = NEWV;
    return run_ce(code, nw, mem, sizeof(mem), KBASE, 8, clauses, regs, res);
}

static int count_spec_load_value(ce_result_t *res, uint64_t val) {
    int c = 0;
    for (int i = 0; i < res->n_entries; ++i) {
        instr_trace_entry_t *e = &res->entries[i];
        if (e->metadata.has_memory_access && !e->metadata.memory_access.is_write
            && e->metadata.speculation_nesting > 0
            && e->metadata.memory_access.before == val) ++c;
    }
    return c;
}

/* ---- SSBB commits the store to memory but does NOT squash speculation (store ADJACENT to barrier) -- */
static void test_integration_barrier_ssbb_commits_no_squash(void) {
    /* STR X0;SSBB;LDR X1 (all [X29]).  bpas alone ignores SSBB and the load speculatively reads STALE.
     * bpas+barrier commits the store to memory at the SSBB, so a load PAST the barrier sees the
     * committed NEWV (no forwarding across the barrier) — but speculation keeps running (not squashed). */
    uint32_t code[] = { enc_str_reg(0, 29), enc_ssbb(), enc_ldr_reg(1, 29) };
    ce_result_t bpas, barr;
    bool bpas_ok = run_bypass(code, 3, EXEC_CLAUSE_BPAS, &bpas);
    bool barr_ok = run_bypass(code, 3, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, &barr);
    EXPECT(bpas_ok); EXPECT(barr_ok);
    if (bpas_ok) EXPECT(saw_spec_load_value(&bpas, STALE));           /* SSBB ignored -> bypass */
    if (barr_ok) {
        EXPECT(!saw_spec_load_value(&barr, STALE));                  /* load past barrier: no stale... */
        EXPECT(saw_load_value(&barr, NEWV));                         /* ...it sees the committed store */
        EXPECT(trace_max_nesting(&barr) >= (uint64_t)1);            /* speculation NOT squashed */
    }
}

/* ---- same, when the window is already open before the barrier (NOP between store and SSBB) ---- */
static void test_integration_barrier_ssbb_commits_no_squash_nonadjacent(void) {
    /* A NOP between the store and SSBB means bpas phase B has already pushed the window before the
     * barrier is reached (the other code path than the adjacent case above). */
    uint32_t code[] = { enc_str_reg(0, 29), enc_nop(), enc_ssbb(), enc_ldr_reg(1, 29) };
    ce_result_t barr;
    bool ok = run_bypass(code, 4, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, &barr);
    EXPECT(ok);
    if (ok) {
        EXPECT(!saw_spec_load_value(&barr, STALE));                  /* load past barrier sees committed */
        EXPECT(saw_load_value(&barr, NEWV));
        EXPECT(trace_max_nesting(&barr) >= (uint64_t)1);            /* speculation NOT squashed */
    }
}

/* ---- DMB is ordering-only: it must NOT cut the store-bypass window ---- */
static void test_integration_barrier_dmb_no_cut(void) {
    uint32_t code[] = { enc_str_reg(0, 29), enc_dmb_sy(), enc_ldr_reg(1, 29) };
    ce_result_t barr;
    bool ok = run_bypass(code, 3, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, &barr);
    EXPECT(ok);
    if (ok) EXPECT(saw_spec_load_value(&barr, STALE));   /* DMB fences nothing -> bypass survives */
}

/* ---- a store-bypass barrier with no pending store is a no-op ---- */
static void test_integration_barrier_ssbb_no_store(void) {
    uint32_t code[] = { enc_ssbb(), enc_ldr_reg(1, 29) };
    uint8_t mem[MEM_SIZE]; memset(mem, 0, sizeof(mem));
    memcpy(mem, &(uint64_t){NEWV}, 8);
    uint64_t regs[REGS_COUNT] = { 0 };
    ce_result_t res;
    bool ok = run_ce(code, 2, mem, sizeof(mem), KBASE, 8,
                     EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, regs, &res);
    EXPECT(ok);
    if (ok) {
        EXPECT_EQ(count_mem_entries(&res), 1);              /* just the architectural load */
        EXPECT_EQ(trace_max_nesting(&res), (uint64_t)0);
    }
}

/* ---- an SSBB commits MULTIPLE open store-bypass windows to memory (no squash) ---- */
static void test_integration_barrier_ssbb_nested_commits_no_squash(void) {
    /* STR X0(SV1);STR X1(SV2);SSBB;LDR X2 (all [X29]).  At the SSBB every open bypassed store is
     * committed to live memory (oldest->newest, last write wins), so a load past the barrier can see
     * the fully-committed SV2, memory is committed (no fully-stale ORIG read past the barrier), and
     * speculation is not squashed. (Partial-bypass flows over-approximate for nested stores, which is
     * conservative for the fuzzer — it predicts more leakage, never less.) */
    uint32_t code[] = { enc_str_reg(0, 29), enc_str_reg(1, 29), enc_ssbb(), enc_ldr_reg(2, 29) };
    uint64_t regs[REGS_COUNT] = { 0 }; regs[0] = SV1; regs[1] = SV2;

    ce_result_t barr;
    bool ok = run_gadget(code, 4, 8, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, regs, &barr);
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_load_value(&barr, SV2));                /* fully-committed value visible past barrier */
        EXPECT(!saw_spec_load_value(&barr, ORIG));         /* memory committed -> no fully-stale read */
        EXPECT(trace_max_nesting(&barr) >= (uint64_t)1);  /* speculation NOT squashed */
    }
}

/* ---- THE LEAK: a value bypassed into a REGISTER before the SSBB stays stale AFTER it ---- */
static void test_integration_barrier_ssbb_register_bypass_survives(void) {
    /* STR X0;LDR X1;SSBB;LDR X2 (all [X29]).  LDR X1 bypasses the store, reading STALE into X1.  The
     * SSBB commits memory (a re-load past it reads NEWV) but must NOT revert X1: a value already
     * bypassed into a register survives the barrier and could be transmitted after it (the N3 leak). */
    uint32_t code[] = { enc_str_reg(0, 29), enc_ldr_reg(1, 29), enc_ssbb(), enc_ldr_reg(2, 29) };
    ce_result_t bpas, barr;
    bool bpas_ok = run_bypass(code, 4, EXEC_CLAUSE_BPAS, &bpas);
    bool barr_ok = run_bypass(code, 4, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, &barr);
    EXPECT(bpas_ok); EXPECT(barr_ok);
    if (bpas_ok) EXPECT(saw_spec_reg_value(&bpas, 1, STALE));   /* bpas alone: X1 bypassed to STALE */
    if (barr_ok) {
        EXPECT(saw_spec_reg_value(&barr, 1, STALE));           /* X1 keeps the bypassed value past the SSBB */
        EXPECT(saw_load_value(&barr, NEWV));                   /* but memory is committed: LDR X2 = NEWV */
        /* (LDR X1, before the barrier, does read STALE — that is the bypass itself.) */
    }
}

/* ---- PSSBB behaves like SSBB: commit memory, do not squash speculation ---- */
static void test_integration_barrier_pssbb_commits_no_squash(void) {
    uint32_t code[] = { enc_str_reg(0, 29), enc_pssbb(), enc_ldr_reg(1, 29) };
    ce_result_t barr;
    bool ok = run_bypass(code, 3, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, &barr);
    EXPECT(ok);
    if (ok) {
        EXPECT(!saw_spec_load_value(&barr, STALE));
        EXPECT(saw_load_value(&barr, NEWV));
        EXPECT(trace_max_nesting(&barr) >= (uint64_t)1);      /* speculation NOT squashed */
    }
}

/* ---- ISB (pipeline flush) is stronger: it DOES squash the store-bypass window (unlike SSBB) ---- */
static void test_integration_barrier_isb_squashes_bpas(void) {
    uint32_t code[] = { enc_str_reg(0, 29), enc_isb(), enc_ldr_reg(1, 29) };
    ce_result_t barr;
    bool ok = run_bypass(code, 3, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, &barr);
    EXPECT(ok);
    if (ok) {
        EXPECT(!saw_spec_load_value(&barr, STALE));
        EXPECT(saw_load_value(&barr, NEWV));
        EXPECT_EQ(trace_max_nesting(&barr), (uint64_t)0);     /* pipeline flush: speculation squashed */
    }
}

/* ---- a control barrier with no open misprediction is a no-op (no over-cut) ---- */
static void test_integration_barrier_sb_no_speculation(void) {
    /* SB with nothing speculating must not disturb the architectural trace. */
    uint32_t code[] = { enc_ldr_reg(0, 29), enc_sb(), enc_ldr_unsigned(1, 29, 8) };
    uint8_t mem[MEM_SIZE]; memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0x1111111111111111), v8 = UINT64_C(0x2222222222222222);
    memcpy(mem, &v0, 8); memcpy(mem + 8, &v8, 8);
    uint64_t regs[REGS_COUNT] = { 0 };
    ce_result_t res;
    bool ok = run_ce(code, 3, mem, sizeof(mem), KBASE, 4,
                     EXEC_CLAUSE_COND | EXEC_CLAUSE_BARRIER, regs, &res);
    EXPECT(ok);
    if (ok) {
        EXPECT_EQ(count_mem_entries(&res), 2);              /* two architectural loads, nothing spec */
        EXPECT_EQ(trace_max_nesting(&res), (uint64_t)0);
        EXPECT(saw_load_value(&res, v0)); EXPECT(saw_load_value(&res, v8));
    }
}

/* ---- cond+bpas+barrier: SSBB cuts the bypass but leaves the branch mispredict intact ---- */
static void test_integration_barrier_ssbb_spares_enclosing_branch(void) {
    /* CBNZ X0,+16 (X0=0 -> arch NOT-taken; ALWAYS_MISPREDICT -> spec TAKEN into the block):
     *   target: STR X1;SSBB;LDR X2  (all [X29]).  On the speculative (taken) path the store's
     * bypass window opens and is cut by SSBB, so LDR X2 reads the committed store (NEWV) rather
     * than STALE — yet the enclosing branch misprediction (nesting>=1) is NOT unwound by the SSBB. */
    uint32_t code[] = {
        enc_cbnz(0, +16),            /* spec-taken -> index 4 (target) */
        enc_ldr_reg(3, 29),          /* arch fall-through load (X0==0 not-taken) */
        enc_nop(), enc_nop(),
        enc_str_reg(1, 29),          /* target: STR X1  (spec) */
        enc_ssbb(),
        enc_ldr_reg(2, 29),          /* LDR X2 — must see committed NEWV, at nesting>=1 */
    };
    uint8_t mem[MEM_SIZE]; memset(mem, 0, sizeof(mem));
    memcpy(mem, &(uint64_t){STALE}, 8);
    uint64_t regs[REGS_COUNT] = { 0 }; regs[1] = NEWV;   /* X0=0, X1=NEWV */
    ce_result_t res;
    bool ok = run_ce(code, 7, mem, sizeof(mem), KBASE, 4,
                     EXEC_CLAUSE_COND | EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, regs, &res);
    EXPECT(ok);
    if (ok) {
        EXPECT(!saw_spec_load_value(&res, STALE));       /* bypass cut by SSBB */
        int saw_spec_newv = 0;                           /* committed store read WHILE speculating */
        for (int i = 0; i < count_mem_entries(&res); ++i) {
            instr_trace_entry_t *e = find_mem_entry(&res, i);
            if (e && !e->metadata.memory_access.is_write &&
                e->metadata.speculation_nesting > 0 &&
                e->metadata.memory_access.before == NEWV) saw_spec_newv = 1;
        }
        EXPECT(saw_spec_newv);                           /* branch window still active past the SSBB */
    }
}

/* ---- adjacency: a store AFTER the barrier is NOT fenced by it ---- */
static void test_integration_barrier_store_after_not_fenced(void) {
    /* SSBB;STR X0;LDR X1.  The store follows the barrier, so SSBB does not order it; the bypass
     * window opens and LDR X1 speculatively reads STALE. */
    uint32_t code[] = { enc_ssbb(), enc_str_reg(0, 29), enc_ldr_reg(1, 29) };
    ce_result_t res;
    bool ok = run_bypass(code, 3, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, &res);
    EXPECT(ok);
    if (ok) EXPECT(saw_spec_load_value(&res, STALE));   /* later store still bypassable */
}

/* ---- adjacency: a load BEFORE the barrier keeps its (legit) pre-barrier bypass ---- */
static void test_integration_barrier_load_before_barrier(void) {
    /* STR X0;LDR X1;SSBB;LDR X2 (all [X29]).  The pre-barrier load may bypass the store (reads
     * STALE speculatively); the post-barrier load may not.  So bpas+barrier yields exactly ONE
     * speculative STALE read, vs TWO when the SSBB is ignored. */
    uint32_t code[] = { enc_str_reg(0, 29), enc_ldr_reg(1, 29), enc_ssbb(), enc_ldr_reg(2, 29) };
    ce_result_t bpas, barr;
    bool bpas_ok = run_bypass(code, 4, EXEC_CLAUSE_BPAS, &bpas);
    bool barr_ok = run_bypass(code, 4, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, &barr);
    EXPECT(bpas_ok); EXPECT(barr_ok);
    if (bpas_ok) EXPECT_EQ(count_spec_load_value(&bpas, STALE), 2);   /* both loads bypass */
    if (barr_ok) {
        EXPECT_EQ(count_spec_load_value(&barr, STALE), 1);           /* only the pre-barrier load */
        EXPECT(saw_load_value(&barr, NEWV));                         /* post-barrier load committed */
    }
}

/* ---- adjacency: load BEFORE the barrier, store AFTER it — later store still bypassable ---- */
static void test_integration_barrier_load_before_store_after(void) {
    /* LDR X1;SSBB;STR X0;LDR X2 (all [X29]).  The store is AFTER the barrier, so SSBB does not
     * order it: LDR X2 may still speculatively bypass it and read STALE.  (LDR X1 just reads STALE
     * architecturally — no earlier store to bypass.) */
    uint32_t code[] = { enc_ldr_reg(1, 29), enc_ssbb(), enc_str_reg(0, 29), enc_ldr_reg(2, 29) };
    ce_result_t res;
    bool ok = run_bypass(code, 4, EXEC_CLAUSE_BPAS | EXEC_CLAUSE_BARRIER, &res);
    EXPECT(ok);
    if (ok) EXPECT(saw_spec_load_value(&res, STALE));   /* post-barrier store still bypassable */
}

/* ---- branch-speculation barriers: ISB and DSB SY cut a mispredict; a narrower DSB ISH does not - */
static void test_integration_barrier_control_fence_variants(void) {
    uint8_t mem[MEM_SIZE]; memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0x1111111111111111);
    memcpy(mem, &v0, 8);
    uint64_t regs[REGS_COUNT] = { 0 };   /* X0=0 -> CBZ arch-taken; ALWAYS_MISPREDICT falls through */

    struct { uint32_t barrier; int cuts; } cases[] = {
        { enc_isb(),     1 },
        { enc_dsb_sy(),  1 },
        { enc_dsb_ish(), 0 },
    };
    for (unsigned k = 0; k < sizeof(cases) / sizeof(cases[0]); ++k) {
        uint32_t code[] = { enc_cbz(0, +12), cases[k].barrier, enc_ldr_reg(1, 29),
                            enc_ldr_unsigned(2, 29, 8) };
        ce_result_t r;
        bool ok = run_ce(code, 4, mem, sizeof(mem), KBASE, 1,
                         EXEC_CLAUSE_COND | EXEC_CLAUSE_BARRIER, regs, &r);
        EXPECT(ok);
        if (!ok) continue;
        if (cases[k].cuts) EXPECT(!saw_spec_load_value(&r, v0));   /* mispredict cut at the barrier */
        else               EXPECT(saw_spec_load_value(&r, v0));    /* narrower DSB: unsound to cut  */
    }
}

/* ---- control-fence granularity: SSBB does NOT cut a branch mispredict, SB does ---- */
static void test_integration_barrier_control_fence_granularity(void) {
    /* CBZ X0,+12 (arch TAKEN, X0=0) jumps past the speculative region; ALWAYS_MISPREDICT falls
     * through to <barrier>;LDR X1 speculatively.  SSBB does not fence control -> the spec load
     * survives; SB does -> it is cut. */
    uint8_t mem[MEM_SIZE]; memset(mem, 0, sizeof(mem));
    uint64_t v0 = UINT64_C(0x1111111111111111);
    memcpy(mem, &v0, 8);
    uint64_t regs[REGS_COUNT] = { 0 };   /* X0=0 -> CBZ architecturally taken */

    uint32_t code_ssbb[] = { enc_cbz(0, +12), enc_ssbb(), enc_ldr_reg(1, 29), enc_ldr_unsigned(2, 29, 8) };
    uint32_t code_sb[]   = { enc_cbz(0, +12), enc_sb(),   enc_ldr_reg(1, 29), enc_ldr_unsigned(2, 29, 8) };

    ce_result_t r_ssbb, r_sb;
    bool ok_ssbb = run_ce(code_ssbb, 4, mem, sizeof(mem), KBASE, 1,
                          EXEC_CLAUSE_COND | EXEC_CLAUSE_BARRIER, regs, &r_ssbb);
    bool ok_sb   = run_ce(code_sb,   4, mem, sizeof(mem), KBASE, 1,
                          EXEC_CLAUSE_COND | EXEC_CLAUSE_BARRIER, regs, &r_sb);
    EXPECT(ok_ssbb); EXPECT(ok_sb);
    if (ok_ssbb) EXPECT(saw_spec_load_value(&r_ssbb, v0));    /* SSBB does not fence control */
    if (ok_sb)   EXPECT(!saw_spec_load_value(&r_sb, v0));     /* SB cuts the mispredicted path */
}

/* ---- three stores, same address: full 2^N value set, bounded by the cap ---- */
static void test_integration_bpas_three_stores_value_set(void) {
    /* STR x0;STR x1;STR x2;LDR x3 (all [x29]).  The end load can read the value of the most
     * recent NON-bypassed store, or ORIG if a suffix of stores is bypassed:
     *   read SV3=none bypassed (nest0); SV2=bypass S3 (nest1); SV1=bypass S3,S2 (nest2);
     *   ORIG=bypass S3,S2,S1 (nest3).  The cap limits how deep the suffix-bypass can go. */
    uint32_t code[] = { enc_str_reg(0,29), enc_str_reg(1,29), enc_str_reg(2,29), enc_ldr_reg(3,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[1]=SV2; regs[2]=SV3;
    ce_result_t r;

    bool ok = run_gadget(code, 4, 3, EXEC_CLAUSE_BPAS, regs, &r);   /* cap 3: all four values */
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_load_value(&r, SV3)); EXPECT(saw_load_value(&r, SV2));
        EXPECT(saw_load_value(&r, SV1)); EXPECT(saw_load_value(&r, ORIG));
        EXPECT_EQ(trace_max_nesting(&r), (uint64_t)3);
    }
    ok = run_gadget(code, 4, 2, EXEC_CLAUSE_BPAS, regs, &r);        /* cap 2: ORIG unreachable */
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_load_value(&r, SV3)); EXPECT(saw_load_value(&r, SV2));
        EXPECT(saw_load_value(&r, SV1)); EXPECT(!saw_load_value(&r, ORIG));
        EXPECT_EQ(trace_max_nesting(&r), (uint64_t)2);
    }
    ok = run_gadget(code, 4, 1, EXEC_CLAUSE_BPAS, regs, &r);        /* cap 1: only SV3,SV2 */
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_load_value(&r, SV3)); EXPECT(saw_load_value(&r, SV2));
        EXPECT(!saw_load_value(&r, SV1)); EXPECT(!saw_load_value(&r, ORIG));
    }
    ok = run_gadget(code, 4, 0, EXEC_CLAUSE_BPAS, regs, &r);        /* cap 0: architectural only */
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_load_value(&r, SV3)); EXPECT(!saw_load_value(&r, SV2));
        EXPECT_EQ(trace_max_nesting(&r), (uint64_t)0);
    }
}

/* ---- branch BEFORE three stores (cond|bpas) ---- */
static void test_integration_cond_bpas_branch_before_stores(void) {
    /* CBZ x3,+? ; STR;STR;STR ; LDR.  x3!=0 -> arch falls through to the stores; the stores fork
     * (bypass/apply) on the architectural path, and the branch adds another fork. */
    uint32_t code[] = { enc_cbz(3,+16), enc_str_reg(0,29), enc_str_reg(1,29), enc_str_reg(2,29),
                        enc_ldr_reg(4,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[1]=SV2; regs[2]=SV3; regs[3]=1;
    ce_result_t r;
    bool ok = run_gadget(code, 5, 8, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_spec_load_value(&r, ORIG));   /* a suffix-of-stores bypass reaches ORIG */
        EXPECT(saw_load_value(&r, SV3));         /* architectural store value */
        EXPECT(trace_max_nesting(&r) >= 3);      /* three stores can nest */
    }
    /* cap bounds the combined depth */
    ok = run_gadget(code, 5, 1, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);
    if (ok) EXPECT(trace_max_nesting(&r) <= 1);
}

/* ---- branch AFTER the stores, before the load ---- */
static void test_integration_cond_bpas_branch_after_stores(void) {
    /* STR;STR; CBZ x3,+4 ; LDR.  The branch sits between the stores and the load. */
    uint32_t code[] = { enc_str_reg(0,29), enc_str_reg(1,29), enc_cbz(3,+8), enc_ldr_reg(4,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[1]=SV2; regs[3]=0;
    ce_result_t r;
    bool ok = run_gadget(code, 4, 8, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_spec_load_value(&r, ORIG));   /* both stores bypassed */
        EXPECT(saw_spec_load_value(&r, SV1));    /* only the 2nd store bypassed */
        EXPECT(saw_load_value(&r, SV2));         /* architectural */
    }
}

/* ---- branch in the MIDDLE of the stores ---- */
static void test_integration_cond_bpas_branch_in_middle(void) {
    /* STR x0 ; CBZ x3,+4 ; STR x1 ; LDR.  Branch interleaved between two stores. */
    uint32_t code[] = { enc_str_reg(0,29), enc_cbz(3,+8), enc_str_reg(1,29), enc_ldr_reg(4,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[1]=SV2; regs[3]=1;   /* arch falls through */
    ce_result_t r;
    bool ok = run_gadget(code, 4, 8, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_spec_load_value(&r, ORIG));   /* both stores bypassed somewhere */
        EXPECT(saw_load_value(&r, SV2));         /* architectural: both applied */
        EXPECT(trace_max_nesting(&r) >= 2);
    }
}

/* ---- store THEN branch ---- */
static void test_integration_cond_bpas_store_then_branch(void) {
    /* STR x0 ; CBZ x3,+4 ; LDR.  Store immediately followed by a branch. */
    uint32_t code[] = { enc_str_reg(0,29), enc_cbz(3,+8), enc_ldr_reg(4,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[3]=1;
    ce_result_t r;
    bool ok = run_gadget(code, 3, 8, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_spec_load_value(&r, ORIG));   /* the store was bypassed */
        EXPECT(saw_load_value(&r, SV1));         /* architectural store value */
    }
}

/* ---- branch THEN store ---- */
static void test_integration_cond_bpas_branch_then_store_flow(void) {
    /* CBZ x3,+4 ; STR x0 ; LDR.  Branch immediately followed by a store. */
    uint32_t code[] = { enc_cbz(3,+8), enc_str_reg(0,29), enc_ldr_reg(4,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[3]=1;   /* arch falls through to the store */
    ce_result_t r;
    bool ok = run_gadget(code, 3, 8, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_spec_load_value(&r, ORIG));   /* store bypassed */
        EXPECT(saw_load_value(&r, SV1));         /* architectural */
    }
}

/* ---- two branches in the middle of two stores ---- */
static void test_integration_cond_bpas_branches_in_middle(void) {
    uint32_t code[] = { enc_str_reg(0,29), enc_cbz(3,+4), enc_cbz(4,+4),
                        enc_str_reg(1,29), enc_ldr_reg(5,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[1]=SV2; regs[3]=1; regs[4]=1;
    ce_result_t r;
    bool ok = run_gadget(code, 5, 8, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_load_value(&r, SV2));         /* architectural */
        EXPECT(saw_spec_load_value(&r, ORIG));   /* both stores bypassed on some path */
    }
}

/* ---- depth cap with only branches ---- */
static void test_integration_cond_only_nesting_cap(void) {
    /* three conditional branches: cap=3 reaches depth 3, cap=1 never exceeds 1, cap=0 no spec. */
    uint32_t code[] = { enc_cbz(0,+4), enc_cbz(1,+4), enc_cbz(2,+4), enc_ldr_reg(3,29) };
    uint64_t regs[REGS_COUNT]={0};   /* x0=x1=x2=0 */
    ce_result_t r;
    bool ok = run_gadget(code, 4, 3, EXEC_CLAUSE_COND, regs, &r);
    EXPECT(ok); if (ok) EXPECT_EQ(trace_max_nesting(&r), (uint64_t)3);
    ok = run_gadget(code, 4, 1, EXEC_CLAUSE_COND, regs, &r);
    EXPECT(ok); if (ok) EXPECT(trace_max_nesting(&r) <= 1);
    ok = run_gadget(code, 4, 0, EXEC_CLAUSE_COND, regs, &r);
    EXPECT(ok); if (ok) EXPECT_EQ(trace_max_nesting(&r), (uint64_t)0);
}

/* ---- depth cap with interleaved stores and branches ---- */
static void test_integration_cond_bpas_interleaved_nesting_cap(void) {
    /* STR ; CBZ ; STR ; CBZ ; LDR — caps must bound the combined fork depth. */
    uint32_t code[] = { enc_str_reg(0,29), enc_cbz(3,+4), enc_str_reg(1,29), enc_cbz(4,+4),
                        enc_ldr_reg(5,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[1]=SV2; regs[3]=1; regs[4]=1;
    ce_result_t r;
    for (uint64_t cap = 0; cap <= 4; ++cap) {
        bool ok = run_gadget(code, 5, cap, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
        EXPECT(ok);
        if (ok) EXPECT(trace_max_nesting(&r) <= cap);   /* cap is always respected */
    }
}

/* ---- a store fork and a branch fork are the SAME 2-way speculation (symmetry) ---- */
static void test_integration_forks_store_vs_branch_symmetry(void) {
    /* Three stores and three branches must reach the end load on the identical set of leaves:
     * count = sum_{i=0}^{cap} C(3,i) = {1,4,7,8} for caps {0,1,2,3}.  Same engine, same shape. */
    uint32_t stores[]   = { enc_str_reg(0,29), enc_str_reg(1,29), enc_str_reg(2,29), enc_ldr_reg(3,29) };
    uint32_t branches[] = { enc_cbz(0,+4), enc_cbz(1,+4), enc_cbz(2,+4), enc_ldr_reg(3,29) };
    uint64_t sregs[REGS_COUNT]={0}; sregs[0]=SV1; sregs[1]=SV2; sregs[2]=SV3;
    uint64_t bregs[REGS_COUNT]={0};   /* x0=x1=x2=0 -> CBZ taken architecturally */
    int expected[] = { 1, 4, 7, 8 };
    for (uint64_t cap = 0; cap <= 3; ++cap) {
        ce_result_t rs, rb;
        bool oks = run_gadget(stores,   4, cap, EXEC_CLAUSE_BPAS, sregs, &rs);
        bool okb = run_gadget(branches, 4, cap, EXEC_CLAUSE_COND, bregs, &rb);
        EXPECT(oks); EXPECT(okb);
        if (oks) EXPECT_EQ(count_loads(&rs), expected[cap]);   /* stores: 2^N leaves, cap-bounded */
        if (okb) EXPECT_EQ(count_loads(&rb), expected[cap]);   /* branches: identical */
    }
}

/* ---- a store-pair (STP) is ONE instruction => ONE fork covering both elements ---- */
static void test_integration_bpas_store_pair_single_fork(void) {
    /* STP x0,x1,[x29] ; LDR x2,[x29].  The pair bypasses as a unit, so the load forks to depth 1
     * (not 2): it reads ORIG (pair bypassed) or SV1 (pair applied -> element 0 = x0). */
    uint32_t code[] = { enc_stp(0,1,29), enc_ldr_reg(2,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[1]=SV2;
    ce_result_t r;
    bool ok = run_gadget(code, 2, 8, EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);
    if (ok) {
        EXPECT(saw_load_value(&r, ORIG));               /* pair bypassed */
        EXPECT(saw_load_value(&r, SV1));                /* pair applied */
        EXPECT_EQ(trace_max_nesting(&r), (uint64_t)1);  /* a pair is a SINGLE fork */
    }
}

/* ---- registry order: a store-bypass window must ENCLOSE a following branch's window ---- */
static void test_integration_cond_bpas_store_encloses_branch(void) {
    /* STR x0 ; CBZ x3,+8 ; LDR x1.  On the CBZ (the instruction after the store) bpas is dispatched
     * BEFORE cond, so the store-bypass window opens first (nest 1) and the branch forks INSIDE it
     * (nest 2).  x3=0 => arch takes the branch and skips the load, so the load is reached only on
     * the branch's speculative path, at nesting 2, where it must read the STALE (bypassed) memory.
     * Reversing the order would nest the branch window OUTSIDE the store window and corrupt the
     * LIFO checkpoint unwind (the engine would trap). */
    uint32_t code[] = { enc_str_reg(0,29), enc_cbz(3,+8), enc_ldr_reg(1,29) };
    uint64_t regs[REGS_COUNT]={0}; regs[0]=SV1; regs[3]=0;
    ce_result_t r;
    bool ok = run_gadget(code, 3, 8, EXEC_CLAUSE_COND|EXEC_CLAUSE_BPAS, regs, &r);
    EXPECT(ok);   /* must not trap the LIFO checkpoint assertion */
    if (ok) {
        int found = 0;
        for (int i = 0; i < r.n_entries; ++i) {
            instr_trace_entry_t *e = &r.entries[i];
            if (e->metadata.has_memory_access && !e->metadata.memory_access.is_write
                && e->metadata.speculation_nesting == 2
                && e->metadata.memory_access.before == ORIG) found = 1;
        }
        EXPECT(found);   /* branch (nest 2) nested inside the store-bypass window (nest 1) reads stale */
    }
}

/* ---- BPU resolves its predictor from the input (not a hardcoded injection) ---- */
static void test_integration_bpu_input_selected_predictor(void) {
    /* build_ce_input selects Neoverse-N3 for a BPU contract; the run must succeed, proving the
     * predictor was resolved from the input. (BPU with no predictor would trap in the CE.) */
    uint32_t code[] = { enc_cbz(0, +8), enc_ldr_reg(1, 29), enc_ldr_unsigned(2, 29, 8) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    ce_result_t res;
    bool ok = run_ce_simple(code, 3, mem, sizeof(mem), KBASE, 4, EXEC_CLAUSE_BPU, &res);
    EXPECT(ok);
    if (ok) EXPECT(count_mem_entries(&res) >= 1);   /* at least the architectural load */
}

/* ---- BPU with no predictor selected is rejected by the CE ---- */
static void test_integration_bpu_without_predictor_traps(void) {
    uint32_t code[] = { enc_cbz(0, +8), enc_ldr_reg(1, 29), enc_ldr_unsigned(2, 29, 8) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    ce_result_t res;
    bool ok = run_ce_full(code, 3, mem, sizeof(mem), KBASE, 4, 0,
                          EXEC_CLAUSE_BPU, BRANCH_PREDICTOR_NONE, NULL, NULL, 0, &res);
    EXPECT(!ok);   /* CE traps: BPU enabled with no predictor */
}

/* ---- Emulated MTE data-processing instructions ------------------------- *
 * The CE software-emulates the MTE address/flag instructions so test cases can
 * use them on a non-MTE CPU. These check that emulation produces the
 * ARM-defined result. Expected values follow the ARM ARM (DDI0487). */

/* SUBP  X<rd>, X<rn>, X<rm>  (tagged pointer difference, no flags) */
static uint32_t enc_subp(int rd, int rn, int rm) {
    return 0x9AC00000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd;
}
/* SUBPS X<rd>, X<rn>, X<rm>  (tagged pointer difference, sets NZCV) */
static uint32_t enc_subps(int rd, int rn, int rm) {
    return 0xBAC00000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd;
}

/* Same address, different tags → tag-stripped operands equal → diff 0, Z=1, C=1. */
static void test_integration_mte_subps_equal_tagged_pointers(void) {
    uint32_t code[] = {
        enc_subps(0, 1, 2),   /* SUBPS X0, X1, X2 */
        enc_ldr_reg(3, 29),   /* LDR  X3, [X29]   (carries cpu_state after SUBPS) */
    };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = UINT64_C(0x0A00000000001000);   /* tag 0xA, address 0x1000 */
    regs[2] = UINT64_C(0x0500000000001000);   /* tag 0x5, address 0x1000 */

    ce_result_t res;
    if (!run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }
    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    /* result = 0 (same address, tags stripped) */
    EXPECT_EQ(e->cpu.gpr[0], (uint64_t)0);
    /* NZCV: Z=1, C=1 (op1>=op2), N=0, V=0  →  0x60000000 */
    EXPECT_EQ(e->cpu.nzcv, UINT64_C(0x60000000));
}

/* SUBPS where op1 < op2 → negative result; N=1, C=0 (borrow). */
static void test_integration_mte_subps_negative(void) {
    uint32_t code[] = {
        enc_subps(0, 1, 2),
        enc_ldr_reg(3, 29),
    };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = UINT64_C(0x1000);
    regs[2] = UINT64_C(0x2000);

    ce_result_t res;
    if (!run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }
    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    /* result = 0x1000 - 0x2000 = -0x1000, sign-extended to 64 bits */
    EXPECT_EQ(e->cpu.gpr[0], UINT64_C(0xFFFFFFFFFFFFF000));
    /* NZCV: N=1, Z=0, C=0, V=0  →  0x80000000 */
    EXPECT_EQ(e->cpu.nzcv, UINT64_C(0x80000000));
}

/* SUBP (no S) computes the difference but must leave NZCV untouched. */
static void test_integration_mte_subp_preserves_nzcv(void) {
    uint32_t code[] = {
        enc_subp(0, 1, 2),
        enc_ldr_reg(3, 29),
    };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));

    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = UINT64_C(0x2000);
    regs[2] = UINT64_C(0x1000);
    regs[6] = UINT64_C(0x40000000);   /* incoming NZCV: Z=1 */

    ce_result_t res;
    if (!run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }
    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->cpu.gpr[0], UINT64_C(0x1000));     /* result written          */
    EXPECT_EQ(e->cpu.nzcv,   UINT64_C(0x40000000)); /* NZCV unchanged (no S)   */
}

/* ADDG  X<rd>, X<rn>, #uimm6, #uimm4  (uimm6 unit = 16 bytes; uimm4 = tag delta) */
static uint32_t enc_addg(int rd, int rn, int uimm6, int uimm4) {
    return 0x91800000u | ((uint32_t)(uimm6 & 0x3F) << 16) | ((uint32_t)(uimm4 & 0xF) << 10)
         | ((uint32_t)rn << 5) | (uint32_t)rd;
}
/* SUBG  X<rd>, X<rn>, #uimm6, #uimm4 */
static uint32_t enc_subg(int rd, int rn, int uimm6, int uimm4) {
    return 0xD1800000u | ((uint32_t)(uimm6 & 0x3F) << 16) | ((uint32_t)(uimm4 & 0xF) << 10)
         | ((uint32_t)rn << 5) | (uint32_t)rd;
}

/* ADDG adds (uimm6*16) to address[55:0] and uimm4 to tag[59:56] independently;
 * attribute[63:60] preserved, no carry between the fields. */
static void test_integration_mte_addg(void) {
    uint32_t code[] = { enc_addg(0, 1, 1, 1), enc_ldr_reg(3, 29) };  /* ADDG X0,X1,#16,#1 */
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = UINT64_C(0xF200000000001000);   /* attr=F, tag=2, addr=0x1000 */
    ce_result_t res;
    if (!run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }
    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;
    /* addr 0x1000+0x10=0x1010, tag 2+1=3, attr F → 0xF300000000001010 */
    EXPECT_EQ(e->cpu.gpr[0], UINT64_C(0xF300000000001010));
}

/* SUBG mirrors ADDG with subtraction; tag wraps mod 16. */
static void test_integration_mte_subg(void) {
    uint32_t code[] = { enc_subg(0, 1, 1, 1), enc_ldr_reg(3, 29) };  /* SUBG X0,X1,#16,#1 */
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = UINT64_C(0xF300000000001010);   /* attr=F, tag=3, addr=0x1010 */
    ce_result_t res;
    if (!run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }
    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;
    /* addr 0x1010-0x10=0x1000, tag 3-1=2, attr F → 0xF200000000001000 */
    EXPECT_EQ(e->cpu.gpr[0], UINT64_C(0xF200000000001000));
}

/* LDG X<rt>, [X<rn>] (offset 0). */
static uint32_t enc_ldg(int rt, int rn) {
    return 0xD9600000u | ((uint32_t)rn << 5) | (uint32_t)rt;
}

/* An emulated memory-tag op (LDG X0,[X0], Rt==Rn) must not leave a uaddr in its base register:
 * base_hook excludes the memory-tag family from the native kaddr<->uaddr translation. After LDG, X0
 * keeps its sandbox address (only the tag field [59:56] is rewritten; tag 0 with no tag memory). */
static void test_integration_ldg_base_not_translated(void) {
    uint32_t code[] = { enc_ldg(0, 0), enc_ldr_reg(9, 29) };   /* LDG X0,[X0] ; LDR X9,[X29] */
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = KBASE;
    ce_result_t res;
    if (!run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }
    instr_trace_entry_t *e = find_mem_entry(&res, 0);          /* LDR X9 entry: post-LDG cpu state */
    EXPECT(e != NULL);
    if (!e) return;
    EXPECT_EQ(e->cpu.gpr[0] & UINT64_C(0x00FFFFFFFFFFFFFF), KBASE & UINT64_C(0x00FFFFFFFFFFFFFF));
}

/* ---- PAC sign / auth / strip — needs the kernel module (EL1 keys) ------- *
 * PAC keys are EL1 registers, so the CE routes these through /dev/executor.
 * They run on the VM once revizor-executor.ko is loaded, and SKIP otherwise. */

static uint32_t enc_pacga(int rd, int rn, int rm) {
    return 0x9AC03000u | ((uint32_t)rm << 16) | ((uint32_t)rn << 5) | (uint32_t)rd;
}
static uint32_t enc_xpaci(int rd) { return 0xDAC143E0u | (uint32_t)rd; }

/* PACGA: MAC in bits[63:32], [31:0] zero, deterministic for equal inputs. */
static void test_integration_pac_pacga(void) {
    SKIP_IF_NO_EXECUTOR();
    uint32_t code[] = { enc_pacga(0, 1, 2), enc_pacga(3, 1, 2), enc_ldr_reg(4, 29) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = KBASE;
    regs[2] = UINT64_C(0x42);

    ce_result_t res;
    if (!run_ce(code, 3, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }
    instr_trace_entry_t *e = find_mem_entry(&res, 0);
    EXPECT(e != NULL);
    if (!e) return;

    EXPECT_EQ(e->cpu.gpr[0] & UINT64_C(0xFFFFFFFF), (uint64_t)0);
    EXPECT(e->cpu.gpr[0] != 0);
    EXPECT_EQ(e->cpu.gpr[0], e->cpu.gpr[3]);
}

/* PACIA then XPACI strips the PAC field back to the original pointer (no auth). */
static void test_integration_pac_sign_then_strip(void) {
    SKIP_IF_NO_EXECUTOR();
    uint32_t code[] = { enc_pacia(0, 1), enc_ldr_reg(9, 29), enc_xpaci(0), enc_ldr_reg(2, 29) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = KBASE;
    regs[1] = UINT64_C(0x99);

    ce_result_t res;
    if (!run_ce(code, 4, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }
    EXPECT_EQ(count_mem_entries(&res), 2);
    instr_trace_entry_t *a = find_mem_entry(&res, 0);
    instr_trace_entry_t *b = find_mem_entry(&res, 1);
    EXPECT(a != NULL); EXPECT(b != NULL);
    if (!a || !b) return;
    EXPECT(a->cpu.gpr[0] != KBASE);
    EXPECT_EQ(b->cpu.gpr[0], KBASE);
}

/* PACIA then AUTIA with matching key+context authenticates and recovers the
 * pointer. A *failing* auth faults the CPU (resets FEAT-FPAC) and exercises the
 * flagged pac.c key-swap path — run on the recoverable VM first. */
static void test_integration_pac_sign_then_auth(void) {
    SKIP_IF_NO_EXECUTOR();
    uint32_t code[] = { enc_pacia(0, 1), enc_ldr_reg(9, 29), enc_autia(0, 1), enc_ldr_reg(2, 29) };
    uint8_t mem[MEM_SIZE];
    memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[0] = KBASE;
    regs[1] = UINT64_C(0x99);

    ce_result_t res;
    if (!run_ce(code, 4, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        ++g_tests_run; ++g_tests_failed;
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__);
        return;
    }
    EXPECT_EQ(count_mem_entries(&res), 2);
    instr_trace_entry_t *a = find_mem_entry(&res, 0);
    instr_trace_entry_t *b = find_mem_entry(&res, 1);
    EXPECT(a != NULL); EXPECT(b != NULL);
    if (!a || !b) return;
    EXPECT(a->cpu.gpr[0] != KBASE);   /* signed */
    EXPECT_EQ(b->cpu.gpr[0], KBASE);  /* authenticated back */
}

/* ============================================================================
 * MTE mode: after-access tag correction
 * ----------------------------------------------------------------------------
 * In MTE-test mode (MTE_TAGS supplied) the pointer used by a data access carries the accessed
 * granule's tag *afterwards* -- the CE fixes-after, genuine fixes-before (the ADDG), so the two
 * flows differ only in the tag at the access itself. Before this correction these would fail.
 * ============================================================================ */

static void test_integration_mte_after_access_correction(void) {
    uint8_t mem[64]; memset(mem, 0, sizeof(mem));
    uint8_t tags[2] = { 0x05, 0x00 };               /* 4 granules; granule 0 = tag 5 */
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = KBASE;                                 /* X1 = clean pointer (tag 0) to granule 0 */

    uint32_t code[] = { 0xF9400020u /* LDR X0,[X1] */, 0xD503201Fu /* NOP */ };
    ce_result_t res;
    if (!run_ce_full(code, 2, mem, sizeof(mem), KBASE, 0, 0, 0u, BRANCH_PREDICTOR_NONE,
                     regs, tags, sizeof(tags), &res)) {
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__); ++g_tests_failed; return;
    }
    EXPECT(res.n_entries >= 2);
    /* KBASE's tag nibble [59:56] is already 0xF; the correction REPLACES it with the cell tag 5. */
    EXPECT_EQ(res.entries[0].cpu.gpr[1], KBASE);                                       /* before */
    EXPECT_EQ(res.entries[1].cpu.gpr[1], (KBASE & ~((uint64_t)0xF << 56)) | ((uint64_t)0x5 << 56));
}

/* Control: no MTE_TAGS -> not MTE mode -> the pointer's tag is NOT touched after an access. */
static void test_integration_mte_inactive_no_correction(void) {
    uint8_t mem[64]; memset(mem, 0, sizeof(mem));
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = KBASE;
    uint32_t code[] = { 0xF9400020u /* LDR X0,[X1] */, 0xD503201Fu /* NOP */ };
    ce_result_t res;
    if (!run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u, regs, &res)) {
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__); ++g_tests_failed; return;
    }
    EXPECT(res.n_entries >= 2);
    EXPECT_EQ(res.entries[1].cpu.gpr[1], KBASE);   /* unchanged: no tag memory */
}

/* BUG #3 corner case: the corrected tag propagates. LDR fixes X1's tag to the cell's (5); a
 * downstream STG stores X1's tag into granule 1; LDG reads it back as 5. Without the after-access
 * correction X1's tag would still be 0 and this would read 0. */
static void test_integration_mte_correction_propagates(void) {
    uint8_t mem[64]; memset(mem, 0, sizeof(mem));
    uint8_t tags[2] = { 0x15, 0x00 };               /* granule 0 = 5, granule 1 = 1 */
    uint64_t regs[REGS_COUNT] = { 0 };
    regs[1] = KBASE;                                 /* X1 -> granule 0 (tag 5) */
    regs[3] = KBASE + 16;                            /* X3 -> granule 1 (tag 1) */

    uint32_t code[] = {
        0xF9400020u,   /* LDR X0,[X1]  -> after-access correction sets X1 tag = 5 */
        0xD9200861u,   /* STG X1,[X3]  -> store X1's tag (5) into granule 1        */
        0xD9600064u,   /* LDG X4,[X3]  -> X4[59:56] = granule 1 tag (now 5)        */
        0xD503201Fu,   /* NOP                                                      */
    };
    ce_result_t res;
    if (!run_ce_full(code, 4, mem, sizeof(mem), KBASE, 0, 0, 0u, BRANCH_PREDICTOR_NONE,
                     regs, tags, sizeof(tags), &res)) {
        fprintf(stderr, "FAIL %s: run_ce failed\n", __func__); ++g_tests_failed; return;
    }
    EXPECT(res.n_entries >= 4);
    EXPECT_EQ(res.entries[3].cpu.gpr[4] >> 56, (uint64_t)0x5);   /* corrected tag propagated */
}

/* ============================================================================
 * Base-register-aliasing stores. A store whose data register aliases the base
 * must persist the kaddr, not the uaddr the CE injects for native execution.
 * Each stores a base-derived pointer, reads it back, and checks the kaddr bits.
 * Regression for the pair second-element bug; covers regular / 64-bit / atomic.
 * ============================================================================ */
#define AL_OFF 0x40u

/* Run store `st` then load `ld`; return the load's trace entry (2nd mem access). If `mem_qword` is
 * non-NULL it preloads 8 bytes at [KBASE+AL_OFF] (for CAS's compare). */
static instr_trace_entry_t* alias_store_load(uint32_t st, uint32_t ld, const uint64_t* regs,
                                             const uint64_t* mem_qword, ce_result_t* res) {
    uint8_t mem[MEM_SIZE]; memset(mem, 0, sizeof(mem));
    if (mem_qword) memcpy(mem + AL_OFF, mem_qword, 8);
    uint32_t code[2] = { st, ld };
    if (!run_ce(code, 2, mem, sizeof(mem), KBASE, 0, 0u, regs, res)) return NULL;
    return find_mem_entry(res, 1);
}
#define AL_FAIL() do { fprintf(stderr, "FAIL %s: run_ce failed\n", __func__); ++g_tests_failed; return; } while (0)

static void test_alias_stp_pair_both_w(void) {          /* STP W1,W1,[X1] — the pair 2nd-element bug */
    uint64_t regs[REGS_COUNT] = {0}; regs[1] = KBASE + AL_OFF;
    ce_result_t res; instr_trace_entry_t* ld = alias_store_load(0x29000421u, 0x29401023u, regs, NULL, &res);
    if (!ld) AL_FAIL();
    uint64_t exp = (KBASE + AL_OFF) & 0xFFFFFFFFu;
    EXPECT(ld->metadata.is_pair);
    EXPECT_EQ(ld->metadata.memory_access.before,  exp);   /* element 0 */
    EXPECT_EQ(ld->metadata.memory_access2.before, exp);   /* element 1 (was uaddr before the fix) */
}
static void test_alias_stp_pair_both_x(void) {          /* STP X1,X1,[X1] — 64-bit both elements */
    uint64_t regs[REGS_COUNT] = {0}; regs[1] = KBASE + AL_OFF;
    ce_result_t res; instr_trace_entry_t* ld = alias_store_load(0xa9000421u, 0xa9401023u, regs, NULL, &res);
    if (!ld) AL_FAIL();
    EXPECT(ld->metadata.is_pair);
    EXPECT_EQ(ld->metadata.memory_access.before,  KBASE + AL_OFF);
    EXPECT_EQ(ld->metadata.memory_access2.before, KBASE + AL_OFF);
}
static void test_alias_stp_pair_elem0(void) {           /* STP W1,W0,[X1] — only element 0 aliases */
    uint64_t regs[REGS_COUNT] = {0}; regs[1] = KBASE + AL_OFF;
    ce_result_t res; instr_trace_entry_t* ld = alias_store_load(0x29000021u, 0x29401023u, regs, NULL, &res);
    if (!ld) AL_FAIL();
    EXPECT_EQ(ld->metadata.memory_access.before,  (KBASE + AL_OFF) & 0xFFFFFFFFu);  /* elem0 = kaddr */
    EXPECT_EQ(ld->metadata.memory_access2.before, (uint64_t)0);                     /* elem1 = X0   */
}
static void test_alias_stp_pair_elem1(void) {           /* STP W0,W1,[X1] — only element 1 aliases */
    uint64_t regs[REGS_COUNT] = {0}; regs[1] = KBASE + AL_OFF;
    ce_result_t res; instr_trace_entry_t* ld = alias_store_load(0x29000420u, 0x29401023u, regs, NULL, &res);
    if (!ld) AL_FAIL();
    EXPECT_EQ(ld->metadata.memory_access.before,  (uint64_t)0);                     /* elem0 = X0   */
    EXPECT_EQ(ld->metadata.memory_access2.before, (KBASE + AL_OFF) & 0xFFFFFFFFu);  /* elem1 = kaddr */
}
static void test_alias_str_x(void) {                    /* STR X1,[X1] — regular 64-bit */
    uint64_t regs[REGS_COUNT] = {0}; regs[1] = KBASE + AL_OFF;
    ce_result_t res; instr_trace_entry_t* ld = alias_store_load(0xf9000021u, 0xf9400023u, regs, NULL, &res);
    if (!ld) AL_FAIL();
    EXPECT_EQ(ld->metadata.memory_access.before, KBASE + AL_OFF);
}
static void test_alias_str_w(void) {                    /* STR W1,[X1] — regular 32-bit */
    uint64_t regs[REGS_COUNT] = {0}; regs[1] = KBASE + AL_OFF;
    ce_result_t res; instr_trace_entry_t* ld = alias_store_load(0xb9000021u, 0xb9400023u, regs, NULL, &res);
    if (!ld) AL_FAIL();
    EXPECT_EQ(ld->metadata.memory_access.before, (KBASE + AL_OFF) & 0xFFFFFFFFu);
}
static void test_alias_atomic_swp(void) {               /* SWP X1,X0,[X1] — source Rs aliases base */
    uint64_t regs[REGS_COUNT] = {0}; regs[1] = KBASE + AL_OFF;
    ce_result_t res; instr_trace_entry_t* ld = alias_store_load(0xf8218020u, 0xf9400023u, regs, NULL, &res);
    if (!ld) AL_FAIL();
    EXPECT_EQ(ld->metadata.memory_access.before, KBASE + AL_OFF);
}
static void test_alias_atomic_ldadd(void) {             /* LDADD X1,X0,[X1] — [X1]=0 + X1 = kaddr */
    uint64_t regs[REGS_COUNT] = {0}; regs[1] = KBASE + AL_OFF;
    ce_result_t res; instr_trace_entry_t* ld = alias_store_load(0xf8210020u, 0xf9400023u, regs, NULL, &res);
    if (!ld) AL_FAIL();
    EXPECT_EQ(ld->metadata.memory_access.before, KBASE + AL_OFF);
}
/* CAS/CASP: the aliasing fixup covers them in principle, but the CE cannot execute CAS natively
 * (pre-existing limitation — it crashes regardless of aliasing), so there is nothing to drive yet. */

int main(void) {
    printf("Running CE integration tests (fork+exec)...\n");

    if (access(ce_binary_path(), X_OK) != 0) {
        fprintf(stderr, "SKIP: %s not found or not executable\n", ce_binary_path());
        printf("\n0 tests, 0 failed (CE binary not found)\n");
        return 0;
    }

    test_integration_ldr_base();
    test_integration_str_base();
    test_integration_mte_after_access_correction();
    test_integration_mte_inactive_no_correction();
    test_integration_mte_correction_propagates();
    test_integration_ldr_unsigned_offset();
    test_integration_ldr_postidx();
    test_integration_ldr_preidx();
    test_integration_ldp();
    test_integration_kaddr_transparency();
    test_integration_sequential_loads();
    test_integration_store_then_load();

    test_integration_nop_no_mem();
    test_integration_access_size_byte();
    test_integration_access_size_halfword();
    test_integration_access_size_word();
    test_integration_strb_partial_write();
    test_integration_non_x29_base();
    test_integration_nzcv_in_cpu_state();
    test_integration_register_offset();
    test_integration_regoff_noncanonical_base_preserved();
    test_integration_load_updates_dest_reg();
    test_integration_postidx_writeback_cpu_state();
    test_integration_always_mispredict_cbz();
    test_integration_window_id_unique_per_excursion();
    test_integration_spec_window_cap();
    test_integration_always_mispredict_cbnz();
    test_integration_tbz_bit32();
    test_integration_always_mispredict_bcond();
    test_integration_always_mispredict_tbnz();
    test_integration_contract_type_comparison();
    test_integration_bpas_store_bypass();
    test_integration_bpas_trailing_store();
    test_integration_bpas_preserves_writeback();
    test_integration_bpas_store_pair();
    test_integration_bpas_max_nesting_zero();
    test_integration_bpas_consecutive_stores();
    test_integration_bpas_store_logged_after_bypassing_load();
    test_integration_bpas_nested_stores_logged_after_stale_load();
    test_integration_cond_bpas_composition();
    test_integration_cond_nesting_cap();
    test_integration_bpas_three_stores_value_set();
    test_integration_barrier_ssbb_commits_no_squash();
    test_integration_barrier_ssbb_commits_no_squash_nonadjacent();
    test_integration_barrier_ssbb_nested_commits_no_squash();
    test_integration_barrier_ssbb_register_bypass_survives();
    test_integration_barrier_pssbb_commits_no_squash();
    test_integration_barrier_isb_squashes_bpas();
    test_integration_barrier_dmb_no_cut();
    test_integration_barrier_ssbb_no_store();
    test_integration_barrier_store_after_not_fenced();
    test_integration_barrier_load_before_barrier();
    test_integration_barrier_load_before_store_after();
    test_integration_barrier_sb_no_speculation();
    test_integration_barrier_ssbb_spares_enclosing_branch();
    test_integration_barrier_control_fence_variants();
    test_integration_barrier_control_fence_granularity();
    test_integration_cond_bpas_branch_before_stores();
    test_integration_cond_bpas_branch_after_stores();
    test_integration_cond_bpas_branch_in_middle();
    test_integration_cond_bpas_store_then_branch();
    test_integration_cond_bpas_branch_then_store_flow();
    test_integration_cond_bpas_branches_in_middle();
    test_integration_cond_only_nesting_cap();
    test_integration_cond_bpas_interleaved_nesting_cap();
    test_integration_cond_bpas_outer_window_survives_excursion();
    test_integration_forks_store_vs_branch_symmetry();
    test_integration_bpas_store_pair_single_fork();
    test_integration_cond_bpas_store_encloses_branch();
    test_integration_unsupported_clauses_rejected();
    test_integration_bpu_input_selected_predictor();
    test_integration_bpu_without_predictor_traps();

    test_integration_ldg_base_not_translated();
    test_integration_mte_subps_equal_tagged_pointers();
    test_integration_mte_subps_negative();
    test_integration_mte_subp_preserves_nzcv();
    test_integration_mte_addg();
    test_integration_mte_subg();

    /* base-register-aliasing stores (the pair second-element fixup bug + variations) */
    test_alias_stp_pair_both_w();
    test_alias_stp_pair_both_x();
    test_alias_stp_pair_elem0();
    test_alias_stp_pair_elem1();
    test_alias_str_x();
    test_alias_str_w();
    test_alias_atomic_swp();
    test_alias_atomic_ldadd();

    // PAC (need /dev/executor; skip without it). sign_then_auth is a matched,
    // non-faulting round-trip — but a failing auth resets FEAT-FPAC silicon, so
    // it also exercises the flagged pac.c key-swap path; run on the VM first.
    test_integration_pac_pacga();
    test_integration_pac_sign_then_strip();
    test_integration_pac_sign_then_auth();
    test_integration_pac_arch_roundtrip();
    test_integration_pac_spec_xpac();
    test_integration_paciza_autiza_roundtrip();
    test_integration_pacda_autda_roundtrip();

    // Still disabled: these sign with one context and auth with another (or on a
    // mismatched key), so the AUT* is expected to fault — fatal on FEAT-FPAC.
    // Only enable behind an FPAC guard.
    // test_integration_pac_arch_roundtrip();
    // test_integration_pac_spec_xpac();
    // test_integration_paciza_autiza_roundtrip();
    // test_integration_pacda_autda_roundtrip();

    printf("\n%d tests, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed ? 1 : 0;
}
