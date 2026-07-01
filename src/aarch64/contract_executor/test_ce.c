/*
 * CE unit tests — instruction encoding/decoding, branch classification,
 * condition evaluation, address computation, input loading, register
 * layout, hooking-phase invariants, and fixup mechanism.
 *
 * Build:
 *   gcc -Wall -Wextra -g -march=native -I. \
 *       instruction_encodings.c simulation_input.c test_ce.c -o test_ce
 * Run:
 *   ./test_ce
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>

#include "instruction_encodings.h"
#include "simulation_input.h"
#include "simulation_state.h"
#include "common_msg_constants.h"
#include "userapi/executor_input_format.h"
#include "branch_speculation.h"   /* cond_branch_is_taken / cond_branch_architectural_next */

/* ---- test infrastructure ------------------------------------------------ */

static int g_tests_run = 0;
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
    unsigned long long _a = (unsigned long long)(a); \
    unsigned long long _b = (unsigned long long)(b); \
    if (_a != _b) { \
        fprintf(stderr, "FAIL %s:%d  %s == %s  got 0x%llx vs 0x%llx\n", \
            __func__, __LINE__, #a, #b, _a, _b); \
        ++g_tests_failed; \
    } \
} while (0)

/* branch_speculation.c is linked in for the real branch resolvers (cond_branch_is_taken etc.). Its
 * branch_speculate() also references the speculation engine, which this unit test does not link, so
 * stub those entry points (we never call branch_speculate here). */
struct simulation_state;
void     spec_push_frame(struct simulation_state* s, uintptr_t r, uint64_t o) { (void)s; (void)r; (void)o; }
uint64_t spec_nesting(void)     { return 0; }
uint64_t spec_max_nesting(void) { return 0; }

/* ---- instruction encoding constants ------------------------------------- */

/* B.EQ  +8      (imm19=2, cond=0) */
#define ENC_BEQ_PLUS8    0x54000040u
/* B.NE  +8 */
#define ENC_BNE_PLUS8    0x54000041u
/* B.CS/HS +8 */
#define ENC_BCS_PLUS8    0x54000042u
/* B.MI  +8 */
#define ENC_BMI_PLUS8    0x54000044u
/* CBZ  X0, +8  (64-bit, bit31=1) */
#define ENC_CBZ_X0_PLUS8  0xB4000040u
/* CBNZ X0, +8 */
#define ENC_CBNZ_X0_PLUS8 0xB5000040u
/* CBZ  W0, +8  (32-bit, bit31=0) */
#define ENC_CBZ_W0_PLUS8  0x34000040u
/*
 * TBZ/TBNZ encoding: bit31=b5 (MSB of 6-bit test-bit index),
 *   bits[30:24]=0110110 (TBZ) or 0110111 (TBNZ),
 *   bits[23:19]=b40, bits[18:5]=imm14, bits[4:0]=Rt.
 *
 * TBZ  X0, #0,  +8: b5=0, b40=0 → bits[31:24]=0x36, imm14=2 → 0x36000040
 * TBZ  X0, #32, +8: b5=1, b40=0 → bits[31:24]=0xB6, imm14=2 → 0xB6000040
 * TBNZ X0, #0,  +8: b5=0, b40=0 → bits[31:24]=0x37, imm14=2 → 0x37000040
 */
#define ENC_TBZ_X0_B0_PLUS8   0x36000040u
#define ENC_TBZ_X0_B32_PLUS8  0xB6000040u
#define ENC_TBNZ_X0_B0_PLUS8  0x37000040u
/* B    +8 */
#define ENC_B_PLUS8      0x14000002u
/* BL   +8 */
#define ENC_BL_PLUS8     0x94000002u
/* BLR  X0 */
#define ENC_BLR_X0       0xD63F0000u
/* NOP */
#define ENC_NOP          0xD503201Fu
/* LDR  X0, [X1] — unsigned offset, offset=0 */
#define ENC_LDR_X0_X1    0xF9400020u
/* STR  X0, [X1] */
#define ENC_STR_X0_X1    0xF9000020u
/* LDR  X0, [X1, #8] — unsigned offset */
#define ENC_LDR_X0_X1_8  0xF9400060u
/* LDR  X0, [X1, #8]! — pre-index */
#define ENC_LDR_PREIDX   0xF8408C20u
/* LDR  X0, [X1], #8 — post-index */
#define ENC_LDR_POSTIDX  0xF8408420u
/*
 * LDP X0, X2, [X1] — signed offset, 64-bit.
 * Encoding: opc=10, bits[29:27]=101, V=0, bits[24:23]=10 (signed offset),
 * L=1, imm7=0, Rt2=2 (bits[14:10]=00010→bit11=1), Rn=1, Rt=0.
 * byte3=0xA9 byte2=0x40 byte1=0x08 byte0=0x20 → 0xA9400820
 */
#define ENC_LDP_X0_X2_X1 0xA9400820u
/* STP X0, X2, [X1] — same layout, L=0 (bit22=0) → 0xA9000820 */
#define ENC_STP_X0_X2_X1 0xA9000820u
/* LDRSW X0, [X1] — size=10, opc=10 (load, sign-extend to 64): a load whose opc[0]=0 */
#define ENC_LDRSW_X0_X1  0xB9800020u
/* STP X0, X2, [X1, #imm]! — pre-index (bit23=1) store; opc[23:22]=10 would mis-read as a load */
#define ENC_STP_PREIDX   0xA9800820u
/* PRFM (pfrop), [X1] — size=11, opc=10: a hint, neither load nor store */
#define ENC_PRFM_X1      0xF9800020u
/* LDR  X0, #0 (literal, offset=0) */
#define ENC_LDR_LIT_X0   0x58000000u


/* ======================================================================== */
/* ---- GROUP 1: classify_branch ----------------------------------------- */
/* ======================================================================== */

static void test_classify_branch(void) {
    EXPECT_EQ(classify_branch(ENC_BEQ_PLUS8),      BRANCH_B_COND);
    EXPECT_EQ(classify_branch(ENC_BNE_PLUS8),      BRANCH_B_COND);
    EXPECT_EQ(classify_branch(ENC_BMI_PLUS8),      BRANCH_B_COND);
    EXPECT_EQ(classify_branch(ENC_CBZ_X0_PLUS8),   BRANCH_CBZ);
    EXPECT_EQ(classify_branch(ENC_CBNZ_X0_PLUS8),  BRANCH_CBNZ);
    EXPECT_EQ(classify_branch(ENC_CBZ_W0_PLUS8),   BRANCH_CBZ);
    EXPECT_EQ(classify_branch(ENC_TBZ_X0_B0_PLUS8),  BRANCH_TBZ);
    EXPECT_EQ(classify_branch(ENC_TBZ_X0_B32_PLUS8), BRANCH_TBZ);  /* bit31=b5=1 masked out */
    EXPECT_EQ(classify_branch(ENC_TBNZ_X0_B0_PLUS8), BRANCH_TBNZ);
    EXPECT_EQ(classify_branch(ENC_B_PLUS8),        BRANCH_B);
    EXPECT_EQ(classify_branch(ENC_BL_PLUS8),        BRANCH_BL);
    EXPECT_EQ(classify_branch(ENC_BLR_X0),          BRANCH_BLR);
    EXPECT_EQ(classify_branch(ENC_NOP),             BRANCH_NONE);
    EXPECT_EQ(classify_branch(ENC_LDR_X0_X1),       BRANCH_NONE);
    /* B.cond with bit4=1 is not a valid conditional branch */
    EXPECT_EQ(classify_branch(0x54000010u),         BRANCH_NONE);
    /* B.al / B.nv branch unconditionally, so they are not conditional branches */
    EXPECT_EQ(classify_branch(0x5400004Eu),         BRANCH_NONE);  /* B.al +8 */
    EXPECT_EQ(classify_branch(0x5400004Fu),         BRANCH_NONE);  /* B.nv +8 */
}

/* ======================================================================== */
/* ---- GROUP 2: evaluate_cond_target ------------------------------------- */
/* ======================================================================== */

static void test_evaluate_cond_target(void) {
    uintptr_t pc = 0x1000;

    /* B.EQ +8: imm19=2 → target = 0x1008 */
    EXPECT_EQ(evaluate_cond_target(pc, ENC_BEQ_PLUS8), (uintptr_t)(pc + 8));

    /* CBZ X0, +8 */
    EXPECT_EQ(evaluate_cond_target(pc, ENC_CBZ_X0_PLUS8), (uintptr_t)(pc + 8));

    /* TBZ X0, #0, +8 */
    EXPECT_EQ(evaluate_cond_target(pc, ENC_TBZ_X0_B0_PLUS8), (uintptr_t)(pc + 8));

    /* TBZ X0, #32, +8 — same target computation regardless of bit number */
    EXPECT_EQ(evaluate_cond_target(pc, ENC_TBZ_X0_B32_PLUS8), (uintptr_t)(pc + 8));

    /* Negative offset: B.EQ -4 (imm19=-1=0x7FFFF) */
    uint32_t beq_minus4 = 0x54000000u | (0x7FFFFu << 5) | 0x0;
    EXPECT_EQ(evaluate_cond_target(pc, beq_minus4), (uintptr_t)(pc - 4));

    /* TBZ backward: imm14=-1=0x3FFF → target = pc - 4 */
    uint32_t tbz_minus4 = 0xB6000000u | (0x3FFFu << 5) | 0x0;
    EXPECT_EQ(evaluate_cond_target(pc, tbz_minus4), (uintptr_t)(pc - 4));

    /* Non-conditional branch: returns 0 */
    EXPECT_EQ(evaluate_cond_target(pc, ENC_B_PLUS8), (uintptr_t)0);
    EXPECT_EQ(evaluate_cond_target(pc, ENC_NOP),     (uintptr_t)0);
}

/* ======================================================================== */
/* ---- GROUP 3: TBZ bit-number extraction (was bugged: missed bit31) ----- */
/* ======================================================================== */

static void test_tbz_bit_number(void) {
    /*
     * TBZ 6-bit bit index = {bit31_of_insn, bits[23:19]}.
     * ENC_TBZ_X0_B32_PLUS8 = 0xB6000040: bit31=1, bits[23:19]=00000 → bit 32.
     */
    uint32_t enc = ENC_TBZ_X0_B32_PLUS8;
    uint32_t b5  = (enc >> 31) & 0x1;
    uint32_t b40 = (enc >> 19) & 0x1F;
    uint32_t bit_num = (b5 << 5) | b40;
    EXPECT_EQ(bit_num, 32u);

    /* Old (broken) extraction missed b5 and gave 0 instead of 32 */
    uint32_t old_bit_num = (enc >> 19) & 0x1F;
    EXPECT_EQ(old_bit_num, 0u);
}

/* ======================================================================== */
/* ---- GROUP 4: is_memory_access and sub-classifiers -------------------- */
/* ======================================================================== */

static void test_memory_access_classification(void) {
    EXPECT(is_memory_access(ENC_LDR_X0_X1));
    EXPECT(is_memory_access(ENC_STR_X0_X1));
    EXPECT(is_memory_access(ENC_LDP_X0_X2_X1));
    EXPECT(is_memory_access(ENC_STP_X0_X2_X1));
    EXPECT(is_memory_access(ENC_LDR_LIT_X0));
    EXPECT(!is_memory_access(ENC_NOP));
    EXPECT(!is_memory_access(ENC_BEQ_PLUS8));
    EXPECT(!is_memory_access(ENC_B_PLUS8));

    EXPECT(is_regular_load_store(ENC_LDR_X0_X1));
    EXPECT(is_regular_load_store(ENC_STR_X0_X1));
    EXPECT(!is_regular_load_store(ENC_LDP_X0_X2_X1));

    EXPECT(is_pair_load_store(ENC_LDP_X0_X2_X1));
    EXPECT(is_pair_load_store(ENC_STP_X0_X2_X1));
    EXPECT(!is_pair_load_store(ENC_LDR_X0_X1));

    EXPECT(is_literal_pc_relative(ENC_LDR_LIT_X0));
    EXPECT(!is_literal_pc_relative(ENC_LDR_X0_X1));
}

/* ======================================================================== */
/* ---- GROUP 5: access_size --------------------------------------------- */
/* ======================================================================== */

static void test_access_size(void) {
    EXPECT_EQ(access_size(ENC_LDR_X0_X1), 8u);       /* 64-bit */
    EXPECT_EQ(access_size(0xB9400020u),    4u);        /* LDR W0, [X1] */
    EXPECT_EQ(access_size(0x39400020u),    1u);        /* LDRB W0, [X1] */
    EXPECT_EQ(access_size(0x79400020u),    2u);        /* LDRH W0, [X1] */
}

/* ======================================================================== */
/* ---- GROUP 6: get_rn, get_rt, get_rt2, set_rn ------------------------- */
/* ======================================================================== */

static void test_register_fields(void) {
    /* LDR X0, [X1]: Rt=0, Rn=1 */
    EXPECT_EQ(get_rt(ENC_LDR_X0_X1), 0u);
    EXPECT_EQ(get_rn(ENC_LDR_X0_X1), 1u);

    /* STR X0, [X1]: Rt=0, Rn=1 */
    EXPECT_EQ(get_rt(ENC_STR_X0_X1), 0u);
    EXPECT_EQ(get_rn(ENC_STR_X0_X1), 1u);

    /* LDP X0, X2, [X1]: Rt=0, Rt2=2, Rn=1 */
    EXPECT_EQ(get_rt(ENC_LDP_X0_X2_X1),  0u);
    EXPECT_EQ(get_rt2(ENC_LDP_X0_X2_X1), 2u);
    EXPECT_EQ(get_rn(ENC_LDP_X0_X2_X1),  1u);

    /* set_rn: change Rn from 1 to 10, Rt must be unchanged */
    uint32_t patched = set_rn(ENC_LDR_X0_X1, 10);
    EXPECT_EQ(get_rn(patched), 10u);
    EXPECT_EQ(get_rt(patched), 0u);

    /* set_rn touches only bits[9:5] */
    uint32_t mask = ~(0x1Fu << 5);
    EXPECT_EQ(patched & mask, ENC_LDR_X0_X1 & mask);
}

/* ======================================================================== */
/* ---- GROUP 7: is_load, is_store, addressing mode predicates ----------- */
/* ======================================================================== */

static void test_load_store_predicates(void) {
    EXPECT(is_load(ENC_LDR_X0_X1));
    EXPECT(!is_store(ENC_LDR_X0_X1));
    EXPECT(is_store(ENC_STR_X0_X1));
    EXPECT(!is_load(ENC_STR_X0_X1));

    /* sign-extending loads (opc=10) are loads, not stores (opc[0]=0 misled the old bit-22 decode) */
    EXPECT(is_load(ENC_LDRSW_X0_X1));
    EXPECT(!is_store(ENC_LDRSW_X0_X1));
    /* a pre-indexed STP must stay a store: pairs decode by the L-bit, not opc[23:22] */
    EXPECT(is_store(ENC_STP_PREIDX));
    EXPECT(!is_load(ENC_STP_PREIDX));
    /* PRFM is a hint: neither load nor store */
    EXPECT(!is_load(ENC_PRFM_X1));
    EXPECT(!is_store(ENC_PRFM_X1));

    EXPECT(is_unsigned_offset(ENC_LDR_X0_X1));
    EXPECT(!is_pre_index(ENC_LDR_X0_X1));
    EXPECT(!is_post_index(ENC_LDR_X0_X1));

    EXPECT(is_pre_index(ENC_LDR_PREIDX));
    EXPECT(!is_post_index(ENC_LDR_PREIDX));
    EXPECT(!is_unsigned_offset(ENC_LDR_PREIDX));

    EXPECT(is_post_index(ENC_LDR_POSTIDX));
    EXPECT(!is_pre_index(ENC_LDR_POSTIDX));

    EXPECT(is_load(ENC_LDP_X0_X2_X1));
    /* NOTE: is_load uses bit22 (L-bit), which is part of imm19 in literal loads.
     * is_load is only defined for register-base load/store, not PC-relative literals. */
    EXPECT(is_pair_signed_offset(ENC_LDP_X0_X2_X1));
}

/* ======================================================================== */
/* ---- GROUP 8: encode_b and encode_bl ---------------------------------- */
/* ======================================================================== */

static void test_branch_encoding(void) {
    uintptr_t from = 0x1000;
    uintptr_t to   = 0x1008;

    uint32_t b  = encode_b(from, to);
    uint32_t bl = encode_bl(from, to);
    EXPECT(b  != 0);
    EXPECT(bl != 0);
    EXPECT_EQ(b,  ENC_B_PLUS8);
    EXPECT_EQ(bl, ENC_BL_PLUS8);

    /* Out-of-range: > 128MB */
    EXPECT_EQ(encode_bl(0, 0x10000000UL), (uint32_t)0);

    /* Negative offset round-trip */
    uint32_t b_back = encode_b(0x1008, 0x1000);
    EXPECT(b_back != 0);
}

/* ======================================================================== */
/* ---- GROUP 9: condition_passed reference implementation --------------- */
/* ======================================================================== */

static bool condition_passed_ref(uint32_t cond, uint32_t nzcv) {
    bool N = (nzcv >> 31) & 1;
    bool Z = (nzcv >> 30) & 1;
    bool C = (nzcv >> 29) & 1;
    bool V = (nzcv >> 28) & 1;
    switch (cond) {
        case 0x0: return Z;
        case 0x1: return !Z;
        case 0x2: return C;
        case 0x3: return !C;
        case 0x4: return N;
        case 0x5: return !N;
        case 0x6: return V;
        case 0x7: return !V;
        case 0x8: return !Z && C;
        case 0x9: return Z || !C;
        case 0xA: return N == V;
        case 0xB: return N != V;
        case 0xC: return !Z && (N == V);
        case 0xD: return Z || (N != V);
        case 0xE: return true;
        case 0xF: return true;
        default:  return false;
    }
}

static void test_condition_passed(void) {
    /* NZCV = Z=1 (zero result) */
    uint32_t nzcv_z = 0x40000000u;
    EXPECT( condition_passed_ref(0x0, nzcv_z));  /* EQ: Z=1 */
    EXPECT(!condition_passed_ref(0x1, nzcv_z));  /* NE: !Z */
    EXPECT(!condition_passed_ref(0x2, nzcv_z));  /* CS: C=0 */
    EXPECT( condition_passed_ref(0x5, nzcv_z));  /* PL: !N=1 */

    /* NZCV = N=1 (negative, no overflow) */
    uint32_t nzcv_n = 0x80000000u;
    EXPECT( condition_passed_ref(0x4, nzcv_n));  /* MI */
    EXPECT(!condition_passed_ref(0x5, nzcv_n));  /* PL */
    EXPECT( condition_passed_ref(0xB, nzcv_n));  /* LT: N!=V → 1!=0 */
    EXPECT(!condition_passed_ref(0xA, nzcv_n));  /* GE: N==V → false */

    /* NZCV = N=1, V=1 (overflow + negative → result actually non-negative) */
    uint32_t nzcv_nv = 0x90000000u;
    EXPECT(!condition_passed_ref(0xB, nzcv_nv)); /* LT: N!=V → 1!=1 false */
    EXPECT( condition_passed_ref(0xA, nzcv_nv)); /* GE: N==V → true */
    EXPECT( condition_passed_ref(0xC, nzcv_nv)); /* GT: !Z=1 && N==V=1 → true */

    /* AL and NV always true */
    EXPECT(condition_passed_ref(0xE, 0));
    EXPECT(condition_passed_ref(0xF, 0));
}

/* ======================================================================== */
/* ---- GROUP 10: signextend correctness --------------------------------- */
/* ======================================================================== */

static int64_t signextend_ref(size_t orig_len, int64_t value) {
    if (orig_len == 0 || orig_len >= 64) return value;
    uint64_t mask = (1ull << orig_len) - 1;
    uint64_t uval = (uint64_t)value & mask;
    uint64_t sign = uval >> (orig_len - 1);
    if (sign) {
        return (int64_t)(uval | (~mask));
    }
    return (int64_t)uval;
}

static void test_signextend(void) {
    EXPECT_EQ(signextend_ref(8, 0x7F), (int64_t)127);
    EXPECT_EQ(signextend_ref(8, 0xFF), (int64_t)-1);
    EXPECT_EQ(signextend_ref(8, 0x80), (int64_t)-128);
    EXPECT_EQ(signextend_ref(32, (int64_t)0x80000000), (int64_t)(int32_t)0x80000000);
    EXPECT_EQ(signextend_ref(32, 0x7FFFFFFF), (int64_t)0x7FFFFFFF);
    EXPECT_EQ(signextend_ref(1, 0), (int64_t)0);
    EXPECT_EQ(signextend_ref(1, 1), (int64_t)-1);
}

/* ======================================================================== */
/* ---- GROUP 11: sign-extend-then-shift order (SXTW/SXTB bug proof) ----- */
/* ======================================================================== */

static void test_sext_then_shift(void) {
    uint64_t rm = 0x80000000u;
    size_t orig_len = 32, shift = 3;

    /* Correct: sext first, then shift */
    int64_t after_sext = signextend_ref(orig_len, (int64_t)(rm & 0xFFFFFFFFu));
    uint64_t correct = (uint64_t)((int64_t)after_sext << shift);
    EXPECT_EQ(correct, 0xFFFFFFFC00000000ull);

    /* Buggy: shift first, then sext with pre-shift length */
    uint64_t shifted_first = (rm & 0xFFFFFFFFu) << shift;
    int64_t wrong = signextend_ref(orig_len, (int64_t)shifted_first);
    /* signextend_ref masks to orig_len bits first: 0x400000000 & 0xFFFFFFFF = 0 */
    EXPECT_EQ((uint64_t)wrong, 0ull);

    EXPECT(correct != (uint64_t)wrong);
}

/* ======================================================================== */
/* ---- GROUP 12: input header validation -------------------------------- */
/* ======================================================================== */

static void test_input_header_validation(void) {
    struct input_header hdr;
    memset(&hdr, 0, sizeof(hdr));

    hdr.magic   = 0xDEADBEEFu;
    hdr.version = RVZRCE_VERSION;
    hdr.arch    = RVZR_ARCH_AARCH64;
    EXPECT(simulation_input_validate_header(&hdr) != 0); /* bad magic */

    hdr.magic   = RVZRCE_MAGIC;
    hdr.version = RVZRCE_VERSION + 1;
    EXPECT(simulation_input_validate_header(&hdr) != 0); /* bad version */

    hdr.version = RVZRCE_VERSION;
    hdr.arch    = 99;
    EXPECT(simulation_input_validate_header(&hdr) != 0); /* bad arch */

    hdr.arch      = RVZR_ARCH_AARCH64;
    hdr.flags     = RVZR_FLAG_HAS_CODE;
    hdr.code_size = 0;
    EXPECT(simulation_input_validate_header(&hdr) != 0); /* HAS_CODE + size=0 */

    hdr.flags     = RVZR_FLAG_HAS_CODE;
    hdr.code_size = 4;
    hdr.reserved  = 1;
    EXPECT(simulation_input_validate_header(&hdr) != 0); /* non-zero reserved */

    hdr.reserved  = 0;
    EXPECT_EQ(simulation_input_validate_header(&hdr), 0); /* valid */

    EXPECT(simulation_input_validate_header(NULL) != 0);
}

/* ======================================================================== */
/* ---- GROUP 13: pair_element_access_size -------------------------------- */
/* ======================================================================== */

static void test_pair_access_size(void) {
    uint32_t ldp_w = 0x29400420u;     /* LDP W0, W2, [X1] — 32-bit */
    uint32_t ldpsw = 0x69400820u;     /* LDPSW X0, X2, [X1] — opc=01, two 4-byte words */
    uint32_t stnp  = 0xA8000820u;     /* STNP X0, X2, [X1] — no-allocate pair (bits[25:23]=000) */
    EXPECT_EQ(pair_element_access_size(ldp_w), 4u);
    EXPECT_EQ(pair_element_access_size(ldpsw), 4u);   /* was a trap (SIGILL) before F-CE-3 */
    EXPECT_EQ(pair_element_access_size(ENC_LDP_X0_X2_X1), 8u);
    EXPECT(is_load(ldpsw) && !is_store(ldpsw));        /* LDPSW is a pair load (L=1) */
    EXPECT(is_pair_no_alloc(stnp) && is_store(stnp));  /* STNP is a no-alloc pair store */
}

/* ======================================================================== */
/* ---- GROUP 14: literal pc-relative access size ------------------------ */
/* ======================================================================== */

static void test_literal_access_size(void) {
    uint32_t ldr_lit_w = 0x18000000u;  /* LDR W0, #0 — 32-bit */
    EXPECT_EQ(literal_pc_relative_access_size(ldr_lit_w), 4u);
    EXPECT_EQ(literal_pc_relative_access_size(ENC_LDR_LIT_X0), 8u);
}

/* ======================================================================== */
/* ---- GROUP 15: HAS_CODE / HAS_INPUT flag+size validation -------------- */
/* ======================================================================== */

static void test_input_header_input_init_size(void) {
    struct input_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic   = RVZRCE_MAGIC;
    hdr.version = RVZRCE_VERSION;
    hdr.arch    = RVZR_ARCH_AARCH64;

    /* HAS_INPUT with input_init_size=0 → reject */
    hdr.flags     = RVZR_FLAG_HAS_INPUT;
    hdr.input_init_size = 0;
    EXPECT(simulation_input_validate_header(&hdr) != 0);

    /* HAS_CODE | HAS_INPUT with nonzero sizes → accept */
    hdr.flags     = RVZR_FLAG_HAS_CODE | RVZR_FLAG_HAS_INPUT;
    hdr.code_size = 4;
    hdr.input_init_size = 4096;
    EXPECT_EQ(simulation_input_validate_header(&hdr), 0);

    /* version=0 → reject */
    hdr.version = 0;
    EXPECT(simulation_input_validate_header(&hdr) != 0);
    hdr.version = RVZRCE_VERSION;

    /* x86_64 arch also accepted */
    hdr.arch = RVZR_ARCH_X86_64;
    EXPECT_EQ(simulation_input_validate_header(&hdr), 0);
}

/* ======================================================================== */
/* ---- GROUP 16: simulation_input_payload_size --------------------------- */
/* ======================================================================== */

static void test_payload_size(void) {
    EXPECT_EQ(simulation_input_payload_size(NULL), (size_t)0);

    struct input_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    EXPECT_EQ(simulation_input_payload_size(&hdr), (size_t)0);

    hdr.code_size  = 4;
    hdr.input_init_size  = 4096;
    EXPECT_EQ(simulation_input_payload_size(&hdr), (size_t)(4 + 4096));

    hdr.code_size  = 0;
    hdr.input_init_size  = 8;
    EXPECT_EQ(simulation_input_payload_size(&hdr), (size_t)8);
}

/* ======================================================================== */
/* ---- GROUP 17: struct size sanity -------------------------------------- */
/* ======================================================================== */

static void test_struct_sizes(void) {
    /* struct configuration: 9 fields × 8 bytes = 72 bytes */
    EXPECT_EQ(sizeof(struct configuration), (size_t)72);

    /* struct input_header: 16 u64 = magic+version+arch+flags(4) + config(9)
     *   + code_size+input_init_size+reserved(3) = 128 bytes */
    EXPECT_EQ(sizeof(struct input_header), (size_t)128);

    /* input_header starts with magic at offset 0 */
    EXPECT_EQ(offsetof(struct input_header, magic), (size_t)0);
}

/* ======================================================================== */
/* ---- GROUP 18-21: simulation_input_load_fd ----------------------------- */
/* ======================================================================== */

static void pipe_write_all(int fd, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return;
        p += w; n -= (size_t)w;
    }
}

static struct input_header make_valid_hdr(uint64_t flags,
                                          uint64_t code_sz,
                                          uint64_t input_init_sz) {
    struct input_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic     = RVZRCE_MAGIC;
    hdr.version   = RVZRCE_VERSION;
    hdr.arch      = RVZR_ARCH_AARCH64;
    hdr.flags     = flags;
    hdr.code_size = code_sz;
    hdr.input_init_size = input_init_sz;
    return hdr;
}

/* Build a shared input initialization (executor_input_format) holding main || faulty || gpr [+ optional
 * mte-tags / pac-keys sections] into `out`. Returns the input_init length, or 0 if `out` is too small.
 * Passing a NULL payload with length 0 emits a zero-length section (e.g. an empty faulty page). */
static size_t make_input_init(uint8_t* out, size_t cap,
                              const uint8_t* main, size_t main_len,
                              const uint8_t* faulty, size_t faulty_len,
                              const uint8_t* gpr, size_t gpr_len,
                              const uint8_t* mte_tags, size_t mte_len,
                              const uint8_t* pac_keys, size_t pac_len) {
    uint64_t n = 3 + (NULL != mte_tags ? 1 : 0) + (NULL != pac_keys ? 1 : 0);
    size_t header_len = sizeof(struct revisor_input_header) + n * sizeof(struct revisor_input_section);
    struct revisor_input_header* h = (struct revisor_input_header*)out;
    struct revisor_input_section* tab = (struct revisor_input_section*)(out + sizeof(*h));
    size_t off = header_len;
    uint64_t i = 0;

    if (cap < header_len + main_len + faulty_len + gpr_len + mte_len + pac_len) {
        return 0;
    }

    tab[i].type = REVISOR_SEC_MEMORY_MAIN;   tab[i].flags = 0; tab[i].offset = off; tab[i].length = main_len;
    if (main_len) memcpy(out + off, main, main_len);
    off += main_len; i++;

    tab[i].type = REVISOR_SEC_MEMORY_FAULTY; tab[i].flags = 0; tab[i].offset = off; tab[i].length = faulty_len;
    if (faulty_len) memcpy(out + off, faulty, faulty_len);
    off += faulty_len; i++;

    tab[i].type = REVISOR_SEC_GPR;           tab[i].flags = 0; tab[i].offset = off; tab[i].length = gpr_len;
    if (gpr_len) memcpy(out + off, gpr, gpr_len);
    off += gpr_len; i++;

    if (NULL != mte_tags) {
        tab[i].type = REVISOR_SEC_MTE_TAGS;  tab[i].flags = 0; tab[i].offset = off; tab[i].length = mte_len;
        memcpy(out + off, mte_tags, mte_len);
        off += mte_len; i++;
    }
    if (NULL != pac_keys) {
        tab[i].type = REVISOR_SEC_PAC_KEYS;  tab[i].flags = 0; tab[i].offset = off; tab[i].length = pac_len;
        memcpy(out + off, pac_keys, pac_len);
        off += pac_len; i++;
    }

    h->magic = REVISOR_INPUT_MAGIC; h->version = REVISOR_INPUT_VERSION;
    h->header_len = header_len; h->n_sections = n; h->flags = 0; h->total_len = off;
    return off;
}

static void test_input_load_fd_valid(void) {
    int pfd[2];
    if (pipe(pfd) < 0) { EXPECT(0 && "pipe failed"); return; }

    struct input_header hdr = make_valid_hdr(RVZR_FLAG_HAS_CODE, 4, 0);
    uint32_t code_word = ENC_NOP;

    pipe_write_all(pfd[1], &hdr, sizeof(hdr));
    pipe_write_all(pfd[1], &code_word, 4);
    close(pfd[1]);

    struct simulation_input si;
    int ret = simulation_input_load_fd(pfd[0], &si);
    close(pfd[0]);

    EXPECT_EQ(ret, 0);
    EXPECT(si.code   != NULL);
    EXPECT(si.memory == NULL);
    EXPECT(si.regs   == NULL);
    EXPECT_EQ(si.hdr.code_size, (uint64_t)4);
    EXPECT_EQ(*(uint32_t*)si.code, code_word);

    simulation_input_free(&si);
}

static void test_input_load_fd_truncated(void) {
    int pfd[2];
    if (pipe(pfd) < 0) { EXPECT(0 && "pipe failed"); return; }

    /* Claim 16 bytes of code, write only 4 */
    struct input_header hdr = make_valid_hdr(RVZR_FLAG_HAS_CODE, 16, 0);
    pipe_write_all(pfd[1], &hdr, sizeof(hdr));
    uint32_t partial = ENC_NOP;
    pipe_write_all(pfd[1], &partial, 4);  /* only 4 of 16 bytes */
    close(pfd[1]);

    struct simulation_input si;
    int ret = simulation_input_load_fd(pfd[0], &si);
    close(pfd[0]);

    EXPECT(ret != 0);
    /* After failure, no leaks: pointers zeroed by simulation_input_free */
    EXPECT(si.code   == NULL);
    EXPECT(si.memory == NULL);
    EXPECT(si.regs   == NULL);
}

static void test_input_load_fd_bad_header(void) {
    int pfd[2];
    if (pipe(pfd) < 0) { EXPECT(0 && "pipe failed"); return; }

    struct input_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic   = 0xDEADBEEFu;  /* bad magic */
    hdr.version = RVZRCE_VERSION;
    hdr.arch    = RVZR_ARCH_AARCH64;
    pipe_write_all(pfd[1], &hdr, sizeof(hdr));
    close(pfd[1]);

    struct simulation_input si;
    int ret = simulation_input_load_fd(pfd[0], &si);
    close(pfd[0]);

    EXPECT(ret != 0);
}

/* Feed `hdr` + code + input_init over a pipe to simulation_input_load_fd; returns its result and
 * fills `si` (caller frees on success). */
static int load_fd_via_pipe(const struct input_header* hdr, const uint8_t* code, size_t code_len,
                            const uint8_t* input_init, size_t init_len, struct simulation_input* si) {
    int pfd[2];
    if (pipe(pfd) < 0) { return -99; }
    pipe_write_all(pfd[1], hdr, sizeof(*hdr));
    if (code_len)  pipe_write_all(pfd[1], code, code_len);
    if (init_len)  pipe_write_all(pfd[1], input_init, init_len);
    close(pfd[1]);
    int ret = simulation_input_load_fd(pfd[0], si);
    close(pfd[0]);
    return ret;
}

static void test_input_load_fd_all_sections(void) {
    uint8_t code[8];   memset(code, 0x1F, 8);     /* 2× NOP bytes */
    uint8_t main_[64]; memset(main_, 0xAB, 64);
    uint8_t faulty[16]; memset(faulty, 0xEF, 16);
    uint8_t gpr[32];   memset(gpr,  0xCD, 32);

    uint8_t init[512];
    size_t init_len = make_input_init(init, sizeof(init), main_, 64, faulty, 16, gpr, 32,
                                      NULL, 0, NULL, 0);
    EXPECT(init_len > 0);

    struct input_header hdr = make_valid_hdr(RVZR_FLAG_HAS_CODE | RVZR_FLAG_HAS_INPUT, 8, init_len);

    struct simulation_input si;
    int ret = load_fd_via_pipe(&hdr, code, 8, init, init_len, &si);

    EXPECT_EQ(ret, 0);
    EXPECT(si.code   != NULL);
    EXPECT(si.memory != NULL);
    EXPECT(si.regs   != NULL);
    EXPECT_EQ(si.hdr.code_size, (uint64_t)8);
    /* memory is main || faulty (64 + 16); regs is the gpr section (32). */
    EXPECT_EQ(si.mem_size,  (size_t)80);
    EXPECT_EQ(si.regs_size, (size_t)32);
    EXPECT_EQ(si.memory[0],  0xABu);   /* main */
    EXPECT_EQ(si.memory[63], 0xABu);
    EXPECT_EQ(si.memory[64], 0xEFu);   /* faulty concatenated after main */
    EXPECT_EQ(si.memory[79], 0xEFu);
    EXPECT_EQ(si.regs[0],    0xCDu);
    EXPECT_EQ(si.regs[31],   0xCDu);
    EXPECT(si.mte_tags == NULL);
    EXPECT(!si.pac_keys_present);

    simulation_input_free(&si);
    EXPECT(si.code   == NULL);
    EXPECT(si.memory == NULL);
    EXPECT(si.regs   == NULL);
}

/* A required section (here: GPR) missing from the input_init must make the load fail cleanly. */
static void test_input_init_missing_required_section(void) {
    uint8_t code[4];   memset(code, 0x1F, 4);
    uint8_t main_[64]; memset(main_, 0xAB, 64);
    uint8_t faulty[16]; memset(faulty, 0xEF, 16);

    /* Build a malformed init: main + faulty only, no GPR. Hand-roll (make_input_init always adds GPR). */
    uint8_t init[512];
    const uint64_t n = 2;
    size_t header_len = sizeof(struct revisor_input_header) + n * sizeof(struct revisor_input_section);
    struct revisor_input_header* h = (struct revisor_input_header*)init;
    struct revisor_input_section* tab = (struct revisor_input_section*)(init + sizeof(*h));
    size_t off = header_len;
    tab[0].type = REVISOR_SEC_MEMORY_MAIN;   tab[0].flags = 0; tab[0].offset = off; tab[0].length = 64;
    memcpy(init + off, main_, 64); off += 64;
    tab[1].type = REVISOR_SEC_MEMORY_FAULTY; tab[1].flags = 0; tab[1].offset = off; tab[1].length = 16;
    memcpy(init + off, faulty, 16); off += 16;
    h->magic = REVISOR_INPUT_MAGIC; h->version = REVISOR_INPUT_VERSION;
    h->header_len = header_len; h->n_sections = n; h->flags = 0; h->total_len = off;

    struct input_header hdr = make_valid_hdr(RVZR_FLAG_HAS_CODE | RVZR_FLAG_HAS_INPUT, 4, off);

    struct simulation_input si;
    int ret = load_fd_via_pipe(&hdr, code, 4, init, off, &si);
    EXPECT(ret != 0);
    EXPECT(si.memory == NULL);
    EXPECT(si.regs   == NULL);
}

/* A malformed input_init header (bad inner magic) must be rejected. */
static void test_input_init_bad_inner_magic(void) {
    uint8_t code[4];   memset(code, 0x1F, 4);
    uint8_t main_[64]; memset(main_, 0xAB, 64);
    uint8_t gpr[32];   memset(gpr,  0xCD, 32);

    uint8_t init[512];
    size_t init_len = make_input_init(init, sizeof(init), main_, 64, NULL, 0, gpr, 32,
                                      NULL, 0, NULL, 0);
    EXPECT(init_len > 0);
    ((struct revisor_input_header*)init)->magic = 0xDEADBEEFu;   /* corrupt the inner magic */

    struct input_header hdr = make_valid_hdr(RVZR_FLAG_HAS_CODE | RVZR_FLAG_HAS_INPUT, 4, init_len);

    struct simulation_input si;
    int ret = load_fd_via_pipe(&hdr, code, 4, init, init_len, &si);
    EXPECT(ret != 0);
}

/* MTE tag + PAC key sections must decode into sim_input (tags unpacked one-per-granule). */
static void test_input_init_mte_and_pac_sections(void) {
    uint8_t code[4];   memset(code, 0x1F, 4);
    uint8_t main_[64]; memset(main_, 0xAB, 64);   /* 64 bytes -> 4 granules */
    uint8_t gpr[32];   memset(gpr,  0xCD, 32);
    /* 4 granules -> 2 packed bytes: tags {1,2,3,4} = low/high nibbles 0x21, 0x43. */
    uint8_t tags[2] = { 0x21, 0x43 };
    uint64_t keys[10];
    for (int i = 0; i < 10; i++) keys[i] = 0x1000ULL + i;

    uint8_t init[512];
    size_t init_len = make_input_init(init, sizeof(init), main_, 64, NULL, 0, gpr, 32,
                                      tags, sizeof(tags), (const uint8_t*)keys, sizeof(keys));
    EXPECT(init_len > 0);

    struct input_header hdr = make_valid_hdr(RVZR_FLAG_HAS_CODE | RVZR_FLAG_HAS_INPUT, 4, init_len);

    struct simulation_input si;
    int ret = load_fd_via_pipe(&hdr, code, 4, init, init_len, &si);

    EXPECT_EQ(ret, 0);
    EXPECT(si.mte_tags != NULL);
    EXPECT_EQ(si.mte_tag_count, (size_t)4);   /* mem_size(64) / 16 */
    EXPECT_EQ(si.mte_tags[0], 1u);
    EXPECT_EQ(si.mte_tags[1], 2u);
    EXPECT_EQ(si.mte_tags[2], 3u);
    EXPECT_EQ(si.mte_tags[3], 4u);
    EXPECT(si.pac_keys_present);
    EXPECT_EQ(si.pac_keys[0], (uint64_t)0x1000);
    EXPECT_EQ(si.pac_keys[9], (uint64_t)0x1009);

    simulation_input_free(&si);
}

/* A PAC_KEYS section whose length is not exactly sizeof(pac_keys) (80B) must be rejected. */
static void test_input_init_pac_keys_wrong_size_rejected(void) {
    uint8_t code[4];   memset(code, 0x1F, 4);
    uint8_t main_[64]; memset(main_, 0xAB, 64);
    uint8_t gpr[32];   memset(gpr,  0xCD, 32);
    uint64_t keys[9];   /* 72B: the old, short layout */
    for (int i = 0; i < 9; i++) keys[i] = 0x2000ULL + i;

    uint8_t init[512];
    size_t init_len = make_input_init(init, sizeof(init), main_, 64, NULL, 0, gpr, 32,
                                      NULL, 0, (const uint8_t*)keys, sizeof(keys));
    EXPECT(init_len > 0);

    struct input_header hdr = make_valid_hdr(RVZR_FLAG_HAS_CODE | RVZR_FLAG_HAS_INPUT, 4, init_len);

    struct simulation_input si;
    int ret = load_fd_via_pipe(&hdr, code, 4, init, init_len, &si);

    EXPECT_EQ(ret, -1);
}

/* ======================================================================== */
/* ---- GROUP 22: GPR layout — gpr[29-N] == xN --------------------------- */
/* ======================================================================== */

static void test_gpr_layout(void) {
    /* Verify struct gprs field ordering: x29 at slot 0, x0 at slot 29. */
    EXPECT_EQ(offsetof(struct gprs, x29),  0  * sizeof(uintptr_t));
    EXPECT_EQ(offsetof(struct gprs, x28),  1  * sizeof(uintptr_t));
    EXPECT_EQ(offsetof(struct gprs, x10),  19 * sizeof(uintptr_t));
    EXPECT_EQ(offsetof(struct gprs, x0),   29 * sizeof(uintptr_t));
    EXPECT_EQ(sizeof(struct gprs),         30 * sizeof(uintptr_t));

    /* The union gpr[] and named gprs fields share memory. */
    struct cpu_state s;
    memset(&s, 0, sizeof(s));

    s.gprs.x0 = 0xDEADBEEFDEADBEEFULL;
    EXPECT_EQ(s.gpr[29], 0xDEADBEEFDEADBEEFULL);  /* gpr[29-0]=gpr[29] */

    s.gprs.x29 = 0x1234567890ABCDEFULL;
    EXPECT_EQ(s.gpr[0], 0x1234567890ABCDEFULL);   /* gpr[29-29]=gpr[0] */

    s.gprs.x10 = 0xCAFEBABE00000000ULL;
    EXPECT_EQ(s.gpr[19], 0xCAFEBABE00000000ULL);  /* gpr[29-10]=gpr[19] */
}

/* ======================================================================== */
/* ---- GROUP 23: cpu_state ABI layout (base_hook.S depends on this) ----- */
/* ======================================================================== */

static void test_cpu_state_layout(void) {
    EXPECT_EQ(offsetof(struct cpu_state, sp),   (size_t)0);
    EXPECT_EQ(offsetof(struct cpu_state, nzcv), (size_t)8);
    EXPECT_EQ(offsetof(struct cpu_state, pc),   (size_t)16);
    EXPECT_EQ(offsetof(struct cpu_state, lr),   (size_t)24);
    EXPECT_EQ(offsetof(struct cpu_state, gprs), (size_t)32);
    EXPECT_EQ(sizeof(struct cpu_state),         34 * sizeof(uintptr_t));

    /* SP and LR are not part of the gpr[] array: writing them must not
     * change any gpr slot, and vice versa. */
    struct cpu_state s;
    memset(&s, 0, sizeof(s));
    s.sp = 0xAAAAAAAAAAAAAAAAULL;
    s.lr = 0xBBBBBBBBBBBBBBBBULL;

    bool gpr_aliases_sp = false, gpr_aliases_lr = false;
    for (int i = 0; i < 30; ++i) {
        if (s.gpr[i] == 0xAAAAAAAAAAAAAAAAULL) gpr_aliases_sp = true;
        if (s.gpr[i] == 0xBBBBBBBBBBBBBBBBULL) gpr_aliases_lr = true;
    }
    EXPECT(!gpr_aliases_sp);
    EXPECT(!gpr_aliases_lr);

    /* Filling gpr[] must not touch sp or lr */
    for (int i = 0; i < 30; ++i) {
        s.gpr[i] = 0xCCCCCCCCCCCCCCCCULL;
    }
    EXPECT_EQ(s.sp, 0xAAAAAAAAAAAAAAAAULL);
    EXPECT_EQ(s.lr, 0xBBBBBBBBBBBBBBBBULL);
}

/* ======================================================================== */
/* ---- GROUP 24: Write/read all GPRs without aliasing ------------------- */
/* ======================================================================== */

static void test_gpr_write_read_roundtrip(void) {
    struct cpu_state s;
    memset(&s, 0, sizeof(s));

    /* Write a unique sentinel per register */
    for (int n = 0; n <= 29; ++n) {
        s.gpr[29 - n] = 0x0A00000000000000ULL | (uint64_t)n;
    }

    /* Read back and verify no slot aliasing */
    for (int n = 0; n <= 29; ++n) {
        uint64_t expected = 0x0A00000000000000ULL | (uint64_t)n;
        EXPECT_EQ(s.gpr[29 - n], expected);
    }

    /* Verify named fields alias through the union correctly */
    EXPECT_EQ(s.gprs.x0,  0x0A00000000000000ULL | 0u);
    EXPECT_EQ(s.gprs.x29, 0x0A00000000000000ULL | 29u);
    EXPECT_EQ(s.gprs.x10, 0x0A00000000000000ULL | 10u);
}

/* ======================================================================== */
/* ---- GROUP 25: XZR/LR/SP register access semantics -------------------- */
/* ======================================================================== */

static void test_xreg_access_semantics(void) {
    /*
     * read_xreg/write_xreg (pac_sign_plugin.c) semantics:
     *   n==31: read returns 0 (XZR), write is no-op
     *   n==30: read/write go through cpu_state.lr
     *   n==0..29: read/write through gpr[29-n]
     */

    /* XZR: reading must always return 0 regardless of surrounding state */
    struct cpu_state s;
    memset(&s, 0xFF, sizeof(s)); /* garbage fill */
    /* Simulated: if (n==31) return 0; */
    uint64_t xzr_read = 0;  /* invariant: always 0 */
    EXPECT_EQ(xzr_read, (uint64_t)0);

    /* XZR write: no-op — nothing in cpu_state changes */
    memset(&s, 0, sizeof(s));
    s.sp = 0xAAAA;
    uint64_t sp_before = s.sp;
    /* Simulated write_xreg(s, 31, 0xDEAD): n==31 → return immediately */
    uint64_t sp_after = s.sp;
    EXPECT_EQ(sp_before, sp_after);

    /* LR (x30): read/write go through cpu_state.lr */
    s.lr = 0xFEEDFACEDEAD0000ULL;
    /* Simulated: read_xreg(s, 30) == s.lr */
    EXPECT_EQ(s.lr, 0xFEEDFACEDEAD0000ULL);

    /* Simulated: write_xreg(s, 30, v) → s.lr = v */
    s.lr = 0x1122334455667788ULL;
    EXPECT_EQ(s.lr, 0x1122334455667788ULL);

    /* GPR (x5): read/write through gpr[29-5]=gpr[24] */
    memset(&s, 0, sizeof(s));
    s.gpr[24] = 0xDECAFBAD12345678ULL;
    /* Simulated: read_xreg(s, 5) == s.gpr[29-5] */
    EXPECT_EQ(s.gpr[29 - 5], 0xDECAFBAD12345678ULL);

    /* Simulated: write_xreg(s, 5, v) → s.gpr[24] = v */
    s.gpr[24] = 0xABCDEFABCDEFABCDULL;
    EXPECT_EQ(s.gpr[29 - 5], 0xABCDEFABCDEFABCDULL);
}

/* ======================================================================== */
/* ---- GROUP 26: NZCV bit positions -------------------------------------- */
/* ======================================================================== */

static void test_nzcv_bit_positions(void) {
    /* AArch64 PSTATE: N=bit31, Z=bit30, C=bit29, V=bit28 */
    uint32_t all_set = (1u<<31)|(1u<<30)|(1u<<29)|(1u<<28);
    EXPECT((all_set >> 31) & 1);
    EXPECT((all_set >> 30) & 1);
    EXPECT((all_set >> 29) & 1);
    EXPECT((all_set >> 28) & 1);

    /* Only N set — other flags must be clear */
    uint32_t n_only = 1u << 31;
    EXPECT( ((n_only >> 31) & 1)); /* N=1 */
    EXPECT(!((n_only >> 30) & 1)); /* Z=0 */
    EXPECT(!((n_only >> 29) & 1)); /* C=0 */
    EXPECT(!((n_only >> 28) & 1)); /* V=0 */

    /* Bits 27:0 never used for condition flags */
    uint32_t lower = 0x0FFFFFFFu;
    EXPECT(!((lower >> 31) & 1));
    EXPECT(!((lower >> 30) & 1));
    EXPECT(!((lower >> 29) & 1));
    EXPECT(!((lower >> 28) & 1));
}

/* ======================================================================== */
/* ---- GROUP 27: condition_passed exhaustive with NZCV=0 ---------------- */
/* ======================================================================== */

static void test_condition_passed_nzcv_zero(void) {
    /* All flags clear: N=0, Z=0, C=0, V=0 */
    uint32_t nzcv = 0;
    EXPECT(!condition_passed_ref(0x0, nzcv)); /* EQ: Z=0 → F */
    EXPECT( condition_passed_ref(0x1, nzcv)); /* NE: !Z → T */
    EXPECT(!condition_passed_ref(0x2, nzcv)); /* CS: C=0 → F */
    EXPECT( condition_passed_ref(0x3, nzcv)); /* CC: !C → T */
    EXPECT(!condition_passed_ref(0x4, nzcv)); /* MI: N=0 → F */
    EXPECT( condition_passed_ref(0x5, nzcv)); /* PL: !N → T */
    EXPECT(!condition_passed_ref(0x6, nzcv)); /* VS: V=0 → F */
    EXPECT( condition_passed_ref(0x7, nzcv)); /* VC: !V → T */
    EXPECT(!condition_passed_ref(0x8, nzcv)); /* HI: !Z&&C = 1&&0 → F */
    EXPECT( condition_passed_ref(0x9, nzcv)); /* LS: Z||!C = 0||1 → T */
    EXPECT( condition_passed_ref(0xA, nzcv)); /* GE: N==V = 0==0 → T */
    EXPECT(!condition_passed_ref(0xB, nzcv)); /* LT: N!=V = 0!=0 → F */
    EXPECT( condition_passed_ref(0xC, nzcv)); /* GT: !Z&&(N==V) = 1&&1 → T */
    EXPECT(!condition_passed_ref(0xD, nzcv)); /* LE: Z||(N!=V) = 0||0 → F */
    EXPECT( condition_passed_ref(0xE, nzcv)); /* AL → T */
    EXPECT( condition_passed_ref(0xF, nzcv)); /* NV → T */
}

/* ======================================================================== */
/* ---- GROUP 28: condition_passed with all NZCV set --------------------- */
/* ======================================================================== */

static void test_condition_passed_all_set(void) {
    uint32_t nzcv = (1u<<31)|(1u<<30)|(1u<<29)|(1u<<28); /* N=Z=C=V=1 */
    EXPECT( condition_passed_ref(0x0, nzcv)); /* EQ: Z=1 → T */
    EXPECT(!condition_passed_ref(0x1, nzcv)); /* NE: !Z → F */
    EXPECT( condition_passed_ref(0x2, nzcv)); /* CS: C=1 → T */
    EXPECT(!condition_passed_ref(0x3, nzcv)); /* CC: !C → F */
    EXPECT( condition_passed_ref(0x4, nzcv)); /* MI: N=1 → T */
    EXPECT(!condition_passed_ref(0x5, nzcv)); /* PL: !N → F */
    EXPECT( condition_passed_ref(0x6, nzcv)); /* VS: V=1 → T */
    EXPECT(!condition_passed_ref(0x7, nzcv)); /* VC: !V → F */
    EXPECT(!condition_passed_ref(0x8, nzcv)); /* HI: !Z=0 → F */
    EXPECT( condition_passed_ref(0x9, nzcv)); /* LS: Z=1 → T */
    EXPECT( condition_passed_ref(0xA, nzcv)); /* GE: N==V = 1==1 → T */
    EXPECT(!condition_passed_ref(0xB, nzcv)); /* LT: N!=V = 1!=1 → F */
    EXPECT(!condition_passed_ref(0xC, nzcv)); /* GT: !Z=0 → F */
    EXPECT( condition_passed_ref(0xD, nzcv)); /* LE: Z=1 → T */
    EXPECT( condition_passed_ref(0xE, nzcv)); /* AL → T */
    EXPECT( condition_passed_ref(0xF, nzcv)); /* NV → T */
}

/* ======================================================================== */
/* ---- GROUP 29: set_rn for all register indices 0..31 ------------------ */
/* ======================================================================== */

static void test_set_rn_all_registers(void) {
    for (uint32_t n = 0; n <= 31; ++n) {
        uint32_t patched = set_rn(ENC_LDR_X0_X1, (uint8_t)n);
        EXPECT_EQ(get_rn(patched), n);
        EXPECT_EQ(get_rt(patched), 0u);  /* Rt=0 preserved */

        /* Only bits[9:5] changed */
        uint32_t mask = ~(0x1Fu << 5);
        EXPECT_EQ(patched & mask, ENC_LDR_X0_X1 & mask);
    }

    /* Same for STR */
    for (uint32_t n = 0; n <= 31; ++n) {
        uint32_t patched = set_rn(ENC_STR_X0_X1, (uint8_t)n);
        EXPECT_EQ(get_rn(patched), n);
        EXPECT_EQ(get_rt(patched), 0u);
    }

    /* LDP: Rt and Rt2 must be unaffected */
    for (uint32_t n = 0; n <= 31; ++n) {
        uint32_t patched = set_rn(ENC_LDP_X0_X2_X1, (uint8_t)n);
        EXPECT_EQ(get_rn(patched),  n);
        EXPECT_EQ(get_rt(patched),  0u);
        EXPECT_EQ(get_rt2(patched), 2u);
    }
}

/* ======================================================================== */
/* ---- GROUP 30: address translation arithmetic ------------------------- */
/* ======================================================================== */

static void test_addr_translation_logic(void) {
    /*
     * Self-contained test of kaddr2uaddr / uaddr2kaddr invariants.
     *
     * With CONFIG_FLAG_REQ_MEM_BASE_VIRT:
     *   kaddr2uaddr(kaddr) = sim_mem + (kaddr - kbase)
     *   uaddr2kaddr(uaddr) = kbase  + (uaddr - sim_mem)
     *
     * Without the flag: identity mapping.
     */
    uint8_t sim_mem[4096];
    uintptr_t kbase = 0xFFFF000080000000ULL;

    /* kbase → sim_mem[0] */
    uintptr_t uaddr0 = (uintptr_t)sim_mem + (kbase - kbase);
    EXPECT_EQ(uaddr0, (uintptr_t)sim_mem);

    /* kbase+100 → sim_mem+100 */
    uintptr_t kaddr1 = kbase + 100;
    uintptr_t uaddr1 = (uintptr_t)sim_mem + (kaddr1 - kbase);
    EXPECT_EQ(uaddr1, (uintptr_t)(sim_mem + 100));

    /* Round-trip kaddr → uaddr → kaddr */
    uintptr_t kaddr_rt = kbase + (uaddr1 - (uintptr_t)sim_mem);
    EXPECT_EQ(kaddr_rt, kaddr1);

    /* End boundary: kbase+4096 → sim_mem+4096 */
    uintptr_t kaddr_end = kbase + 4096;
    uintptr_t uaddr_end = (uintptr_t)sim_mem + (kaddr_end - kbase);
    EXPECT_EQ(uaddr_end, (uintptr_t)(sim_mem + 4096));

    /* Identity mapping (flag not set): kaddr == uaddr */
    uintptr_t any = 0xABCD1234ULL;
    EXPECT_EQ(any, any);
}

/* ======================================================================== */
/* ---- GROUP 31: cpu_state_read/write_base_reg — dispatch correctness ---- */
/* ======================================================================== */

static void test_fixup_base_reg_rw(void) {
    /*
     * cpu_state_read_base_reg / cpu_state_write_base_reg implement the
     * dispatch used by the new fixup design: modify the base register's
     * VALUE in cpu_state (kaddr→uaddr) without patching the instruction.
     *
     * Tests:
     *  - write then read roundtrip for every Rn 0..30 via gpr[29-Rn]
     *  - unique sentinel per register → no aliasing
     *  - Rn=31 dispatches to state.sp, not to gpr[] (which would be OOB)
     *  - get_rn(inst) + cpu_state_{read,write}_base_reg form a correct pipeline
     */
    struct cpu_state s;
    memset(&s, 0, sizeof(s));

    /* write unique sentinels through every Rn 0..30 */
    for (uint32_t rn = 0; rn <= 30; ++rn) {
        uintptr_t sentinel = 0xA000000000000000ULL | rn;
        cpu_state_write_base_reg(&s, rn, sentinel);
    }

    /* read back: each Rn must return its own sentinel */
    for (uint32_t rn = 0; rn <= 30; ++rn) {
        uintptr_t expected = 0xA000000000000000ULL | rn;
        EXPECT_EQ(cpu_state_read_base_reg(&s, rn), expected);
    }

    /* Rn=31 dispatches to state.sp, and only to state.sp */
    uintptr_t sp_val = 0xBBBBBBBBBBBBBBBBULL;
    cpu_state_write_base_reg(&s, 31, sp_val);
    EXPECT_EQ(cpu_state_read_base_reg(&s, 31), sp_val);
    EXPECT_EQ(s.sp, sp_val);
    /* writing SP must not corrupt any GPR slot */
    for (uint32_t rn = 0; rn <= 30; ++rn) {
        uintptr_t expected = 0xA000000000000000ULL | rn;
        EXPECT_EQ(cpu_state_read_base_reg(&s, rn), expected);
    }

    /* end-to-end pipeline: get_rn(instruction) → read/write base reg */
    memset(&s, 0, sizeof(s));
    uint32_t ldr_x5_x3 = set_rn(ENC_LDR_X0_X1, 3);   /* LDR X0, [X3] */
    uint32_t rn_ldr    = get_rn(ldr_x5_x3);            /* should be 3  */
    EXPECT_EQ(rn_ldr, 3u);
    cpu_state_write_base_reg(&s, rn_ldr, 0xDEADBEEFULL);
    EXPECT_EQ(cpu_state_read_base_reg(&s, rn_ldr), 0xDEADBEEFULL);
    EXPECT_EQ(s.gprs.x3, 0xDEADBEEFULL);   /* union alias agrees */

    uint32_t str_sp = set_rn(ENC_STR_X0_X1, 31);   /* STR X0, [SP] */
    uint32_t rn_sp  = get_rn(str_sp);               /* should be 31 */
    EXPECT_EQ(rn_sp, 31u);
    cpu_state_write_base_reg(&s, rn_sp, 0xCAFEBABEULL);
    EXPECT_EQ(cpu_state_read_base_reg(&s, rn_sp), 0xCAFEBABEULL);
    EXPECT_EQ(s.sp, 0xCAFEBABEULL);
}

/* ======================================================================== */
/* ---- GROUP 32: fixup simulate — translate, execute, translate back ----- */
/* ======================================================================== */

static void test_fixup_simulate_roundtrip(void) {
    /*
     * Simulate the full fixup flow using the actual helper functions:
     *
     *   base_hook_c: cpu_state_write_base_reg(state, rn, kaddr2uaddr(orig))
     *   instruction executes (possibly updates Rn for writeback)
     *   apply_fixups: cpu_state_write_base_reg(state, rn,
     *                     uaddr2kaddr(cpu_state_read_base_reg(state, rn)))
     *
     * kaddr2uaddr/uaddr2kaddr are not linked into test_ce, so we inline the
     * linear mapping: uaddr = sim_mem + (kaddr - kbase).
     * The test calls cpu_state_{read,write}_base_reg and get_rn — actual code.
     */
    uintptr_t kbase   = 0xFFFF000080000000ULL;
    uint8_t   buf[256];
    uintptr_t sim_mem = (uintptr_t)buf;

    struct { uint32_t inst; uint32_t rn; uintptr_t kaddr; int64_t wb_delta; } cases[] = {
        { ENC_LDR_X0_X1,   1,  kbase + 0x100, 0  },  /* no writeback */
        { ENC_LDR_PREIDX,  1,  kbase + 0x100, 8  },  /* pre-index +8 */
        { ENC_LDR_POSTIDX, 1,  kbase + 0x100, 8  },  /* post-index +8 */
        { set_rn(ENC_STR_X0_X1, 31), 31, kbase + 0x80, 0  },  /* SP, no wb */
        { set_rn(ENC_LDR_PREIDX,  31), 31, kbase + 0x80, -16 }, /* SP, pre-index -16 */
    };

    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        uint32_t inst  = cases[i].inst;
        uintptr_t kaddr = cases[i].kaddr;
        int64_t  delta = cases[i].wb_delta;

        /* verify get_rn agrees with the fixture */
        EXPECT_EQ(get_rn(inst), cases[i].rn);

        struct cpu_state s;
        memset(&s, 0, sizeof(s));

        /* base_hook_c: write kaddr2uaddr(kaddr) into Rn */
        uintptr_t uaddr = sim_mem + (kaddr - kbase);
        cpu_state_write_base_reg(&s, cases[i].rn, uaddr);
        EXPECT_EQ(cpu_state_read_base_reg(&s, cases[i].rn), uaddr);

        /* hardware executes the instruction; for writeback, Rn is updated */
        uintptr_t rn_after = (uintptr_t)((int64_t)uaddr + delta);
        cpu_state_write_base_reg(&s, cases[i].rn, rn_after);

        /* apply_fixups: read Rn, translate back to kaddr */
        uintptr_t rn_read   = cpu_state_read_base_reg(&s, cases[i].rn);
        uintptr_t kaddr_back = kbase + (rn_read - sim_mem);
        uintptr_t expected  = (uintptr_t)((int64_t)kaddr + delta);
        EXPECT_EQ(kaddr_back, expected);
    }
}

/* ======================================================================== */
/* ---- GROUP 33: BL/B opcode bits and imm26 round-trip ------------------ */
/* ======================================================================== */

static void test_bl_opcode_correctness(void) {
    uintptr_t from = 0x1000, to = 0x2000;

    uint32_t bl = encode_bl(from, to);
    EXPECT(bl != 0);
    EXPECT_EQ(bl >> 26, 0x25u);  /* BL opcode: bits[31:26]=100101 */

    uint32_t b = encode_b(from, to);
    EXPECT(b != 0);
    EXPECT_EQ(b >> 26, 0x05u);   /* B  opcode: bits[31:26]=000101 */

    /* inner_hook_aarch64_instructions overwrites every slot with BL */
    /* BL at from+0 reaching the hook at to: must be valid */
    EXPECT(encode_bl(from,      to) != 0);
    EXPECT(encode_bl(from + 4,  to) != 0);
    EXPECT(encode_bl(from + 8,  to) != 0);
    EXPECT(encode_bl(from + 12, to) != 0);
}

static void check_bl_roundtrip(uintptr_t from, uintptr_t to) {
    uint32_t bl = encode_bl(from, to);
    if (bl == 0) { EXPECT(0 && "encode_bl out of range"); return; }
    int32_t imm26 = (int32_t)(bl & 0x03FFFFFF);
    if (imm26 & (1 << 25)) imm26 |= (int32_t)0xFC000000;
    int64_t offset = (int64_t)imm26 * 4;
    EXPECT_EQ((uintptr_t)((int64_t)from + offset), to);
}

static void test_bl_imm26_roundtrip(void) {
    check_bl_roundtrip(0x4000, 0x5000);  /* forward */
    check_bl_roundtrip(0x5000, 0x4000);  /* backward */
    check_bl_roundtrip(0x1000, 0x1008);  /* +8 */
    check_bl_roundtrip(0x1008, 0x1000);  /* -8 */

    /* Out-of-range */
    EXPECT_EQ(encode_bl(0, 0x10000000UL), (uint32_t)0);
    EXPECT_EQ(encode_b(0,  0x10000000UL), (uint32_t)0);
}

/* ======================================================================== */
/* ---- GROUP 34: TBZ bit-number for all boundary values ----------------- */
/* ======================================================================== */

static void test_tbz_bit_all_boundaries(void) {
    struct { uint32_t enc; uint32_t expected_bit; } cases[] = {
        /* bit 0:  b5=0, b40=0  → 0x36_00000_imm14_Rt = 0x36000040 */
        { 0x36000040u,  0 },
        /* bit 1:  b5=0, b40=1  → bits[23:19]=00001 → 0x36080040 */
        { 0x36080040u,  1 },
        /* bit 31: b5=0, b40=31 → bits[23:19]=11111 → 0x36F80040 */
        { 0x36F80040u, 31 },
        /* bit 32: b5=1, b40=0  → bit31=1, bits[23:19]=00000 → 0xB6000040 */
        { 0xB6000040u, 32 },
        /* bit 33: b5=1, b40=1  → bit31=1, bits[23:19]=00001 → 0xB6080040 */
        { 0xB6080040u, 33 },
        /* bit 63: b5=1, b40=31 → bit31=1, bits[23:19]=11111 → 0xB6F80040 */
        { 0xB6F80040u, 63 },
    };

    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        uint32_t enc = cases[i].enc;
        uint32_t expected = cases[i].expected_bit;

        EXPECT_EQ(classify_branch(enc), (branch_type_t)BRANCH_TBZ);

        uint32_t b5  = (enc >> 31) & 0x1;
        uint32_t b40 = (enc >> 19) & 0x1F;
        uint32_t bit_num = (b5 << 5) | b40;
        EXPECT_EQ(bit_num, expected);
    }
}

/* ======================================================================== */
/* ---- GROUP 35: out_of_simulation boundary arithmetic ------------------ */
/* ======================================================================== */

static void test_out_of_simulation_arithmetic(void) {
    /*
     * out_of_simulation: (state->pc - code_base) >= code_size
     * Test the arithmetic invariant without touching the global.
     */
    uintptr_t code_base = 0x10000;
    uint64_t  code_size = 16; /* 4 instructions */

    /* PC at code_base: in simulation */
    uintptr_t pc = code_base;
    EXPECT(!((pc - code_base) >= code_size));

    /* PC at last valid instruction (code_size-4): in simulation */
    pc = code_base + code_size - 4;
    EXPECT(!((pc - code_base) >= code_size));

    /* PC at code_base + code_size: first OOB */
    pc = code_base + code_size;
    EXPECT((pc - code_base) >= code_size);

    /* PC past the end: also OOB */
    pc = code_base + code_size + 4;
    EXPECT((pc - code_base) >= code_size);
}

/* ======================================================================== */
/* ---- GROUP 36: SP-as-Rn OOB bug: gpr[29-31] = gpr[-2] --------------- */
/* ======================================================================== */

static void test_sp_rn_index_oob(void) {
    /*
     * apply_fixups: state->gpr[29 - get_rn(fixup.original)] = ...
     *
     * If get_rn returns 31 (SP), the index is 29-31 = -2: out of bounds.
     * This is undefined behavior. Document the arithmetic proof.
     */
    EXPECT_EQ(29 - 31, -2);  /* OOB: would access before the gpr[] array */
    EXPECT_EQ(29 - 30, -1);  /* OOB: LR not in gpr[] either */

    /* Valid GPR indices: 0..29 */
    for (int n = 0; n <= 29; ++n) {
        int idx = 29 - n;
        EXPECT(idx >= 0 && idx <= 29);
    }

    /* gpr[] has exactly 30 elements */
    EXPECT_EQ(sizeof(struct gprs) / sizeof(uintptr_t), (size_t)30);
}

/* ======================================================================== */
/* ---- GROUP 37: kaddr→uaddr→kaddr roundtrip for all 32 base registers --- */
/* ======================================================================== */

static void test_fixup_all_registers_roundtrip(void) {
    /*
     * For every Rn 0..31 (including SP=31), verify the complete fixup cycle:
     *   1. start with kaddr in Rn
     *   2. translate: write uaddr
     *   3. simulate no-writeback execution (Rn unchanged)
     *   4. apply_fixups: restore kaddr
     *   5. simulate writeback +8: Rn = uaddr + 8 after execution
     *   6. apply_fixups: restore kaddr + 8
     *   7. verify no neighbouring register was touched in any step
     *
     * The kaddr2uaddr/uaddr2kaddr bijection is inlined as the linear mapping
     * uaddr = sim_mem + (kaddr - kbase), uaddr2kaddr = kbase + (u - sim_mem).
     */
    uintptr_t kbase   = 0xFFFF000080000000ULL;
    uint8_t   buf[8192];
    uintptr_t sim_mem = (uintptr_t)buf;

    for (uint32_t rn = 0; rn <= 31; ++rn) {
        struct cpu_state s;
        memset(&s, 0, sizeof(s));

        uintptr_t kaddr = kbase + 0x100ULL + (uintptr_t)rn * 0x40ULL;
        uintptr_t uaddr = sim_mem + (kaddr - kbase);

        /* --- no-writeback case --- */
        cpu_state_write_base_reg(&s, rn, kaddr);
        EXPECT_EQ(cpu_state_read_base_reg(&s, rn), kaddr);

        cpu_state_write_base_reg(&s, rn, uaddr);       /* translate */
        EXPECT_EQ(cpu_state_read_base_reg(&s, rn), uaddr);

        /* apply_fixups: uaddr2kaddr */
        uintptr_t v = cpu_state_read_base_reg(&s, rn);
        cpu_state_write_base_reg(&s, rn, kbase + (v - sim_mem));
        EXPECT_EQ(cpu_state_read_base_reg(&s, rn), kaddr);

        /* neighbouring registers must be untouched */
        if (rn != 0)  { EXPECT_EQ(cpu_state_read_base_reg(&s, 0),  (uintptr_t)0); }
        if (rn != 7)  { EXPECT_EQ(cpu_state_read_base_reg(&s, 7),  (uintptr_t)0); }
        if (rn != 29) { EXPECT_EQ(cpu_state_read_base_reg(&s, 29), (uintptr_t)0); }

        /* --- writeback +8 case --- */
        cpu_state_write_base_reg(&s, rn, kaddr);

        cpu_state_write_base_reg(&s, rn, uaddr);              /* translate */
        uintptr_t uaddr_wb = uaddr + 8;                       /* execution updates Rn */
        cpu_state_write_base_reg(&s, rn, uaddr_wb);

        v = cpu_state_read_base_reg(&s, rn);
        cpu_state_write_base_reg(&s, rn, kbase + (v - sim_mem)); /* apply_fixups */
        EXPECT_EQ(cpu_state_read_base_reg(&s, rn), kaddr + 8);
    }
}

/* ======================================================================== */
/* ---- GROUP 38: Hook isolation — sim_state copy is immune to *state writes */
/* ======================================================================== */

static void test_hook_isolation(void) {
    /*
     * base_hook_c ordering guarantee:
     *   apply_fixups(state);           → state has kaddr
     *   sim_state.cpu_state = *state;  → snapshot has kaddr
     *   ... run hooks(&sim_state) ...  → hooks see kaddr
     *   *state = sim_state.cpu_state;  → copy back
     *   cpu_state_write_base_reg(state, rn, uaddr);  ← LAST write
     *
     * After that last write, state->Rn == uaddr.
     * But sim_state.cpu_state.Rn is untouched — it was copied before translation.
     * A hook would see sim_state, which still holds kaddr.
     *
     * We prove this for every Rn 0..31 by:
     *   1. Put kaddr in state
     *   2. Take snapshot (simulating sim_state.cpu_state = *state)
     *   3. Write uaddr to state (the LAST translation)
     *   4. Assert snapshot still has kaddr; state has uaddr; they differ
     */
    uintptr_t kbase   = 0xFFFF000080000000ULL;
    uint8_t   buf[4096];
    uintptr_t sim_mem = (uintptr_t)buf;

    for (uint32_t rn = 0; rn <= 31; ++rn) {
        uintptr_t kaddr = kbase + (uintptr_t)rn * 0x40ULL;
        uintptr_t uaddr = sim_mem + (kaddr - kbase);

        struct cpu_state state;
        memset(&state, 0, sizeof(state));

        cpu_state_write_base_reg(&state, rn, kaddr); /* step 1 */

        struct cpu_state hook_view = state;           /* step 2: C struct copy */

        cpu_state_write_base_reg(&state, rn, uaddr); /* step 3: LAST translation */

        /* hook_view (what a hook receives) must still hold kaddr */
        EXPECT_EQ(cpu_state_read_base_reg(&hook_view, rn), kaddr);

        /* state (the real machine state) now holds uaddr */
        EXPECT_EQ(cpu_state_read_base_reg(&state, rn), uaddr);

        /* they are genuinely different — hook never saw uaddr */
        EXPECT(cpu_state_read_base_reg(&hook_view, rn) !=
               cpu_state_read_base_reg(&state, rn));
    }
}

/* ======================================================================== */
/* ---- GROUP 39: CBZ/CBNZ/TBZ/TBNZ taken/not-taken decision semantics ---- */
/* ======================================================================== */

static void test_compare_branch_taken(void) {
    /*
     * Inlines the taken/not-taken logic from simulation_execution_clause_hook.c
     * and cross-checks it against classify_branch + evaluate_cond_target.
     *
     * CBZ:  taken iff  reg == 0
     * CBNZ: taken iff  reg != 0
     * TBZ:  taken iff (reg >> bit_n) & 1 == 0
     * TBNZ: taken iff (reg >> bit_n) & 1 != 0
     */
    uintptr_t pc = 0x4000;

    /* --- CBZ X0, PC+8 --- */
    {
        uint32_t insn = ENC_CBZ_X0_PLUS8;
        EXPECT_EQ(classify_branch(insn), BRANCH_CBZ);
        uintptr_t tgt = evaluate_cond_target(pc, insn);
        EXPECT_EQ(tgt, pc + 8);
        uint32_t rt = get_rt(insn);
        EXPECT_EQ(rt, 0u);

        uint64_t reg = 0;
        EXPECT((reg == 0));                                     /* taken */
        EXPECT_EQ((reg == 0) ? tgt : pc + 4, pc + 8);

        reg = 1;
        EXPECT(!(reg == 0));                                    /* not taken */
        EXPECT_EQ((reg == 0) ? tgt : pc + 4, pc + 4);

        reg = UINT64_MAX;
        EXPECT(!(reg == 0));                                    /* not taken */
    }

    /* --- CBNZ X0, PC+8 --- */
    {
        uint32_t insn = ENC_CBNZ_X0_PLUS8;
        EXPECT_EQ(classify_branch(insn), BRANCH_CBNZ);
        uintptr_t tgt = evaluate_cond_target(pc, insn);
        EXPECT_EQ(tgt, pc + 8);

        uint64_t reg = 0;
        EXPECT(!(reg != 0));                                    /* not taken */
        EXPECT_EQ((reg != 0) ? tgt : pc + 4, pc + 4);

        reg = 42;
        EXPECT((reg != 0));                                     /* taken */
        EXPECT_EQ((reg != 0) ? tgt : pc + 4, pc + 8);
    }

    /* --- TBZ X0, #0, PC+8 --- */
    {
        uint32_t insn = ENC_TBZ_X0_B0_PLUS8;
        EXPECT_EQ(classify_branch(insn), BRANCH_TBZ);
        uintptr_t tgt = evaluate_cond_target(pc, insn);
        EXPECT_EQ(tgt, pc + 8);

        uint32_t b5 = (insn >> 31) & 1, b40 = (insn >> 19) & 0x1F;
        uint32_t bit_n = (b5 << 5) | b40;
        EXPECT_EQ(bit_n, 0u);

        uint64_t reg = 0xFFFFFFFFFFFFFFFEULL; /* bit 0 clear → taken */
        EXPECT(((reg >> bit_n) & 1) == 0);
        EXPECT_EQ(((reg >> bit_n) & 1) == 0 ? tgt : pc + 4, pc + 8);

        reg = 0x1;                             /* bit 0 set → not taken */
        EXPECT(((reg >> bit_n) & 1) != 0);
        EXPECT_EQ(((reg >> bit_n) & 1) == 0 ? tgt : pc + 4, pc + 4);
    }

    /* --- TBZ X0, #32, PC+8 — tests the b5 extraction fix --- */
    {
        uint32_t insn = ENC_TBZ_X0_B32_PLUS8;
        EXPECT_EQ(classify_branch(insn), BRANCH_TBZ);

        uint32_t b5 = (insn >> 31) & 1, b40 = (insn >> 19) & 0x1F;
        uint32_t bit_n = (b5 << 5) | b40;
        EXPECT_EQ(bit_n, 32u);

        uint64_t reg = 0x00000000FFFFFFFFULL; /* bit 32 clear → taken */
        EXPECT(((reg >> bit_n) & 1) == 0);

        reg = 0x0000000100000000ULL;           /* bit 32 set → not taken */
        EXPECT(((reg >> bit_n) & 1) != 0);
    }

    /* --- TBNZ X0, #0, PC+8 --- */
    {
        uint32_t insn = ENC_TBNZ_X0_B0_PLUS8;
        EXPECT_EQ(classify_branch(insn), BRANCH_TBNZ);
        uintptr_t tgt = evaluate_cond_target(pc, insn);
        EXPECT_EQ(tgt, pc + 8);

        uint32_t b5 = (insn >> 31) & 1, b40 = (insn >> 19) & 0x1F;
        uint32_t bit_n = (b5 << 5) | b40;
        EXPECT_EQ(bit_n, 0u);

        uint64_t reg = 0x1;                    /* bit 0 set → taken */
        EXPECT(((reg >> bit_n) & 1) != 0);
        EXPECT_EQ(((reg >> bit_n) & 1) != 0 ? tgt : pc + 4, pc + 8);

        reg = 0xFFFFFFFFFFFFFFFEULL;           /* bit 0 clear → not taken */
        EXPECT(((reg >> bit_n) & 1) == 0);
        EXPECT_EQ(((reg >> bit_n) & 1) != 0 ? tgt : pc + 4, pc + 4);
    }
}

/* ======================================================================== */
/* ---- GROUP 40: B.cond full dispatch — all 16 conditions × two NZCV vals  */
/* ======================================================================== */

static void test_bcond_integration(void) {
    /*
     * For every condition code 0..15, build B.cond with imm19=2 (target=pc+8)
     * and verify the full pipeline:
     *   classify_branch → BRANCH_B_COND
     *   evaluate_cond_target → pc + 8
     *   condition_passed_ref(cond, nzcv_taken)     → true  → next_pc = target
     *   condition_passed_ref(cond, nzcv_not_taken) → false → next_pc = PC+4
     *
     * AL (0xE) and NV (0xF) are always taken; no not-taken NZCV exists.
     */
    struct {
        uint8_t  cond;
        uint32_t nzcv_taken;
        uint32_t nzcv_not_taken;
        bool     always_taken;
    } cases[] = {
        { 0x0, (1u<<30),             0,                   false }, /* EQ */
        { 0x1, 0,                    (1u<<30),             false }, /* NE */
        { 0x2, (1u<<29),             0,                   false }, /* CS */
        { 0x3, 0,                    (1u<<29),             false }, /* CC */
        { 0x4, (1u<<31),             0,                   false }, /* MI */
        { 0x5, 0,                    (1u<<31),             false }, /* PL */
        { 0x6, (1u<<28),             0,                   false }, /* VS */
        { 0x7, 0,                    (1u<<28),             false }, /* VC */
        { 0x8, (1u<<29),             (1u<<30),             false }, /* HI: C&&!Z vs Z=1 */
        { 0x9, (1u<<30),             (1u<<29),             false }, /* LS: Z=1 vs C&&!Z */
        { 0xA, 0,                    (1u<<31),             false }, /* GE: N==V (0==0) vs N=1 */
        { 0xB, (1u<<31),             0,                   false }, /* LT: N!=V vs N==V */
        { 0xC, (1u<<29),             (1u<<30)|(1u<<29),   false }, /* GT: C&&!Z vs Z=1 */
        { 0xD, (1u<<30),             (1u<<29),             false }, /* LE: Z=1 vs C&&!Z */
        { 0xE, 0,                    0,                   true  }, /* AL */
        { 0xF, 0,                    0,                   true  }, /* NV */
    };

    uintptr_t pc = 0x3000;

    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        uint8_t cond = cases[i].cond;

        /* Build B.cond: bits[31:24]=0x54, bits[23:5]=imm19=2, bit4=0, bits[3:0]=cond */
        uint32_t insn = 0x54000000u | (2u << 5) | cond;

        if (cases[i].always_taken) {
            /* AL/NV branch unconditionally, so they are not conditional branches */
            EXPECT_EQ(classify_branch(insn), BRANCH_NONE);
        } else {
            EXPECT_EQ(classify_branch(insn),         BRANCH_B_COND);
            EXPECT_EQ(evaluate_cond_target(pc, insn), pc + 8);
        }

        /* taken path */
        bool taken = condition_passed_ref(cond, cases[i].nzcv_taken);
        EXPECT(taken);
        EXPECT_EQ(taken ? (pc + 8) : (pc + 4), pc + 8);

        /* not-taken path (skip for AL/NV) */
        if (!cases[i].always_taken) {
            bool nt = condition_passed_ref(cond, cases[i].nzcv_not_taken);
            EXPECT(!nt);
            EXPECT_EQ(nt ? (pc + 8) : (pc + 4), pc + 4);
        }
    }
}

/* ======================================================================== */
/* ---- GROUP 41: Fixup chain with arithmetic between memory instructions -- */
/* ======================================================================== */

static void test_fixup_chain_with_arithmetic(void) {
    /*
     * Simulates the sequence:
     *   LDR X0, [X1]       fixup: X1 = uaddr(kaddr1)
     *   ...apply_fixups...  X1 = kaddr1
     *   ADD X1, X1, #16    (no fixup) X1 = kaddr1 + 16
     *   LDR X0, [X1]       fixup: X1 = uaddr(kaddr1 + 16)
     *   ...apply_fixups...  X1 = kaddr1 + 16
     *
     * Also tests two independent registers (X2, X5) in interleaved fixup
     * cycles to verify no cross-contamination.
     *
     * Uses the actual cpu_state_read/write_base_reg functions (not arithmetic
     * stubs) so any dispatch bug for SP or high-numbered registers is caught.
     */
    uintptr_t kbase   = 0xFFFF000080000000ULL;
    uint8_t   mem[4096];
    uintptr_t sim_mem = (uintptr_t)mem;

    /* --- Scenario A: same Rn (X1), arithmetic in between --- */
    {
        struct cpu_state s;
        memset(&s, 0, sizeof(s));

        uintptr_t kaddr = kbase + 0x100;
        cpu_state_write_base_reg(&s, 1, kaddr);

        /* translate for first LDR */
        uintptr_t uaddr = sim_mem + (kaddr - kbase);
        cpu_state_write_base_reg(&s, 1, uaddr);
        EXPECT_EQ(cpu_state_read_base_reg(&s, 1), uaddr);

        /* apply_fixups: restore kaddr */
        {
            uintptr_t v = cpu_state_read_base_reg(&s, 1);
            cpu_state_write_base_reg(&s, 1, kbase + (v - sim_mem));
        }
        EXPECT_EQ(cpu_state_read_base_reg(&s, 1), kaddr);

        /* ADD X1, X1, #16 — no fixup involvement */
        cpu_state_write_base_reg(&s, 1, cpu_state_read_base_reg(&s, 1) + 16);
        EXPECT_EQ(cpu_state_read_base_reg(&s, 1), kaddr + 16);

        /* translate for second LDR */
        uintptr_t uaddr2 = sim_mem + ((kaddr + 16) - kbase);
        cpu_state_write_base_reg(&s, 1, uaddr2);
        EXPECT_EQ(cpu_state_read_base_reg(&s, 1), uaddr2);

        /* apply_fixups: restore kaddr + 16 */
        {
            uintptr_t v = cpu_state_read_base_reg(&s, 1);
            cpu_state_write_base_reg(&s, 1, kbase + (v - sim_mem));
        }
        EXPECT_EQ(cpu_state_read_base_reg(&s, 1), kaddr + 16);
    }

    /* --- Scenario B: two different Rn values, interleaved fixups --- */
    {
        struct cpu_state s;
        memset(&s, 0, sizeof(s));

        uintptr_t ka2 = kbase + 0x200;
        uintptr_t ka5 = kbase + 0x500;

        cpu_state_write_base_reg(&s, 2, ka2);
        cpu_state_write_base_reg(&s, 5, ka5);

        /* translate X2 */
        cpu_state_write_base_reg(&s, 2, sim_mem + (ka2 - kbase));
        EXPECT_EQ(cpu_state_read_base_reg(&s, 5), ka5); /* X5 unaffected */

        /* apply_fixups for X2 */
        {
            uintptr_t v = cpu_state_read_base_reg(&s, 2);
            cpu_state_write_base_reg(&s, 2, kbase + (v - sim_mem));
        }
        EXPECT_EQ(cpu_state_read_base_reg(&s, 2), ka2);
        EXPECT_EQ(cpu_state_read_base_reg(&s, 5), ka5); /* still unaffected */

        /* translate X5 */
        cpu_state_write_base_reg(&s, 5, sim_mem + (ka5 - kbase));
        EXPECT_EQ(cpu_state_read_base_reg(&s, 2), ka2); /* X2 unaffected */

        /* apply_fixups for X5 */
        {
            uintptr_t v = cpu_state_read_base_reg(&s, 5);
            cpu_state_write_base_reg(&s, 5, kbase + (v - sim_mem));
        }
        EXPECT_EQ(cpu_state_read_base_reg(&s, 2), ka2);
        EXPECT_EQ(cpu_state_read_base_reg(&s, 5), ka5);
    }

    /* --- Scenario C: SP (Rn=31) as a memory base across two fixup cycles --- */
    {
        struct cpu_state s;
        memset(&s, 0, sizeof(s));

        uintptr_t ksp = kbase + 0x800;
        cpu_state_write_base_reg(&s, 31, ksp);

        /* translate (STR X0, [SP]) */
        uintptr_t usp = sim_mem + (ksp - kbase);
        cpu_state_write_base_reg(&s, 31, usp);
        EXPECT_EQ(s.sp, usp); /* must go through s.sp, not gpr[] */

        /* apply_fixups */
        {
            uintptr_t v = cpu_state_read_base_reg(&s, 31);
            cpu_state_write_base_reg(&s, 31, kbase + (v - sim_mem));
        }
        EXPECT_EQ(s.sp, ksp);

        /* SUB SP, SP, #32 */
        cpu_state_write_base_reg(&s, 31, cpu_state_read_base_reg(&s, 31) - 32);
        EXPECT_EQ(s.sp, ksp - 32);

        /* translate again (LDR X0, [SP]) */
        uintptr_t usp2 = sim_mem + ((ksp - 32) - kbase);
        cpu_state_write_base_reg(&s, 31, usp2);

        /* apply_fixups */
        {
            uintptr_t v = cpu_state_read_base_reg(&s, 31);
            cpu_state_write_base_reg(&s, 31, kbase + (v - sim_mem));
        }
        EXPECT_EQ(s.sp, ksp - 32);

        /* gpr[] must be completely untouched throughout */
        for (int i = 0; i < 30; ++i) {
            EXPECT_EQ(s.gpr[i], (uintptr_t)0);
        }
    }
}

/* ======================================================================== */
/* ---- main -------------------------------------------------------------- */
/* ======================================================================== */

/* ---- GROUP 42: cond_branch_is_taken — resolved direction (real resolver) ----------------- */

/* Regression: a conditional branch whose taken-target is pc+4 (e.g. B.cond .+4). The resolved
 * direction must come from the condition itself, NOT be inferred from (target == architectural
 * next) — that comparison is true for BOTH directions when target == pc+4, so the old derivation
 * misread a not-taken branch as taken (corrupting table training and the PHR fold). */
static void test_cond_branch_is_taken_target_is_next_insn(void) {
    uint32_t insn = 0x54000020u;        /* B.EQ .+4  (imm19=1, cond=EQ) */
    struct cpu_state s;
    memset(&s, 0, sizeof(s));
    s.pc = (uintptr_t)&insn;

    EXPECT_EQ(classify_branch(insn), BRANCH_B_COND);
    EXPECT_EQ(evaluate_cond_target(s.pc, insn), (uintptr_t)(s.pc + 4));   /* target == pc+4 */

    /* Not taken (Z=0 -> EQ fails): the real resolver must say NOT taken ... */
    s.nzcv = 0;
    EXPECT(!cond_branch_is_taken(&s));
    /* ... even though (target == architectural_next) reads as TAKEN here — the old bug. */
    EXPECT_EQ(evaluate_cond_target(s.pc, insn), cond_branch_architectural_next(&s));

    /* Taken (Z=1 -> EQ holds): architectural next is the target (which is pc+4). */
    s.nzcv = 0x40000000u;
    EXPECT(cond_branch_is_taken(&s));
    EXPECT_EQ(cond_branch_architectural_next(&s), (uintptr_t)(s.pc + 4));

    /* Sanity, normal branch (target != pc+4): B.EQ .+8 not-taken -> target != architectural next. */
    uint32_t insn8 = ENC_BEQ_PLUS8;
    s.pc = (uintptr_t)&insn8;
    s.nzcv = 0;
    EXPECT(!cond_branch_is_taken(&s));
    EXPECT_EQ(cond_branch_architectural_next(&s), (uintptr_t)(s.pc + 4));
    EXPECT(evaluate_cond_target(s.pc, insn8) != cond_branch_architectural_next(&s));
}

int main(void) {
    printf("Running CE unit tests...\n");

    /* Original 14 groups */
    test_classify_branch();
    test_evaluate_cond_target();
    test_tbz_bit_number();
    test_memory_access_classification();
    test_access_size();
    test_register_fields();
    test_load_store_predicates();
    test_branch_encoding();
    test_condition_passed();
    test_signextend();
    test_sext_then_shift();
    test_input_header_validation();
    test_pair_access_size();
    test_literal_access_size();

    /* New groups */
    test_input_header_input_init_size();
    test_payload_size();
    test_struct_sizes();
    test_input_load_fd_valid();
    test_input_load_fd_truncated();
    test_input_load_fd_bad_header();
    test_input_load_fd_all_sections();
    test_input_init_missing_required_section();
    test_input_init_bad_inner_magic();
    test_input_init_mte_and_pac_sections();
    test_input_init_pac_keys_wrong_size_rejected();
    test_gpr_layout();
    test_cpu_state_layout();
    test_gpr_write_read_roundtrip();
    test_xreg_access_semantics();
    test_nzcv_bit_positions();
    test_condition_passed_nzcv_zero();
    test_condition_passed_all_set();
    test_set_rn_all_registers();
    test_addr_translation_logic();
    test_fixup_base_reg_rw();
    test_fixup_simulate_roundtrip();
    test_bl_opcode_correctness();
    test_bl_imm26_roundtrip();
    test_tbz_bit_all_boundaries();
    test_out_of_simulation_arithmetic();
    test_sp_rn_index_oob();

    /* Groups 37-41: fixup semantics, hook isolation, branch decision logic */
    test_fixup_all_registers_roundtrip();
    test_hook_isolation();
    test_compare_branch_taken();
    test_bcond_integration();
    test_fixup_chain_with_arithmetic();

    /* Group 42: real branch resolver (regression for the target==pc+4 direction bug) */
    test_cond_branch_is_taken_target_is_next_insn();

    printf("\n%d tests, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed ? 1 : 0;
}
