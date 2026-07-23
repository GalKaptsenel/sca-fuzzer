/*
 * Unit tests for the architected QARMA3/QARMA5 pointer-auth (qarma.c). Pure software (no device).
 * The known-answer vectors are real hardware output (pacia under a fixed key), so a match proves the
 * software model is bit-exact with QARMA5 hardware.
 *
 * Build: make test_qarma   Run: ./test_qarma
 */
#include <stdint.h>
#include <stdio.h>

#include "qarma.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL %s:%d ", __func__, __LINE__); \
                              printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)

/* Fixed test key (APIA). */
static const uint64_t KLO = 0x0123456789abcdefull, KHI = 0xfedcba9876543210ull;
static const uint64_t CTX = 0x1122334455667788ull;
static const struct pac_profile QARMA5 = { .iterations = 4, .tsz = 25, .tbi = 1, .pauth2 = 1 };
static const struct pac_profile QARMA3 = { .iterations = 2, .tsz = 16, .tbi = 1, .pauth2 = 1 };

/* pacia outputs measured on real hardware (VA=39). HW selects TBI per pointer (bit 55): a low-half
 * (user) pointer uses TBI on (tbi=1); high-half (kernel) pointers use TBI off (tbi=0). The executor
 * signs kernel pointers -- tbi=1 there yields a wrong signature that FPAC-faults, so each vector must
 * match under its own TBI and differ under the other. */
static void test_qarma5_matches_hardware(void)
{
    const struct { uint64_t ptr, want; int tbi; } vec[] = {
        { 0x0000000012345000ull, 0x002a920012345000ull, 1 },   /* user,   TBI on  */
        { 0xffffffc012345000ull, 0xb1e67d4012345000ull, 0 },   /* kernel, TBI off */
        { 0xffffff8000abc000ull, 0xd5a28d0000abc000ull, 0 },   /* kernel, TBI off */
    };
    for (unsigned i = 0; i < sizeof(vec) / sizeof(vec[0]); ++i) {
        struct pac_profile right = { .iterations = 4, .tsz = 25, .tbi = vec[i].tbi, .pauth2 = 1 };
        struct pac_profile wrong = { .iterations = 4, .tsz = 25, .tbi = 1 - vec[i].tbi, .pauth2 = 1 };
        uint64_t g = qarma_addpac(vec[i].ptr, CTX, KLO, KHI, right);
        CHECK(g == vec[i].want, "ptr %016llx tbi=%d got %016llx want %016llx",
              (unsigned long long)vec[i].ptr, vec[i].tbi, (unsigned long long)g,
              (unsigned long long)vec[i].want);
        CHECK(qarma_addpac(vec[i].ptr, CTX, KLO, KHI, wrong) != vec[i].want,
              "ptr %016llx: wrong TBI unexpectedly matched", (unsigned long long)vec[i].ptr);
    }
}

/* Signing sets only the PAC field; stripping recovers the canonical pointer exactly. */
static void test_sign_then_strip_roundtrips(void)
{
    uint64_t user = 0x0000000000abc000ull;             /* canonical low-half pointer (VA=39) */
    for (uint64_t ctx = 0; ctx < 4; ++ctx) {
        uint64_t s5 = qarma_addpac(user, ctx, KLO, KHI, QARMA5);
        CHECK(qarma_strip(s5, QARMA5) == user, "QARMA5 strip ctx=%llu", (unsigned long long)ctx);
        uint64_t s3 = qarma_addpac(user, ctx, KLO, KHI, QARMA3);
        CHECK(qarma_strip(s3, QARMA3) == user, "QARMA3 strip ctx=%llu", (unsigned long long)ctx);
    }
}

/* A wrong context yields a different signature (so a decoy fails auth); signing is deterministic. */
static void test_context_and_key_sensitivity(void)
{
    uint64_t user = 0x0000000000abc000ull;
    uint64_t a = qarma_addpac(user, 0x11, KLO, KHI, QARMA5);
    uint64_t b = qarma_addpac(user, 0x12, KLO, KHI, QARMA5);
    uint64_t c = qarma_addpac(user, 0x11, KLO ^ 1, KHI, QARMA5);
    CHECK(a != b, "wrong-context sig collides");
    CHECK(a != c, "wrong-key sig collides");
    CHECK(a == qarma_addpac(user, 0x11, KLO, KHI, QARMA5), "sign not deterministic");
    CHECK(qarma_strip(a, QARMA5) == qarma_strip(b, QARMA5), "sigs strip to different pointers");
}

/* QARMA3 and QARMA5 are distinct algorithms (different round counts -> different signatures). */
static void test_qarma3_differs_from_qarma5(void)
{
    uint64_t user = 0x0000000000abc000ull;
    CHECK(qarma_addpac(user, 0x11, KLO, KHI, QARMA3) != qarma_addpac(user, 0x11, KLO, KHI, QARMA5),
          "QARMA3 == QARMA5");
}

int main(void)
{
    test_qarma5_matches_hardware();
    test_sign_then_strip_roundtrips();
    test_context_and_key_sensitivity();
    test_qarma3_differs_from_qarma5();
    printf("%s\n", g_fail ? "QARMA TESTS FAILED" : "all qarma tests passed");
    return g_fail != 0;
}
