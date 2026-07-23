/*
 * Unit tests for the architected QARMA3/QARMA5 pointer-auth (qarma.c). Pure software (no device).
 * The known-answer vector is the real Pixel 8 hardware output (pacia under a fixed key), so a match
 * proves the software model is bit-exact with QARMA5 hardware.
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
static const struct pac_profile QARMA5 = { .iterations = 4, .tsz = 25, .tbi = 1, .pauth2 = 1 };
static const struct pac_profile QARMA3 = { .iterations = 2, .tsz = 16, .tbi = 1, .pauth2 = 1 };

/* pacia 0x0000000012345000, ctx 0x1122334455667788, key {lo=KLO,hi=KHI} on a Pixel 8 (VA=39). */
static void test_qarma5_matches_pixel_hardware(void)
{
    uint64_t got = qarma_addpac(0x0000000012345000ull, 0x1122334455667788ull, KLO, KHI, QARMA5);
    CHECK(got == 0x002a920012345000ull, "got %016llx want 002a920012345000", (unsigned long long)got);
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
    test_qarma5_matches_pixel_hardware();
    test_sign_then_strip_roundtrips();
    test_context_and_key_sensitivity();
    test_qarma3_differs_from_qarma5();
    printf("%s\n", g_fail ? "QARMA TESTS FAILED" : "all qarma tests passed");
    return g_fail != 0;
}
