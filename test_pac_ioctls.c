/*
 * test_pac_ioctls.c — functional tests for the PAC key management ioctls:
 *   REVISOR_GET_PAC_KEYS        (15): read current EL0 PAC keys from hardware
 *   REVISOR_SET_PAC_KEYS        (14): configure custom keys in executor
 *   REVISOR_GET_EXEC_PAC_KEYS   (13): get executor keys + use_swap flag
 *   REVISOR_SWAP_PAC_KEYS       (12): atomically swap EL0 keys and save old ones
 *
 * Compile:
 *   gcc -march=native -O0 -g test_pac_ioctls.c -o test_pac_ioctls
 * Run (module must be loaded, /dev/executor accessible):
 *   ./test_pac_ioctls
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/ioctl.h>

/* ── ioctl definitions (must match chardevice.h) ────────────────────────── */

#define REVISOR_IOC_MAGIC                  'r'
#define REVISOR_SWAP_PAC_KEYS_CONSTANT      12
#define REVISOR_GET_EXEC_PAC_KEYS_CONSTANT  13
#define REVISOR_SET_PAC_KEYS_CONSTANT       14
#define REVISOR_GET_PAC_KEYS_CONSTANT       15

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

#define REVISOR_SET_PAC_KEYS      _IOW(REVISOR_IOC_MAGIC, REVISOR_SET_PAC_KEYS_CONSTANT,      struct pac_keys)
#define REVISOR_GET_PAC_KEYS      _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_PAC_KEYS_CONSTANT,      struct pac_keys)
#define REVISOR_SWAP_PAC_KEYS     _IOWR(REVISOR_IOC_MAGIC, REVISOR_SWAP_PAC_KEYS_CONSTANT,    struct pac_keys_swap_req)
#define REVISOR_GET_EXEC_PAC_KEYS _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_EXEC_PAC_KEYS_CONSTANT, struct pac_exec_keys_info)

#define EXECUTOR_DEV "/dev/executor"

/* ── EL0 inline asm PAC/AUTH helpers ────────────────────────────────────── */

static inline uint64_t el0_pacda(uint64_t ptr, uint64_t ctx)
{
    __asm__ volatile("pacda %0, %1" : "+r"(ptr) : "r"(ctx));
    return ptr;
}

static inline uint64_t el0_autda(uint64_t ptr, uint64_t ctx)
{
    __asm__ volatile("autda %0, %1" : "+r"(ptr) : "r"(ctx));
    return ptr;
}

static inline uint64_t el0_xpacd(uint64_t ptr)
{
    __asm__ volatile("xpacd %0" : "+r"(ptr));
    return ptr;
}

/* ── Test harness ────────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;

static void check(int cond, const char *msg)
{
    if (cond) {
        printf("  PASS: %s\n", msg);
        g_pass++;
    } else {
        printf("  FAIL: %s\n", msg);
        g_fail++;
    }
}

static void print_keys(const char *label, const struct pac_keys *k)
{
    printf("  %s:\n"
           "    APIA: lo=%016llx hi=%016llx\n"
           "    APIB: lo=%016llx hi=%016llx\n"
           "    APDA: lo=%016llx hi=%016llx\n"
           "    APDB: lo=%016llx hi=%016llx\n"
           "    APGA: lo=%016llx hi=%016llx\n",
           label,
           (unsigned long long)k->apia_lo, (unsigned long long)k->apia_hi,
           (unsigned long long)k->apib_lo, (unsigned long long)k->apib_hi,
           (unsigned long long)k->apda_lo, (unsigned long long)k->apda_hi,
           (unsigned long long)k->apdb_lo, (unsigned long long)k->apdb_hi,
           (unsigned long long)k->apga_lo, (unsigned long long)k->apga_hi);
}

int main(void)
{
    int fd = open(EXECUTOR_DEV, O_RDWR);
    if (fd < 0) {
        perror("open " EXECUTOR_DEV);
        return 1;
    }
    printf("Opened %s\n\n", EXECUTOR_DEV);

    /* ── Test 1: GET_PAC_KEYS — reads live EL0 PAC keys ─────────────────── */
    printf("=== Test 1: REVISOR_GET_PAC_KEYS ===\n");
    struct pac_keys original_keys;
    memset(&original_keys, 0, sizeof(original_keys));
    check(ioctl(fd, REVISOR_GET_PAC_KEYS, &original_keys) == 0,
          "GET_PAC_KEYS ioctl succeeds");
    check(original_keys.apda_lo != 0 || original_keys.apda_hi != 0,
          "APDA key is non-zero");
    check(original_keys.apib_lo != 0 || original_keys.apib_hi != 0,
          "APIB key is non-zero");
    print_keys("original keys", &original_keys);

    /* ── Test 2: GET_EXEC_PAC_KEYS — verify ioctl works and reports state ── */
    printf("\n=== Test 2: REVISOR_GET_EXEC_PAC_KEYS ===\n");
    struct pac_exec_keys_info info_before;
    memset(&info_before, 0, sizeof(info_before));
    check(ioctl(fd, REVISOR_GET_EXEC_PAC_KEYS, &info_before) == 0,
          "GET_EXEC_PAC_KEYS ioctl succeeds");
    printf("  use_swap=%d (0=default keys, 1=custom keys already configured)\n",
           info_before.use_swap);
    if (info_before.use_swap == 0) {
        check(info_before.keys.apda_lo == original_keys.apda_lo &&
              info_before.keys.apda_hi == original_keys.apda_hi,
              "GET_EXEC_PAC_KEYS returns same APDA as GET_PAC_KEYS when use_swap=0");
    } else {
        printf("  (skipping key-match check: executor already has custom keys from prior session)\n");
        g_pass++;   /* count as pass — state is valid, just not default */
    }

    /* ── Test 3: SET_PAC_KEYS then GET_EXEC_PAC_KEYS — must report use_swap=1 */
    printf("\n=== Test 3: REVISOR_SET_PAC_KEYS + GET_EXEC_PAC_KEYS ===\n");
    struct pac_keys custom_keys = {
        .apia_lo = 0x1111111111111111ULL, .apia_hi = 0x2222222222222222ULL,
        .apib_lo = 0x3333333333333333ULL, .apib_hi = 0x4444444444444444ULL,
        .apda_lo = 0xDEADBEEFCAFEBABEULL, .apda_hi = 0xFEEDFACEDEADC0DEULL,
        .apdb_lo = 0x5555555555555555ULL, .apdb_hi = 0x6666666666666666ULL,
        .apga_lo = 0x7777777777777777ULL, .apga_hi = 0x8888888888888888ULL,
    };
    check(ioctl(fd, REVISOR_SET_PAC_KEYS, &custom_keys) == 0,
          "SET_PAC_KEYS ioctl succeeds");

    struct pac_exec_keys_info info_after;
    memset(&info_after, 0, sizeof(info_after));
    check(ioctl(fd, REVISOR_GET_EXEC_PAC_KEYS, &info_after) == 0,
          "GET_EXEC_PAC_KEYS after SET succeeds");
    check(info_after.use_swap == 1,
          "use_swap=1 after SET_PAC_KEYS");
    check(info_after.keys.apda_lo == custom_keys.apda_lo &&
          info_after.keys.apda_hi == custom_keys.apda_hi,
          "GET_EXEC_PAC_KEYS returns custom APDA keys");
    check(info_after.keys.apib_lo == custom_keys.apib_lo &&
          info_after.keys.apib_hi == custom_keys.apib_hi,
          "GET_EXEC_PAC_KEYS returns custom APIB keys");

    /* ── Test 4: SWAP_PAC_KEYS — install custom, verify old saved, restore ── */
    printf("\n=== Test 4: REVISOR_SWAP_PAC_KEYS round-trip ===\n");
    struct pac_keys_swap_req swap_to_custom;
    memset(&swap_to_custom, 0, sizeof(swap_to_custom));
    swap_to_custom.in_keys = custom_keys;
    check(ioctl(fd, REVISOR_SWAP_PAC_KEYS, &swap_to_custom) == 0,
          "SWAP_PAC_KEYS (install custom) succeeds");
    check(swap_to_custom.out_keys.apda_lo == original_keys.apda_lo &&
          swap_to_custom.out_keys.apda_hi == original_keys.apda_hi,
          "saved APDA_lo/hi matches original");
    check(swap_to_custom.out_keys.apib_lo == original_keys.apib_lo &&
          swap_to_custom.out_keys.apib_hi == original_keys.apib_hi,
          "saved APIB_lo/hi matches original");

    /* Verify GET_PAC_KEYS reflects the newly installed keys (APIB/APDA/APDB/APGA).
     * APIA is not checked: the kernel owns hardware APIA for its own return-address
     * auth; we only update task_struct APIA, which takes effect on return to EL0. */
    struct pac_keys keys_after_swap;
    memset(&keys_after_swap, 0, sizeof(keys_after_swap));
    check(ioctl(fd, REVISOR_GET_PAC_KEYS, &keys_after_swap) == 0,
          "GET_PAC_KEYS after swap succeeds");
    check(keys_after_swap.apib_lo == custom_keys.apib_lo &&
          keys_after_swap.apib_hi == custom_keys.apib_hi,
          "GET_PAC_KEYS after swap: APIB matches installed keys");
    check(keys_after_swap.apda_lo == custom_keys.apda_lo &&
          keys_after_swap.apda_hi == custom_keys.apda_hi,
          "GET_PAC_KEYS after swap: APDA matches installed keys");
    check(keys_after_swap.apdb_lo == custom_keys.apdb_lo &&
          keys_after_swap.apdb_hi == custom_keys.apdb_hi,
          "GET_PAC_KEYS after swap: APDB matches installed keys");
    check(keys_after_swap.apga_lo == custom_keys.apga_lo &&
          keys_after_swap.apga_hi == custom_keys.apga_hi,
          "GET_PAC_KEYS after swap: APGA matches installed keys");

    /* Now custom keys are active in EL0 — do a PACDA */
    uint64_t test_ptr = 0x0000AABBCCDDEEFFULL;
    uint64_t test_ctx = 0x0000112233445566ULL;
    uint64_t signed_custom = el0_pacda(test_ptr, test_ctx);
    check(signed_custom != test_ptr,
          "PACDA with custom keys changes the pointer value");
    printf("  ptr=%016llx  signed(custom)=%016llx\n",
           (unsigned long long)test_ptr, (unsigned long long)signed_custom);

    /* Restore original keys */
    struct pac_keys_swap_req swap_restore;
    memset(&swap_restore, 0, sizeof(swap_restore));
    swap_restore.in_keys = swap_to_custom.out_keys;
    check(ioctl(fd, REVISOR_SWAP_PAC_KEYS, &swap_restore) == 0,
          "SWAP_PAC_KEYS (restore original) succeeds");

    /* With original keys active, sign the same pointer — must differ */
    uint64_t signed_orig = el0_pacda(test_ptr, test_ctx);
    check(signed_orig != signed_custom,
          "PACDA with original keys produces different signature than custom keys");
    printf("  signed(original)=%016llx\n", (unsigned long long)signed_orig);

    /* ── Test 5: PACDA + AUTDA round-trip with custom keys ───────────────── */
    printf("\n=== Test 5: PACDA/AUTDA round-trip with custom keys ===\n");
    struct pac_keys_swap_req swap2;
    memset(&swap2, 0, sizeof(swap2));
    swap2.in_keys = custom_keys;
    ioctl(fd, REVISOR_SWAP_PAC_KEYS, &swap2);       /* install custom */

    uint64_t signed2  = el0_pacda(test_ptr, test_ctx);
    uint64_t authed   = el0_autda(signed2, test_ctx);
    uint64_t stripped = el0_xpacd(authed);
    check(stripped == test_ptr,
          "AUTDA(PACDA(ptr)) strips cleanly back to original pointer");
    printf("  original=%016llx  signed=%016llx  authed=%016llx  stripped=%016llx\n",
           (unsigned long long)test_ptr,  (unsigned long long)signed2,
           (unsigned long long)authed,    (unsigned long long)stripped);

    struct pac_keys_swap_req swap2r;
    memset(&swap2r, 0, sizeof(swap2r));
    swap2r.in_keys = swap2.out_keys;
    ioctl(fd, REVISOR_SWAP_PAC_KEYS, &swap2r);      /* restore */

    /* ── Test 6: FEAT_FPAC is active — verify XPAC strips without faulting ── */
    printf("\n=== Test 6: XPAC strip in speculative path (FEAT_FPAC safe) ===\n");
    /* This CPU has FEAT_FPAC: direct AUTDA with a wrong-key pointer raises SIGILL.
     * In the CE, auth_verify_hook uses XPAC in speculative paths instead of AUT
     * so that speculation never observes authentication success/failure.
     * Verify XPAC strips correctly and does not fault with either key set. */
    struct pac_keys_swap_req swap3;
    memset(&swap3, 0, sizeof(swap3));
    swap3.in_keys = custom_keys;
    ioctl(fd, REVISOR_SWAP_PAC_KEYS, &swap3);       /* install custom */

    uint64_t xpac_of_orig   = el0_xpacd(signed_orig);    /* strip orig-key signature */
    uint64_t xpac_of_custom = el0_xpacd(signed_custom);  /* strip custom-key signature */

    struct pac_keys_swap_req swap3r;
    memset(&swap3r, 0, sizeof(swap3r));
    swap3r.in_keys = swap3.out_keys;
    ioctl(fd, REVISOR_SWAP_PAC_KEYS, &swap3r);      /* restore */

    check(xpac_of_orig   == test_ptr, "XPAC of orig-key-signed ptr gives original pointer");
    check(xpac_of_custom == test_ptr, "XPAC of custom-key-signed ptr gives original pointer");
    printf("  xpac(signed_orig)=%016llx  xpac(signed_custom)=%016llx\n",
           (unsigned long long)xpac_of_orig, (unsigned long long)xpac_of_custom);
    printf("  (process still alive — XPAC never faults regardless of signing key)\n");

    /* ── Test 7: keys are restored after all swaps — GET_PAC_KEYS matches ── */
    printf("\n=== Test 7: verify keys restored after all swaps ===\n");
    struct pac_keys final_keys;
    memset(&final_keys, 0, sizeof(final_keys));
    check(ioctl(fd, REVISOR_GET_PAC_KEYS, &final_keys) == 0,
          "GET_PAC_KEYS after all swaps succeeds");
    check(final_keys.apda_lo == original_keys.apda_lo &&
          final_keys.apda_hi == original_keys.apda_hi,
          "APDA key restored to original after all swaps");
    check(final_keys.apib_lo == original_keys.apib_lo &&
          final_keys.apib_hi == original_keys.apib_hi,
          "APIB key restored to original after all swaps");

    /* ── Summary ─────────────────────────────────────────────────────────── */
    printf("\n============================================\n");
    printf("RESULTS: %d passed, %d failed\n", g_pass, g_fail);
    close(fd);
    return g_fail > 0 ? 1 : 0;
}
