/*
 * fr_verify.c — independent userspace Flush+Reload verification harness.
 *
 * A small, self-contained cross-check of the kernel executor's F+R result. It uses NEITHER
 * /dev/executor NOR the PMU: it runs the *same* raw test-case body and the *same* REIF input in
 * userspace and decides cache hit/miss purely from a timer (cntvct_el0, 1 GHz here).
 *
 * Method (per measured set, in its OWN fresh run — "per-set isolation"):
 *   1. flush the whole sandbox (dc civac)                 -> nothing resident
 *   2. set x0..x5 from the input, x29 = main base, run the real test-case bytes
 *   3. time-reload ONLY this set's line; faster than threshold == resident
 * Probing all 64 sets in one sweep trips the region prefetcher (it fills the whole small sandbox once
 * it sees a dense stream), so each set gets its own run: the reload never emits a walkable stream.
 * The body is deterministic, so a set measured on its own run reflects the same execution as a joint
 * sweep would. Timing is dependency-ordered (a branch on the loaded value gates the 2nd timestamp) so
 * we measure only the target load's latency and do NOT drain in-flight prefetches with a dsb.
 *
 * Layout mirrors the kernel: main and faulty are embedded in overflow-padding pages
 * [ pad | main(4K) | faulty(4K) | pad ], x29 = main base, so the prefetcher's spatial fetches spill
 * into unmeasured padding. set = (addr - x29)/64 mod 64. main and faulty are reported both jointly
 * (OR = the kernel htrace bit) and separately (which physical page each set's residency came from).
 *
 * Build: aarch64-linux-gnu-gcc -static -O2 -o fr_verify fr_verify.c
 * Run:   ./fr_verify <tc.bin> <input.reif> <n_reps>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sched.h>
#include <sys/mman.h>

#define PAGE            4096u
#define LINE            64u
#define NSETS           64u
#define NPAD            2u                          /* padding pages on each side of main|faulty */
#define MAIN_OFF        (NPAD * PAGE)               /* x29 = sandbox + MAIN_OFF */
#define FAULTY_OFF      (MAIN_OFF + PAGE)
#define SANDBOX_SIZE    ((2u + 2u * NPAD) * PAGE)

#define REIF_MAGIC      0x49525A5652ULL
#define SEC_MAIN        0x01
#define SEC_FAULTY      0x02
#define SEC_GPR         0x03

static inline uint64_t rd_u64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }

/* dependency-ordered timed load: a branch on the loaded value orders the second timestamp after the
 * load, without a dsb draining every in-flight prefetch. Returns cntvct delta (ns at 1 GHz). */
static inline uint64_t time_load(const volatile void* addr) {
    uint64_t t0, t1, v;
    __asm__ __volatile__(
        "isb\n mrs %[t0], cntvct_el0\n isb\n"
        "ldr %[v], [%[a]]\n"
        "cbnz %[v], 1f\n 1:\n"                       /* consume the load result -> orders t1 */
        "isb\n mrs %[t1], cntvct_el0\n"
        : [t0]"=&r"(t0), [t1]"=&r"(t1), [v]"=&r"(v)
        : [a]"r"(addr)
        : "memory", "cc");
    return t1 - t0;
}

static inline void flush_line(const volatile void* addr) {
    __asm__ __volatile__("dc civac, %0" :: "r"(addr) : "memory");
}

static inline unsigned bitrev6(unsigned x) {
    unsigned r = 0;
    for (int i = 0; i < 6; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

/* trampoline: x0=body, x1=base(->x29), x2=&regs[0..5]; runs body, which ends by branching to the
 * RET we appended right after it. */
extern void call_body(void* body, uint64_t base, const uint64_t* regs);
__asm__(
    ".text\n.globl call_body\n.type call_body,%function\n"
    "call_body:\n"
    "  stp x29, x30, [sp, #-16]!\n"
    "  mov x9, x0\n"
    "  mov x29, x1\n"
    "  ldr x3, [x2, #24]\n"
    "  ldr x4, [x2, #32]\n"
    "  ldr x5, [x2, #40]\n"
    "  ldp x0, x1, [x2]\n"
    "  ldr x2, [x2, #16]\n"
    "  blr x9\n"
    "  ldp x29, x30, [sp], #16\n"
    "  ret\n");

static const uint8_t* find_section(const uint8_t* f, size_t n, uint64_t want, uint64_t* len_out) {
    if (n < 48 || rd_u64(f) != REIF_MAGIC) { fprintf(stderr, "bad REIF magic\n"); exit(1); }
    uint64_t n_sec = rd_u64(f + 24);
    for (uint64_t i = 0; i < n_sec; ++i) {
        const uint8_t* d = f + 48 + i * 32;
        if (rd_u64(d) == want) {
            uint64_t off = rd_u64(d + 16), len = rd_u64(d + 24);
            if (off + len > n) { fprintf(stderr, "REIF section out of range\n"); exit(1); }
            *len_out = len;
            return f + off;
        }
    }
    fprintf(stderr, "REIF section 0x%lx missing\n", (unsigned long)want);
    exit(1);
}

static uint8_t* read_file(const char* path, size_t* sz) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(1); }
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    uint8_t* b = malloc(n);
    if (fread(b, 1, n, fp) != (size_t)n) { perror("fread"); exit(1); }
    fclose(fp);
    *sz = n;
    return b;
}

static int cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return (x > y) - (x < y);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <tc.bin> <input.reif> <n_reps>\n", argv[0]); return 1; }
    uint64_t n_reps = strtoull(argv[3], NULL, 0);

    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
    sched_setaffinity(0, sizeof(set), &set);

    /* test-case body -> executable page, RET appended so its `b.al 0x40` returns */
    size_t tc_len = 0;
    uint8_t* tc = read_file(argv[1], &tc_len);
    uint8_t* code = mmap(NULL, PAGE, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memcpy(code, tc, tc_len);
    uint32_t ret_insn = 0xd65f03c0;
    memcpy(code + tc_len, &ret_insn, 4);
    __builtin___clear_cache((char*)code, (char*)code + tc_len + 4);

    /* REIF input: registers + memory content */
    size_t reif_sz = 0;
    uint8_t* reif = read_file(argv[2], &reif_sz);
    uint64_t glen = 0, mlen = 0, flen = 0;
    const uint8_t* gpr = find_section(reif, reif_sz, SEC_GPR, &glen);
    const uint8_t* main_c = find_section(reif, reif_sz, SEC_MAIN, &mlen);
    const uint8_t* faulty_c = find_section(reif, reif_sz, SEC_FAULTY, &flen);
    uint64_t regs[6];
    for (int i = 0; i < 6; ++i) { regs[i] = rd_u64(gpr + i * 8); }

    /* padded sandbox; x29 = main base */
    uint8_t* sandbox = mmap(NULL, SANDBOX_SIZE, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint8_t* mainp = sandbox + MAIN_OFF;
    uint8_t* faultp = sandbox + FAULTY_OFF;
    uint64_t base = (uint64_t)mainp;
    memcpy(mainp,  main_c,   mlen < PAGE ? mlen : PAGE);
    memcpy(faultp, faulty_c, flen < PAGE ? flen : PAGE);

    printf("[i] tc=%zuB  x0..x5 =", tc_len);
    for (int i = 0; i < 6; ++i) { printf(" 0x%lx", (unsigned long)regs[i]); }
    printf("\n[i] main@%p faulty@%p  x29=main  n_reps=%lu  (dep-ordered cntvct timing)\n",
           mainp, faultp, (unsigned long)n_reps);

    /* calibrate hit/miss on a scratch line */
    uint8_t* scratch = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    const int CAL = 4000;
    uint64_t* hh = malloc(CAL * 8); uint64_t* mm = malloc(CAL * 8);
    volatile uint8_t sink = 0; (void)sink;
    for (int i = 0; i < CAL; ++i) {
        sink = scratch[0]; hh[i] = time_load(scratch);
        flush_line(scratch); __asm__ __volatile__("dsb sy\nisb\n" ::: "memory");
        mm[i] = time_load(scratch);
    }
    qsort(hh, CAL, 8, cmp_u64); qsort(mm, CAL, 8, cmp_u64);
    uint64_t hit_med = hh[CAL / 2], miss_med = mm[CAL / 2];
    uint64_t thresh = (hit_med + miss_med) / 2;
    printf("[i] calibration: hit_median=%lu  miss_median=%lu  -> threshold=%lu ns\n",
           (unsigned long)hit_med, (unsigned long)miss_med, (unsigned long)thresh);

    /* Per-set isolation. On this N3 the data prefetcher warms the whole small sandbox once the body
     * touches it, so a threshold hit/miss over-counts. MEAN reload latency is robust: the truly
     * resident (most-recently-used) line stays a few ns below the prefetched-L1 floor. We report both:
     * mean latency (primary) and threshold hit-rate (secondary), per page. */
    double sum_m[64] = {0}, sum_f[64] = {0};
    uint64_t hit_m[64] = {0}, hit_f[64] = {0}, total = 0;

    /* CRITICAL: probe exactly ONE line per body-run. Probing two lines back-to-back lets the first
     * probe's miss trigger a forward prefetch that pollutes the second line's measurement. */
    for (uint64_t r = 0; r < n_reps; ++r) {
        for (int page = 0; page < 2; ++page) {
            uint8_t* region = page ? faultp : mainp;
            for (unsigned c = 0; c < NSETS; ++c) {
                unsigned s = bitrev6(c);
                for (unsigned off = 0; off < SANDBOX_SIZE; off += LINE) { flush_line(sandbox + off); }
                __asm__ __volatile__("dsb sy\nisb\n" ::: "memory");
                call_body(code, base, regs);
                __asm__ __volatile__("dsb sy\nisb\n" ::: "memory");
                uint64_t t = time_load(region + s * LINE);
                if (page) { sum_f[s] += t; if (t < thresh) { hit_f[s]++; } }
                else      { sum_m[s] += t; if (t < thresh) { hit_m[s]++; } }
            }
        }
        total++;
    }

    /* find the global-minimum mean latency across both pages = the resident line */
    int min_set = -1, min_page = 0; double min_lat = 1e18;
    for (int b = 0; b < 64; ++b) {
        if (sum_m[b] / total < min_lat) { min_lat = sum_m[b] / total; min_set = b; min_page = 0; }
        if (sum_f[b] / total < min_lat) { min_lat = sum_f[b] / total; min_set = b; min_page = 1; }
    }

    printf("\n=== per-set MEAN reload latency (ns) over %lu runs  [set 63 ... set 0] ===\n",
           (unsigned long)total);
    printf("set |  main   faulty  | main-hit%%  faulty-hit%%\n");
    for (int b = 63; b >= 0; --b) {
        printf(" %2d | %5.1f  %5.1f   |  %6.1f    %6.1f%s\n", b, sum_m[b] / total, sum_f[b] / total,
               100.0 * hit_m[b] / total, 100.0 * hit_f[b] / total,
               (b == min_set) ? "   <== lowest latency (RESIDENT)" : "");
    }
    printf("\n[verdict] lowest mean latency = %.1f ns at set %d on the %s page = the resident line\n",
           min_lat, min_set, min_page ? "faulty" : "main");
    return 0;
}
