#include "qarma.h"

/* Bit helpers (QEMU semantics). */
static inline uint64_t ext64(uint64_t v, int s, int l) { return (v >> s) & (((uint64_t)1 << l) - 1); }
static inline int64_t sext64(uint64_t v, int s, int l)
{
    uint64_t x = (v >> s) & (((uint64_t)1 << l) - 1);
    return (x & ((uint64_t)1 << (l - 1))) ? (int64_t)(x - ((uint64_t)1 << l)) : (int64_t)x;
}
static inline uint64_t dep64(uint64_t v, int s, int l, uint64_t f)
{
    uint64_t m = (((uint64_t)1 << l) - 1) << s;
    return (v & ~m) | ((f << s) & m);
}
static inline uint64_t bmask(int s, int l) { return (((uint64_t)1 << l) - 1) << s; }

static const uint8_t SUB[16]  = {0xb,0x6,0x8,0xf,0xc,0x0,0x9,0xe,0x3,0x7,0x4,0x5,0xd,0x2,0x1,0xa};
static const uint8_t ISUB[16] = {0x5,0xe,0xd,0x8,0xa,0xb,0x1,0x9,0x2,0x6,0xf,0x0,0x4,0xc,0x7,0x3};
static const uint64_t RC[5] = {
    0x0000000000000000ull, 0x13198A2E03707344ull, 0xA4093822299F31D0ull,
    0x082EFA98EC4E6C89ull, 0x452821E638D01377ull,
};
static const uint64_t ALPHA = 0xC0AC29B7C97C50DDull;

static uint64_t pac_sub(uint64_t i)
{
    uint64_t o = 0;
    for (int b = 0; b < 64; b += 4) o |= (uint64_t)SUB[(i >> b) & 0xf] << b;
    return o;
}
static uint64_t pac_inv_sub(uint64_t i)
{
    uint64_t o = 0;
    for (int b = 0; b < 64; b += 4) o |= (uint64_t)ISUB[(i >> b) & 0xf] << b;
    return o;
}
static int rot_cell(int cell, int n)
{
    cell &= 0xf; cell |= cell << 4;
    return (cell >> (4 - n)) & 0xf;
}

static uint64_t cell_shuffle(uint64_t i)
{
    static const int idx[16] = {52,24,44,0,28,48,4,40,32,12,56,20,8,36,16,60};
    uint64_t o = 0;
    for (int k = 0; k < 16; ++k) o |= ext64(i, idx[k], 4) << (4 * k);
    return o;
}
static uint64_t cell_inv_shuffle(uint64_t i)
{
    static const int idx[16] = {12,24,48,36,56,44,4,16,32,52,28,8,20,0,40,60};
    uint64_t o = 0;
    for (int k = 0; k < 16; ++k) o |= ext64(i, idx[k], 4) << (4 * k);
    return o;
}
static uint64_t pac_mult(uint64_t i)
{
    uint64_t o = 0;
    for (int b = 0; b < 16; b += 4) {
        int i0 = ext64(i,b,4), i4 = ext64(i,b+16,4), i8 = ext64(i,b+32,4), ic = ext64(i,b+48,4);
        int t0 = rot_cell(i8,1) ^ rot_cell(i4,2) ^ rot_cell(i0,1);
        int t1 = rot_cell(ic,1) ^ rot_cell(i4,1) ^ rot_cell(i0,2);
        int t2 = rot_cell(ic,2) ^ rot_cell(i8,1) ^ rot_cell(i0,1);
        int t3 = rot_cell(ic,1) ^ rot_cell(i8,2) ^ rot_cell(i4,1);
        o |= (uint64_t)t3 << b;
        o |= (uint64_t)t2 << (b + 16);
        o |= (uint64_t)t1 << (b + 32);
        o |= (uint64_t)t0 << (b + 48);
    }
    return o;
}

static uint64_t tweak_rot(uint64_t c)     { return (c >> 1) | (((c ^ (c >> 1)) & 1) << 3); }
static uint64_t tweak_inv_rot(uint64_t c) { return ((c << 1) & 0xf) | ((c & 1) ^ (c >> 3)); }

static uint64_t tweak_shuffle(uint64_t i)
{
    static const int src[16] = {16,20,24,28,44,8,12,32,48,52,56,60,0,4,40,36};
    static const int rot[16] = {0,0,1,0,1,0,0,1,0,0,0,1,1,0,1,1};
    uint64_t o = 0;
    for (int k = 0; k < 16; ++k) {
        uint64_t c = ext64(i, src[k], 4);
        o |= (rot[k] ? tweak_rot(c) : c) << (4 * k);
    }
    return o;
}
static uint64_t tweak_inv_shuffle(uint64_t i)
{
    static const int src[16] = {48,52,20,24,0,4,8,12,28,60,56,16,32,36,40,44};
    static const int rot[16] = {1,0,0,0,0,0,1,0,1,1,1,1,0,0,0,1};
    uint64_t o = 0;
    for (int k = 0; k < 16; ++k) {
        uint64_t c = ext64(i, src[k], 4);
        o |= (rot[k] ? tweak_inv_rot(c) : c) << (4 * k);
    }
    return o;
}

uint64_t qarma_computepac(uint64_t data, uint64_t modifier,
                          uint64_t key_lo, uint64_t key_hi, int iterations)
{
    uint64_t key0 = key_hi, key1 = key_lo;
    uint64_t modk0 = (key0 << 63) | ((key0 >> 1) ^ (key0 >> 63));
    uint64_t rmod = modifier, w = data ^ key0;

    for (int i = 0; i <= iterations; ++i) {
        w ^= key1 ^ rmod;
        w ^= RC[i];
        if (i > 0) w = pac_mult(cell_shuffle(w));
        w = pac_sub(w);
        rmod = tweak_shuffle(rmod);
    }
    w ^= modk0 ^ rmod;
    w = pac_mult(cell_shuffle(w));
    w = pac_sub(w);
    w = pac_mult(cell_shuffle(w));
    w ^= key1;
    w = cell_inv_shuffle(w);
    w = pac_inv_sub(w);
    w = pac_mult(w);
    w = cell_inv_shuffle(w);
    w ^= key0 ^ rmod;
    for (int i = 0; i <= iterations; ++i) {
        w = pac_inv_sub(w);
        if (i < iterations) w = cell_inv_shuffle(pac_mult(w));
        rmod = tweak_inv_shuffle(rmod);
        w ^= RC[iterations - i];
        w ^= key1 ^ rmod;
        w ^= ALPHA;
    }
    return w ^ modk0;
}

/* TBI of the pointer's VA half (bit 55 selects TTBR0/low vs TTBR1/high). */
static int profile_tbi(uint64_t ptr, struct pac_profile p)
{
    return ((ptr >> 55) & 1) ? p.tbi1 : p.tbi0;
}

uint64_t qarma_addpac(uint64_t ptr, uint64_t modifier,
                      uint64_t key_lo, uint64_t key_hi, struct pac_profile p)
{
    int tbi = profile_tbi(ptr, p);
    int64_t ext = tbi ? sext64(ptr, 55, 1) : sext64(ptr, 63, 1);
    int top_bit = 64 - (tbi ? 8 : 0);
    int bot_bit = 64 - p.tsz;
    uint64_t ext_ptr = dep64(ptr, bot_bit, top_bit - bot_bit, (uint64_t)ext);
    uint64_t pac = qarma_computepac(ext_ptr, modifier, key_lo, key_hi, p.iterations);

    int64_t test = sext64(ptr, bot_bit, top_bit - bot_bit);
    if (test != 0 && test != -1 && !p.pauth2) {
        pac ^= bmask(top_bit - 2, 1);
    }
    if (p.pauth2) {
        pac ^= ptr;
    }
    if (tbi) {
        ptr &= ~bmask(bot_bit, 55 - bot_bit + 1);
        pac &= bmask(bot_bit, 54 - bot_bit + 1);
    } else {
        ptr &= bmask(0, bot_bit);
        pac &= ~(bmask(55, 1) | bmask(0, bot_bit));
    }
    return pac | ((uint64_t)ext & bmask(55, 1)) | ptr;
}

uint64_t qarma_strip(uint64_t ptr, struct pac_profile p)
{
    int tbi = profile_tbi(ptr, p);
    int bot_bit = 64 - p.tsz;
    int top_bit = 64 - (tbi ? 8 : 0);
    uint64_t mask = bmask(bot_bit, top_bit - bot_bit);
    return ext64(ptr, 55, 1) ? (ptr | mask) : (ptr & ~mask);
}
