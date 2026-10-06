#include "qarma.h"

#include <stdio.h>
#include <stdlib.h>

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

/* S-box per QARMA variant: QARMA5 (iterations 4) uses sigma2 (SUB5/ISUB5); QARMA3 (iterations 2) uses
 * sigma1 (SUB3), which is involutory so its inverse is itself. Wrong box => signatures the CPU rejects. */
static const uint8_t SUB5[16]  = {0xb,0x6,0x8,0xf,0xc,0x0,0x9,0xe,0x3,0x7,0x4,0x5,0xd,0x2,0x1,0xa};
static const uint8_t ISUB5[16] = {0x5,0xe,0xd,0x8,0xa,0xb,0x1,0x9,0x2,0x6,0xf,0x0,0x4,0xc,0x7,0x3};
static const uint8_t SUB3[16]  = {10,13,14,6,15,7,3,5,9,8,0,12,11,1,2,4};
static const uint64_t RC[5] = {
    0x0000000000000000ull, 0x13198A2E03707344ull, 0xA4093822299F31D0ull,
    0x082EFA98EC4E6C89ull, 0x452821E638D01377ull,
};
static const uint64_t ALPHA = 0xC0AC29B7C97C50DDull;

static uint64_t pac_sub(uint64_t i, const uint8_t *sbox)
{
    uint64_t o = 0;
    for (int b = 0; b < 64; b += 4) o |= (uint64_t)sbox[(i >> b) & 0xf] << b;
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
    if (2 != iterations && 4 != iterations) {
        fprintf(stderr, "[CE FATAL] QARMA iterations %d (expected 2 = QARMA3 or 4 = QARMA5)\n", iterations);
        abort();
    }
    const uint8_t *S  = (2 == iterations) ? SUB3 : SUB5;
    const uint8_t *IS = (2 == iterations) ? SUB3 : ISUB5;
    uint64_t modk0 = (key0 << 63) | ((key0 >> 1) ^ (key0 >> 63));
    uint64_t rmod = modifier, w = data ^ key0;

    for (int i = 0; i <= iterations; ++i) {
        w ^= key1 ^ rmod;
        w ^= RC[i];
        if (i > 0) w = pac_mult(cell_shuffle(w));
        w = pac_sub(w, S);
        rmod = tweak_shuffle(rmod);
    }
    w ^= modk0 ^ rmod;
    w = pac_mult(cell_shuffle(w));
    w = pac_sub(w, S);
    w = pac_mult(cell_shuffle(w));
    w ^= key1;
    w = cell_inv_shuffle(w);
    w = pac_sub(w, IS);
    w = pac_mult(w);
    w = cell_inv_shuffle(w);
    w ^= key0 ^ rmod;
    for (int i = 0; i <= iterations; ++i) {
        w = pac_sub(w, IS);
        if (i < iterations) w = cell_inv_shuffle(pac_mult(w));
        rmod = tweak_inv_shuffle(rmod);
        w ^= RC[iterations - i];
        w ^= key1 ^ rmod;
        w ^= ALPHA;
    }
    return w ^ modk0;
}

static bool pauth2(struct pac_profile p)
{
    return p.level >= 3;
}

static bool epac(struct pac_profile p)
{
    return 2 == p.level;
}

static int bit(uint64_t v, int b)
{
    return (int)((v >> b) & 1);
}

/* EffectiveTBI (EL1): the TBI/TBID of the half ptr<55> selects; TBID disables TBI for instr keys. */
static int effective_tbi(uint64_t ptr, struct pac_profile p, int is_instr)
{
    int high = bit(ptr, 55);
    int tbi = high ? p.tbi1 : p.tbi0;
    int tbid = high ? p.tbid1 : p.tbid0;
    return (tbi && !(is_instr && tbid)) ? 1 : 0;
}

/* CalculateBottomPACBit: 64 - TxSZ of the half `half` selects (TTBR1 if 1). */
static int bottom_pac_bit(int half, struct pac_profile p)
{
    return 64 - (half ? p.t1sz : p.t0sz);
}

/* AddPAC's selbit (EL1, two VA ranges): ptr<55> if any half has TBI for this key kind, else ptr<63>;
 * always ptr<55> with FEAT_PAuth2 + FEAT_CONSTPACFIELD. */
static int selbit_of(uint64_t ptr, struct pac_profile p, int is_instr)
{
    int any_tbi;
    if (pauth2(p) && p.constpacfield) {
        return bit(ptr, 55);
    }
    if (is_instr) {
        any_tbi = (p.tbi1 && !p.tbid1) || (p.tbi0 && !p.tbid0);
    } else {
        any_tbi = p.tbi1 || p.tbi0;
    }
    return any_tbi ? bit(ptr, 55) : bit(ptr, 63);
}

uint64_t qarma_addpac(uint64_t ptr, uint64_t modifier,
                      uint64_t key_lo, uint64_t key_hi, struct pac_profile p, int is_instr)
{
    int tbi = effective_tbi(ptr, p, is_instr);
    int top_bit = tbi ? 55 : 63;
    int selbit = selbit_of(ptr, p, is_instr);
    int bottom = bottom_pac_bit(selbit, p);
    uint64_t ext = selbit ? ~0ull : 0ull;
    uint64_t low = ptr & bmask(0, bottom);
    uint64_t ext_ptr = tbi ? ((ptr & bmask(56, 8)) | (ext & bmask(bottom, 56 - bottom)) | low)
                           : ((ext & bmask(bottom, 64 - bottom)) | low);
    uint64_t pac = qarma_computepac(ext_ptr, modifier, key_lo, key_hi, p.iterations);
    uint64_t field = bmask(bottom, 55 - bottom);
    uint64_t unused = field | (tbi ? bmask(56, 8) : 0);   /* spec unusedbits_mask */

    if (0 != (ptr & unused) && unused != (ptr & unused)) {
        if (epac(p)) {
            pac = 0;
        } else if (!pauth2(p)) {
            pac ^= bmask(top_bit - 1, 1);
        }
    }
    if (pauth2(p)) {
        pac ^= ptr;
    }
    if (tbi) {
        return (ptr & bmask(56, 8)) | ((uint64_t)selbit << 55) | (pac & field) | low;
    }
    return (pac & bmask(56, 8)) | ((uint64_t)selbit << 55) | (pac & field) | low;
}

uint64_t qarma_strip(uint64_t ptr, struct pac_profile p, int is_instr)
{
    int half = bit(ptr, 55);
    int bottom = bottom_pac_bit(half, p);
    int top = effective_tbi(ptr, p, is_instr) ? 56 : 64;
    uint64_t mask = bmask(bottom, top - bottom);
    return half ? (ptr | mask) : (ptr & ~mask);
}
