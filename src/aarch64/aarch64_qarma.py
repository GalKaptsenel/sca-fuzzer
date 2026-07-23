"""Architected ARM QARMA3/QARMA5 pointer-auth, ported from QEMU target/arm/tcg/pauth_helper.c.

Bit-exact with real hardware for the architected algorithm (the same port as the CE's qarma.c). Used
by the sealer to bake signatures the device will authenticate: the host reproduces the runner's PAC
offline instead of signing on local hardware (which may use a different algorithm).
"""
from typing import NamedTuple, Sequence

M = (1 << 64) - 1


class PacProfile(NamedTuple):
    """iterations = 2 (QARMA3) or 4 (QARMA5); tsz = 64 - VA_size; tbi = top-byte-ignore;
    pauth2 = FEAT_PAuth2 (APA/APA3 >= 3)."""
    iterations: int
    tsz: int
    tbi: int
    pauth2: bool


_VERSION_ITERATIONS = {3: 2, 5: 4}


def profile(qarma_version: int, va_size: int, tbi: bool, pauth2: bool) -> PacProfile:
    if qarma_version not in _VERSION_ITERATIONS:
        raise ValueError(f"unsupported QARMA version {qarma_version} (expected 3 or 5)")
    return PacProfile(_VERSION_ITERATIONS[qarma_version], 64 - va_size, int(tbi), pauth2)


def _ext(v, s, l): return (v >> s) & ((1 << l) - 1)
def _sext(v, s, l):
    x = (v >> s) & ((1 << l) - 1)
    return x - (1 << l) if x & (1 << (l - 1)) else x
def _dep(v, s, l, f):
    m = ((1 << l) - 1) << s
    return (v & ~m & M) | ((f << s) & m)
def _mask(s, l): return (((1 << l) - 1) << s) & M

_SUB = [0xb,0x6,0x8,0xf,0xc,0x0,0x9,0xe,0x3,0x7,0x4,0x5,0xd,0x2,0x1,0xa]
_ISUB = [0x5,0xe,0xd,0x8,0xa,0xb,0x1,0x9,0x2,0x6,0xf,0x0,0x4,0xc,0x7,0x3]
_RC = [0x0000000000000000,0x13198A2E03707344,0xA4093822299F31D0,0x082EFA98EC4E6C89,0x452821E638D01377]
_ALPHA = 0xC0AC29B7C97C50DD

def _sub(i):  return sum(_SUB[(i >> b) & 0xf] << b for b in range(0, 64, 4))
def _isub(i): return sum(_ISUB[(i >> b) & 0xf] << b for b in range(0, 64, 4))

def _rot(cell, n):
    cell &= 0xf; cell |= cell << 4
    return (cell >> (4 - n)) & 0xf

def _shuf(i):
    idx = [52,24,44,0,28,48,4,40,32,12,56,20,8,36,16,60]
    return sum(_ext(i, idx[k], 4) << (4 * k) for k in range(16))
def _ishuf(i):
    idx = [12,24,48,36,56,44,4,16,32,52,28,8,20,0,40,60]
    return sum(_ext(i, idx[k], 4) << (4 * k) for k in range(16))

def _mult(i):
    o = 0
    for b in range(0, 16, 4):
        i0, i4, i8, ic = _ext(i,b,4), _ext(i,b+16,4), _ext(i,b+32,4), _ext(i,b+48,4)
        t0 = _rot(i8,1) ^ _rot(i4,2) ^ _rot(i0,1)
        t1 = _rot(ic,1) ^ _rot(i4,1) ^ _rot(i0,2)
        t2 = _rot(ic,2) ^ _rot(i8,1) ^ _rot(i0,1)
        t3 = _rot(ic,1) ^ _rot(i8,2) ^ _rot(i4,1)
        o |= (t3 << b) | (t2 << (b+16)) | (t1 << (b+32)) | (t0 << (b+48))
    return o

def _trot(c):  return (c >> 1) | (((c ^ (c >> 1)) & 1) << 3)
def _tirot(c): return ((c << 1) & 0xf) | ((c & 1) ^ (c >> 3))
def _tshuf(i):
    r = [(16,0),(20,0),(24,1),(28,0),(44,1),(8,0),(12,0),(32,1),
         (48,0),(52,0),(56,0),(60,1),(0,1),(4,0),(40,1),(36,1)]
    return sum((_trot(_ext(i,src,4)) if rot else _ext(i,src,4)) << (4*k) for k,(src,rot) in enumerate(r))
def _tishuf(i):
    r = [(48,1),(52,0),(20,0),(24,0),(0,0),(4,0),(8,1),(12,0),
         (28,1),(60,1),(56,1),(16,1),(32,0),(36,0),(40,0),(44,1)]
    return sum((_tirot(_ext(i,src,4)) if rot else _ext(i,src,4)) << (4*k) for k,(src,rot) in enumerate(r))


def computepac(data: int, modifier: int, key_lo: int, key_hi: int, iterations: int) -> int:
    """The raw QARMA MAC (before pointer-field insertion). key0 = key_hi, key1 = key_lo."""
    key0, key1 = key_hi, key_lo
    modk0 = ((key0 << 63) | ((key0 >> 1) ^ (key0 >> 63))) & M
    rmod, w = modifier, data ^ key0
    for i in range(iterations + 1):
        w ^= key1 ^ rmod
        w ^= _RC[i]
        if i > 0:
            w = _mult(_shuf(w))
        w = _sub(w)
        rmod = _tshuf(rmod)
    w ^= modk0 ^ rmod
    w = _mult(_shuf(w)); w = _sub(w); w = _mult(_shuf(w))
    w ^= key1
    w = _ishuf(w); w = _isub(w); w = _mult(w); w = _ishuf(w)
    w ^= key0 ^ rmod
    for i in range(iterations + 1):
        w = _isub(w)
        if i < iterations:
            w = _ishuf(_mult(w))
        rmod = _tishuf(rmod)
        w ^= _RC[iterations - i]
        w ^= key1 ^ rmod
        w ^= _ALPHA
    return (w ^ modk0) & M


def addpac(ptr: int, modifier: int, key_lo: int, key_hi: int, p: PacProfile) -> int:
    """Sign a pointer: insert the PAC into the field bits per `p` (the AddPAC pseudocode)."""
    ext = _sext(ptr, 55, 1) if p.tbi else _sext(ptr, 63, 1)
    top_bit = 64 - (8 if p.tbi else 0)
    bot_bit = 64 - p.tsz
    ext_ptr = _dep(ptr, bot_bit, top_bit - bot_bit, ext & M)
    pac = computepac(ext_ptr, modifier, key_lo, key_hi, p.iterations)
    test = _sext(ptr, bot_bit, top_bit - bot_bit)
    if test not in (0, -1) and not p.pauth2:
        pac ^= _mask(top_bit - 2, 1)
    if p.pauth2:
        pac ^= ptr
    if p.tbi:
        ptr = ptr & (~_mask(bot_bit, 55 - bot_bit + 1) & M)
        pac &= _mask(bot_bit, 54 - bot_bit + 1)
    else:
        ptr = ptr & _mask(0, bot_bit)
        pac &= ~(_mask(55, 1) | _mask(0, bot_bit)) & M
    return (pac | ((ext & M) & _mask(55, 1)) | ptr) & M


def strip(ptr: int, p: PacProfile) -> int:
    """Recover the canonical pointer (XPAC)."""
    mask = _mask(64 - p.tsz, (64 - (8 if p.tbi else 0)) - (64 - p.tsz))
    return (ptr | mask) if _ext(ptr, 55, 1) else (ptr & ~mask)


# mnemonic -> index of {lo,hi} in a 10-word PAC key set (apia,apib,apda,apdb,apga).
_KEY_WORD = {"pacia": 0, "paciza": 0, "pacib": 2, "pacizb": 2,
             "pacda": 4, "pacdza": 4, "pacdb": 6, "pacdzb": 6, "pacga": 8}


def sign(ptr: int, ctx: int, mnemonic: str, keys: Sequence[int], p: PacProfile) -> int:
    """Sign `ptr` with the key `mnemonic` selects, reproducing the runner's signed pointer."""
    w = _KEY_WORD[mnemonic.lower()]
    return addpac(ptr, ctx, keys[w], keys[w + 1], p)
