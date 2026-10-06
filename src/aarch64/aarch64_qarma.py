"""Architected ARM QARMA3/QARMA5 pointer-auth: ComputePAC ported from QEMU, AddPAC/Strip per the ARM ARM
shared pseudocode (aarch64/functions/pac).

Bit-exact with real hardware for the architected algorithm (the same model as the CE's qarma.c). Used
by the sealer to bake signatures the device will authenticate: the host reproduces the runner's PAC
offline instead of signing on local hardware (which may use a different algorithm).
"""
from typing import NamedTuple, Sequence

M = (1 << 64) - 1


class PacProfile(NamedTuple):
    """The EL1 PAC state AddPAC/Strip/ComputePAC read. iterations / generic_iterations = 2 (QARMA3) or 4
    (QARMA5) for address / generic (PACGA) auth, generic 0 = no modeled generic algorithm; level =
    the APA/APA3 feature value (1 PAuth, 2 EPAC, 3 PAuth2, 4 FPAC, 5 FPACCOMBINE); t0sz/t1sz, tbi*,
    tbid* = TCR_EL1 fields; constpacfield = FEAT_CONSTPACFIELD."""
    iterations: int
    generic_iterations: int
    level: int
    t0sz: int
    t1sz: int
    tbi0: int
    tbi1: int
    tbid0: int
    tbid1: int
    constpacfield: bool

    @property
    def pauth2(self) -> bool:
        return self.level >= 3

    @property
    def epac(self) -> bool:
        return self.level == 2


_VERSION_ITERATIONS = {3: 2, 5: 4}
_LEVELS = range(1, 6)
# TxSZ range modeled without PACEffectiveTxSZ clamping: no FEAT_LVA (min 16), no FEAT_TTST (max 39).
_TXSZ_MIN, _TXSZ_MAX = 16, 39


def profile(qarma_version: int, generic_qarma_version: int, auth_level: int, va_size0: int,
            va_size1: int, tbi0: bool, tbi1: bool, tbid0: bool, tbid1: bool, constpacfield: bool,
            mtx0: bool, mtx1: bool) -> PacProfile:
    """Validate and build a profile; generic_qarma_version 0 = no modeled generic (PACGA) algorithm."""
    if qarma_version not in _VERSION_ITERATIONS:
        raise ValueError(f"unsupported address-auth QARMA version {qarma_version} (expected 3 or 5)")
    if generic_qarma_version not in (0, *_VERSION_ITERATIONS):
        raise ValueError(f"unsupported generic QARMA version {generic_qarma_version} (expected 0, 3 or 5)")
    if auth_level not in _LEVELS:
        raise ValueError(f"unsupported address-auth level {auth_level} (expected 1..5)")
    for name, va in (("va_size0", va_size0), ("va_size1", va_size1)):
        if not _TXSZ_MIN <= 64 - va <= _TXSZ_MAX:
            raise ValueError(f"{name}={va}: TxSZ outside [{_TXSZ_MIN},{_TXSZ_MAX}] (clamping/LVA not modeled)")
    if mtx0 or mtx1:
        raise ValueError("TCR_EL1.MTX set: FEAT_MTE4 canonical tag checking is not modeled")
    return PacProfile(_VERSION_ITERATIONS[qarma_version], _VERSION_ITERATIONS.get(generic_qarma_version, 0),
                      auth_level, 64 - va_size0, 64 - va_size1, int(bool(tbi0)), int(bool(tbi1)),
                      int(bool(tbid0)), int(bool(tbid1)), bool(constpacfield))


class PacRegisters(NamedTuple):
    """The PAC-relevant fields of TCR_EL1 / ID_AA64ISAR1_EL1 / ID_AA64ISAR2_EL1 (ARM ARM bit positions)."""
    qarma_version: int
    generic_qarma_version: int
    auth_level: int
    va_size0: int
    va_size1: int
    tbi0: bool
    tbi1: bool
    tbid0: bool
    tbid1: bool
    constpacfield: bool
    mtx0: bool
    mtx1: bool


def decode_registers(tcr: int, isar1: int, isar2: int) -> PacRegisters:
    """Decode the raw registers. Address auth must be exactly one architected algorithm (APA = QARMA5 or
    APA3 = QARMA3; API = IMPDEF is not modeled); generic auth at most one (GPA, GPA3; GPI not modeled)."""
    f = lambda v, lo: (v >> lo) & 0xf    # noqa: E731  (4-bit ID field)
    apa, api, gpa, gpi = f(isar1, 4), f(isar1, 8), f(isar1, 24), f(isar1, 28)
    gpa3, apa3, pac_frac = f(isar2, 8), f(isar2, 12), f(isar2, 24)
    if api:
        raise ValueError(f"ID_AA64ISAR1_EL1.API={api}: IMPDEF address-auth algorithm is not modeled")
    if bool(apa) == bool(apa3):
        raise ValueError(f"expected exactly one address-auth algorithm, got APA={apa} APA3={apa3}")
    if (gpa and gpa3) or (gpi and (gpa or gpa3)):
        raise ValueError(f"conflicting generic-auth algorithms GPA={gpa} GPA3={gpa3} GPI={gpi}")
    for name, v in (("GPA", gpa), ("GPA3", gpa3), ("GPI", gpi)):
        if v > 1:
            raise ValueError(f"{name}={v}: reserved generic-auth value")
    if pac_frac > 1:
        raise ValueError(f"ID_AA64ISAR2_EL1.PAC_frac={pac_frac}: reserved")
    bit = lambda b: bool((tcr >> b) & 1)    # noqa: E731
    return PacRegisters(qarma_version=5 if apa else 3,
                        generic_qarma_version=5 if gpa else (3 if gpa3 else 0),
                        auth_level=apa or apa3,
                        va_size0=64 - (tcr & 0x3f), va_size1=64 - ((tcr >> 16) & 0x3f),
                        tbi0=bit(37), tbi1=bit(38), tbid0=bit(51), tbid1=bit(52),
                        constpacfield=pac_frac == 1, mtx0=bit(60), mtx1=bit(61))


def _ext(v, s, l): return (v >> s) & ((1 << l) - 1)
def _sext(v, s, l):
    x = (v >> s) & ((1 << l) - 1)
    return x - (1 << l) if x & (1 << (l - 1)) else x
def _dep(v, s, l, f):
    m = ((1 << l) - 1) << s
    return (v & ~m & M) | ((f << s) & m)
def _mask(s, l): return (((1 << l) - 1) << s) & M

# S-box per QARMA variant, keyed by `iterations`: QARMA5 (4) uses sigma2, QARMA3 (2) uses sigma1.
# Using the wrong box silently produces signatures the CPU's AUT* rejects (FPAC).
_SBOX = {
    4: [0xb,0x6,0x8,0xf,0xc,0x0,0x9,0xe,0x3,0x7,0x4,0x5,0xd,0x2,0x1,0xa],
    2: [10,13,14,6,15,7,3,5,9,8,0,12,11,1,2,4],
}
def _invert(s):
    o = [0] * 16
    for i, v in enumerate(s):
        o[v] = i
    return o
_ISBOX = {k: _invert(v) for k, v in _SBOX.items()}
_RC = [0x0000000000000000,0x13198A2E03707344,0xA4093822299F31D0,0x082EFA98EC4E6C89,0x452821E638D01377]
_ALPHA = 0xC0AC29B7C97C50DD

def _sub(i, S):  return sum(S[(i >> b) & 0xf] << b for b in range(0, 64, 4))
def _isub(i, S): return sum(S[(i >> b) & 0xf] << b for b in range(0, 64, 4))

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
    S, IS = _SBOX[iterations], _ISBOX[iterations]
    modk0 = ((key0 << 63) | ((key0 >> 1) ^ (key0 >> 63))) & M
    rmod, w = modifier, data ^ key0
    for i in range(iterations + 1):
        w ^= key1 ^ rmod
        w ^= _RC[i]
        if i > 0:
            w = _mult(_shuf(w))
        w = _sub(w, S)
        rmod = _tshuf(rmod)
    w ^= modk0 ^ rmod
    w = _mult(_shuf(w)); w = _sub(w, S); w = _mult(_shuf(w))
    w ^= key1
    w = _ishuf(w); w = _isub(w, IS); w = _mult(w); w = _ishuf(w)
    w ^= key0 ^ rmod
    for i in range(iterations + 1):
        w = _isub(w, IS)
        if i < iterations:
            w = _ishuf(_mult(w))
        rmod = _tishuf(rmod)
        w ^= _RC[iterations - i]
        w ^= key1 ^ rmod
        w ^= _ALPHA
    return (w ^ modk0) & M


def _bit(v: int, b: int) -> int:
    return (v >> b) & 1


def effective_tbi(ptr: int, p: PacProfile, is_instr: bool) -> int:
    """EffectiveTBI (EL1): the TBI/TBID of the half ptr<55> selects; TBID disables TBI for instr keys."""
    high = _bit(ptr, 55)
    tbi = p.tbi1 if high else p.tbi0
    tbid = p.tbid1 if high else p.tbid0
    return int(bool(tbi) and not (is_instr and tbid))


def _bottom_pac_bit(half: int, p: PacProfile) -> int:
    """CalculateBottomPACBit: 64 - TxSZ of the half `half` selects (TTBR1 if 1)."""
    return 64 - (p.t1sz if half else p.t0sz)


def _selbit(ptr: int, p: PacProfile, is_instr: bool) -> int:
    """AddPAC's selbit (EL1, two VA ranges): ptr<55> if any half has TBI for this key kind, else ptr<63>;
    always ptr<55> with FEAT_PAuth2 + FEAT_CONSTPACFIELD."""
    if p.pauth2 and p.constpacfield:
        return _bit(ptr, 55)
    if is_instr:
        any_tbi = (p.tbi1 and not p.tbid1) or (p.tbi0 and not p.tbid0)
    else:
        any_tbi = p.tbi1 or p.tbi0
    return _bit(ptr, 55) if any_tbi else _bit(ptr, 63)


def pac_field_mask(ptr: int, p: PacProfile, is_instr: bool) -> int:
    """The bits AddPAC writes the PAC into for `ptr`: [54:bottom], plus [63:56] when TBI is off."""
    bottom = _bottom_pac_bit(_selbit(ptr, p, is_instr), p)
    field = _mask(bottom, 55 - bottom)
    return field if effective_tbi(ptr, p, is_instr) else field | _mask(56, 8)


def addpac(ptr: int, modifier: int, key_lo: int, key_hi: int, p: PacProfile,
           is_instr: bool) -> int:
    """AddPAC: insert the PAC of `ptr` (with good extension bits) into its PAC field."""
    tbi = effective_tbi(ptr, p, is_instr)
    top_bit = 55 if tbi else 63
    selbit = _selbit(ptr, p, is_instr)
    bottom = _bottom_pac_bit(selbit, p)
    ext = M if selbit else 0
    low = ptr & _mask(0, bottom)
    if tbi:
        ext_ptr = (ptr & _mask(56, 8)) | (ext & _mask(bottom, 56 - bottom)) | low
    else:
        ext_ptr = (ext & _mask(bottom, 64 - bottom)) | low
    pac = computepac(ext_ptr, modifier, key_lo, key_hi, p.iterations)
    unused = _mask(bottom, 55 - bottom) | (_mask(56, 8) if tbi else 0)   # spec unusedbits_mask
    if (ptr & unused) not in (0, unused):
        if p.epac:
            pac = 0
        elif not p.pauth2:
            pac ^= 1 << (top_bit - 1)
    field = _mask(bottom, 55 - bottom)
    if p.pauth2:
        pac ^= ptr
    if tbi:
        return (ptr & _mask(56, 8)) | (selbit << 55) | (pac & field) | low
    return (pac & _mask(56, 8)) | (selbit << 55) | (pac & field) | low


def strip(ptr: int, p: PacProfile, is_instr: bool) -> int:
    """Strip (XPAC): replace the PAC field with copies of ptr<55>."""
    half = _bit(ptr, 55)
    bottom = _bottom_pac_bit(half, p)
    top = 56 if effective_tbi(ptr, p, is_instr) else 64
    mask = _mask(bottom, top - bottom)
    return (ptr | mask) if half else (ptr & ~mask & M)


# AddPAC mnemonic -> index of {lo,hi} in a 10-word PAC key set (apia,apib,apda,apdb,apga).
_KEY_WORD = {"pacia": 0, "paciza": 0, "pacib": 2, "pacizb": 2,
             "pacda": 4, "pacdza": 4, "pacdb": 6, "pacdzb": 6}


def is_instr_key(mnemonic: str) -> bool:
    """Whether an AddPAC mnemonic uses an instruction key (IA/IB) rather than a data key (DA/DB)."""
    mn = mnemonic.lower()
    if mn not in _KEY_WORD:
        raise ValueError(f"not an AddPAC mnemonic: {mnemonic}")
    return mn.startswith("paci")


def sign(ptr: int, ctx: int, mnemonic: str, keys: Sequence[int], p: PacProfile) -> int:
    """Sign `ptr` with the key `mnemonic` selects, reproducing the runner's signed pointer."""
    w = _KEY_WORD[mnemonic.lower()]
    return addpac(ptr, ctx, keys[w], keys[w + 1], p, is_instr_key(mnemonic))
