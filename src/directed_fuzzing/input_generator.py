from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import random

def require(params, *keys):
    try:
        return tuple(params[k] for k in keys)
    except KeyError as e:
        raise ValueError(f"missing param {e}")

# module-level default RNG
_default_rng = random.Random()

# Type alias: all value generators accept (params, rnd)
ValueGenerator = Callable[[Optional[Dict[str, Any]], Optional[random.Random]], int]

def value_range(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    mn = int(params.get("min", -16))
    mx = int(params.get("max", 16))
    return rnd.randint(mn, mx)


def random_nbits(
        nbits: int,
        rnd: Optional[random.Random] = None
    ) -> int:
    rnd = rnd or _default_rng
    return rnd.getrandbits(nbits)


def random_64(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    return random_nbits(64, rnd)


def random_32(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    return random_nbits(32, rnd)


def random_64_low_entropy(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    bits = random_nbits(16, rnd)
    return (bits << 32) | bits


def random_32_low_entropy(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    bits = random_nbits(16, rnd)
    return bits

def bitmask(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    fixed = params.get("fixed_bits", {})
    free = params.get("free_bits", [])

    v = 0
    for b, val in fixed.items():
        assert b >= 0
        if val:
            v |= (1 << b)

    for b in free:
        assert b >= 0
        if rnd.getrandbits(1):
            v |= (1 << b)

    return v

def whitelist(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    candidates = require(params, 'candidates')
    if not candidates:
        raise ValueError("whitelist requires non-empty 'candidates'")

    return rnd.choice(candidates)

def boundary_32(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    params = {
            'candidates': [
                0x7FFFFFFF,
                0x80000000,
                0xFFFFFFFF,
                0x00000000,
            ]
    }

    return whitelist(params, rnd)


def aligned_addresses(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    base, alignment, rng = require(params, 'base', 'alignment', 'range')
    base = int(base)
    alignment = int(alignment)
    rng = int(rng)

    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError("alignment must be a positive power of two")

    offset = rnd.randint(0, rng // alignment) * alignment
    return base + offset


def random_addr(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    base, rng = require(params, 'base', 'range')
    base = int(base)
    rng = int(rng)

    return base + rnd.randrange(rng)


def cache_set(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    """
    Generates an address mapping to a specific cache set.
    """
    params = params or {}
    rnd = rnd or _default_rng

    set_idx, set_cnt, line_size = require(params, 'set_idx', 'set_cnt', 'line_size')
    set_idx = int(set_idx)
    set_cnt = int(set_cnt)
    line_size = int(line_size)

    if line_size <= 0 or (line_size & (line_size - 1)) != 0:
        raise ValueError("line_size must be a power of two")

    if not (0 <= set_idx < set_cnt):
        raise ValueError(f"set_idx ({set_idx}) is out of the range 0 <= set_idx < set_cnt ({set_cnt})")


    # mask off lower bits covering set+line range
    set_mask = ~((set_cnt * line_size) - 1)

    high_bits = rnd.getrandbits(64) & set_mask

    return high_bits | (set_idx * line_size)


def mte_tagged_addr(
            params: Optional[Dict[str, Any]] = None,
            rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    tag_loc = 56 
    tag = params.get("tag", rnd.randrange(16))
    tag_mask = ~(0xF << tag_loc)

    return (tag << tag_loc) | (random_addr(params, rnd) & tag_mask)


def pac_garble_addr(
            params: Optional[Dict[str, Any]] = None,
            rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    pac_loc  = int(params.get("pac_loc", 48))
    pac_bits = int(params.get("pac_bits", 16))
    pac = int(params.get("pac", rnd.getrandbits(pac_bits)))

    pac_mask = ~(((1 << pac_bits) - 1) << pac_loc)

    return (pac << pac_loc) | (random_addr(params, rnd) & pac_mask)

def concrete(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    (value,) = require(params, 'value')
    return  int(value)

# --------------------------
# Generator Map
# --------------------------

GEN_MAP: Dict[str, ValueGenerator] = {
    "value_range": value_range,
    "random_64": random_64,
    "random_64_low_entropy": random_64_low_entropy,
    "random_32": random_32,
    "random_32_low_entropy": random_32_low_entropy,
    "bitmask": bitmask,
    "whitelist": whitelist,
    "boundary_32": boundary_32,
    "aligned_addresses": aligned_addresses,
    "random_addr": random_addr,
    "cache_set": cache_set,
    "mte_tagged_addr": mte_tagged_addr,
    "pac_garble_addr": pac_garble_addr,
    "concrete": concrete,
}

