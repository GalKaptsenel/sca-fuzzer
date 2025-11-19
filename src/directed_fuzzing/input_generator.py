from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import random

# module-level default RNG
_default_rng = random.Random()

# Type alias: all value generators accept (params, rnd)
ValueGenerator = Callable[[Optional[Dict[str, Any]], Optional[random.Random]], int]


def zero(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    return 0


def small_range(
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


def boundary_32(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    candidates = [
        0x7FFFFFFF,
        0x80000000,
        0xFFFFFFFF,
        0x00000000,
    ]
    rnd = rnd or _default_rng
    return rnd.choice(candidates)


def aligned_addresses(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    base = int(params.get("base", 0x10000000))
    alignment = int(params.get("alignment", 8))

    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError("alignment must be a positive power of two")

    offset = rnd.randint(0, 4096 // alignment) * alignment
    return base + offset


def random_addr(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    base = int(params.get("base", 0x10000000))
    rng  = int(params.get("rng", 0x01000000))

    return base + rnd.randrange(rng)


def cache_set_conflict(
        params: Optional[Dict[str, Any]] = None,
        rnd: Optional[random.Random] = None
    ) -> int:
    """
    Generates an address mapping to a specific cache set.
    """
    params = params or {}
    rnd = rnd or _default_rng

    set_cnt   = int(params.get("set_cnt", 1024))
    line_size = int(params.get("line_size", 64))

    if line_size <= 0 or (line_size & (line_size - 1)) != 0:
        raise ValueError("line_size must be a power of two")

    # pick a set
    set_idx = rnd.randrange(set_cnt)

    # mask off lower bits covering set+line range
    set_mask = ~((set_cnt * line_size) - 1)

    high_bits = rnd.getrandbits(32) & set_mask

    return high_bits | (set_idx * line_size)


def mte_tagged_addr(
            params: Optional[Dict[str, Any]] = None,
            rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng

    tag = params.get("tag", rnd.randrange(16))
    base = params.get("base", 0x10000000)
    addr = (tag << 56) | (base + rnd.randrange(0x1000))
    return addr


def pac_garble_addr(
            params: Optional[Dict[str, Any]] = None,
            rnd: Optional[random.Random] = None
    ) -> int:
    params = params or {}
    rnd = rnd or _default_rng
    base = params.get("base", 0x10000000)
    addr = base + rnd.getrandbits(20)
    addr ^= (rnd.getrandbits(16) << 48)
    return addr

# --------------------------
# Generator Map
# --------------------------

GEN_MAP: Dict[str, ValueGenerator] = {
    "zero": zero,
    "small_range": small_range,
    "random_64": random_64,
    "random_32": random_32,
    "boundary_32": boundary_32,
    "aligned_addresses": aligned_addresses,
    "random_addr": random_addr,
    "cache_set_conflict": cache_set_conflict,
    "mte_tagged_addr": mte_tagged_addr,
    "pac_garble_addr": pac_garble_addr
}

