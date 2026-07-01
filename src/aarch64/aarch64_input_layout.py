"""
File: AArch64 input-buffer layout.
  - NZCVScheme: per-flag NZCV encoding of input register slot 6
  - PSTATE reconstruction of that slot before execution
  - Mapping of register/flag operands to the input byte offsets they read/write

The input register region is laid out as 8-byte slots: slot i = x{i} (i < 6),
slot 6 = NZCV flags (per-flag encoding, see NZCVScheme), slot 7 = sp.
"""
from typing import List

from ..interfaces import InputFragment

# Byte offset of the GPR register region within an input fragment. Read straight from
# the InputFragment dtype (drift-proof) rather than re-summing the memory-area sizes.
REGISTER_REGION_OFFSET = InputFragment.fields['gpr'][1]


class NZCVScheme:
    """Single source of truth for per-flag NZCV encoding in the input register slot.

    Each of the 4 flags occupies bit 0 of its own byte within slot 6 of the GPR
    register region.  Bytes 4-7 of the slot mirror bytes 0-3 (standard Revizor
    duplication pattern).  Reconstruction to PSTATE is done just before execution.

    Full 4-way byte-granularity taint separation:
        N → byte 48, Z → byte 49, C → byte 50, V → byte 51.
    """
    SLOT_IDX: int = 6
    SLOT_BASE_BYTE: int = SLOT_IDX * 8  # = 48, byte offset in register region

    # flag → (byte_offset_within_slot, PSTATE_bit)
    _LAYOUT: dict = {
        'n': (0, 31),
        'z': (1, 30),
        'c': (2, 29),
        'v': (3, 28),
    }

    @classmethod
    def input_byte(cls, flag: str) -> int:
        """Return the register-region byte offset for the given flag (for taint tracking)."""
        return cls.SLOT_BASE_BYTE + cls._LAYOUT[flag.lower()][0]

    @classmethod
    def to_pstate(cls, raw_slot: int) -> int:
        """Convert raw per-flag slot value (bit 0 of each byte) to ARM PSTATE format."""
        pstate = 0
        for byte_off, pstate_bit in cls._LAYOUT.values():
            pstate |= ((raw_slot >> (byte_off * 8)) & 1) << pstate_bit
        return pstate

    @classmethod
    def make_random(cls, rng) -> int:
        """Generate a random NZCV slot: bit 0 of bytes 0-3 is random, bytes 4-7 mirror."""
        val = 0
        for byte_off, _ in cls._LAYOUT.values():
            val |= int(rng.integers(0, 2)) << (byte_off * 8)
        return val | (val << 32)

    @classmethod
    def flag_names(cls):
        return cls._LAYOUT.keys()


# The 4 flags must map to 4 distinct bytes / PSTATE bits, and the flags slot must fit in the GPR
# register region -- otherwise the encoding silently collides with a register slot.
assert len(NZCVScheme._LAYOUT) == 4
assert len({b for b, _ in NZCVScheme._LAYOUT.values()}) == 4
assert len({p for _, p in NZCVScheme._LAYOUT.values()}) == 4
assert NZCVScheme.SLOT_BASE_BYTE + 4 <= InputFragment.fields['gpr'][0].itemsize


def _reconstruct_pstate(view: memoryview) -> None:
    """Convert the per-flag NZCV encoding in the flags slot to ARM PSTATE format."""
    view[NZCVScheme.SLOT_IDX] = NZCVScheme.to_pstate(int(view[NZCVScheme.SLOT_IDX]))


def _input_bytes_with_pstate(inp) -> bytes:
    """Return inp.tobytes() with the flags slot converted from per-flag to PSTATE format."""
    data = bytearray(inp.tobytes())
    _reconstruct_pstate(memoryview(data)[REGISTER_REGION_OFFSET:].cast('Q'))
    return bytes(data)


def map_register_to_offsets(register: str) -> List[int]:
    """Input byte offsets of the register's 8-byte slot, or the single NZCV flag byte."""
    reg = register.lower()
    if reg in ("fp", "lr", "xzr", "wzr"):
        return []
    # xN and wN are the same architectural register (wN = low 32 bits of xN); both map to
    # input slot N. A read through wN still reads xN's input bytes, so the taint must
    # preserve the whole slot — otherwise boosting mutates it and the trace diverges.
    if reg[:1] in ("x", "w") and reg[1:].isdigit():
        n = int(reg[1:])
        if n < 0 or n > 5:
            return []
    elif reg in NZCVScheme.flag_names():
        return [NZCVScheme.input_byte(reg)]
    elif reg == "sp":
        n = 7
    else:
        return []
    return list(range(n * 8, n * 8 + 8))
