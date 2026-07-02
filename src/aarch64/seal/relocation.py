"""General relocation table for sealed test-case variants.

On the PAC/NI hardware path every input produces genuine + decoy variants that are
byte-identical to a shared skeleton *except* at the sealing slots — the MOVK that
carries the per-input signature and the AUT*/XPAC that follows it. Re-running the
external assembler (asm_to_bytes -> as + objcopy, three process spawns) for each
variant is pure overhead, since only a few 4-byte words change.

A relocation is `(offset, type, value)`: for now the only type is WORD32, which
overwrites the 4-byte little-endian word at `offset`. A variant's machine code is
then `apply_relocations(skeleton_bytes, relocs)` — no assembler involved. The
skeleton and the harvested per-slot words come from assembling a couple of
reference fills once per test case; the varying MOVK word is the harvested MOVK
with its 16-bit immediate field rewritten to the signature (see set_movk_imm16).
"""
from enum import Enum
from typing import List, NamedTuple


NOP_WORD = 0xD503201F  # AArch64 NOP encoding


class RelocType(Enum):
    WORD32 = 1  # overwrite the 4-byte little-endian word at `offset` with `value`


class Relocation(NamedTuple):
    offset: int
    value: int
    rtype: RelocType = RelocType.WORD32


def apply_relocations(base: bytes, relocs: List[Relocation]) -> bytes:
    """Return `base` with every relocation applied. `base` is never mutated."""
    out = bytearray(base)
    for r in relocs:
        if r.rtype is not RelocType.WORD32:
            raise ValueError(f"unsupported relocation type {r.rtype}")
        if r.offset < 0 or r.offset + 4 > len(out):
            raise ValueError(f"relocation offset {r.offset} out of range (len {len(out)})")
        out[r.offset:r.offset + 4] = (r.value & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(out)


def read_word32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 4], "little")


# ---- AArch64 MOVK (64-bit) bit surgery -----------------------------------------
# Encoding: sf(1)=1 opc(11)=11 100101 hw(2) imm16(16) Rd(5).
# Bits [31:23] == 0b1_11_100101 == 0x1E5 identify a 64-bit MOVK; imm16 is [20:5].
_MOVK_OPC = 0x1E5
_MOVK_IMM_SHIFT = 5
_MOVK_IMM_MASK = 0xFFFF


def is_movk64(word: int) -> bool:
    return ((word >> 23) & 0x1FF) == _MOVK_OPC


def set_movk_imm16(word: int, imm16: int) -> int:
    """Rewrite only the 16-bit immediate field of a 64-bit MOVK word (Rd/hw untouched)."""
    assert is_movk64(word), f"not a 64-bit MOVK word: 0x{word:08x}"
    cleared = word & ~(_MOVK_IMM_MASK << _MOVK_IMM_SHIFT)
    return cleared | ((imm16 & _MOVK_IMM_MASK) << _MOVK_IMM_SHIFT)


def get_movk_imm16(word: int) -> int:
    assert is_movk64(word), f"not a 64-bit MOVK word: 0x{word:08x}"
    return (word >> _MOVK_IMM_SHIFT) & _MOVK_IMM_MASK
