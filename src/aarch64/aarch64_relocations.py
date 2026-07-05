"""AArch64 machine-code manipulation: rewrite instruction-word immediate fields, and splice fixed
words into a pre-assembled base (relocation)."""
from enum import Enum
from typing import List, NamedTuple


NOP_WORD = 0xD503201F


# ---- instruction-word immediate-field surgery (the encode side of aarch64_disasm.py) ------------

def get_imm_field(word: int, shift: int, width: int) -> int:
    return (word >> shift) & ((1 << width) - 1)


def set_imm_field(word: int, shift: int, width: int, value: int) -> int:
    mask = (1 << width) - 1
    if value & ~mask:
        raise ValueError(f"value 0x{value:x} does not fit {width} bits")
    return (word & ~(mask << shift)) | (value << shift)


# 64-bit MOVK: bits[31:23] == 0x1E5 (opc=11 separates it from MOVZ/MOVN); imm16 at [20:5].
_MOVK64_ID = 0x1E5


def is_movk64(word: int) -> bool:
    return ((word >> 23) & 0x1FF) == _MOVK64_ID


def set_movk_imm16(word: int, imm16: int) -> int:
    if not is_movk64(word):
        raise ValueError(f"not a 64-bit MOVK word: 0x{word:08x}")
    return set_imm_field(word, 5, 16, imm16)


def get_movk_imm16(word: int) -> int:
    if not is_movk64(word):
        raise ValueError(f"not a 64-bit MOVK word: 0x{word:08x}")
    return get_imm_field(word, 5, 16)


def xpac_word(data_key: bool, rd: int) -> int:
    """XPACD Xd (data key) or XPACI Xd (instruction key) — the seal's arch-safe strip op."""
    return (0xDAC147E0 if data_key else 0xDAC143E0) | (rd & 0x1F)


def addg_word(rd: int, tag_delta: int) -> int:
    """ADDG Xd, Xd, #0, #tag_delta — the seal's MTE retag (uimm6=0, Xn=Xd=rd)."""
    return 0x91800000 | ((tag_delta & 0xF) << 10) | ((rd & 0x1F) << 5) | (rd & 0x1F)


def movk_word(rd: int, imm16: int, shift: int) -> int:
    """MOVK Xd, #imm16, LSL #shift (shift in {0, 16, 32, 48})."""
    return 0xF2800000 | ((shift // 16) << 21) | ((imm16 & 0xFFFF) << 5) | (rd & 0x1F)


# AUT* base opcodes (Rd=Rn=0); the Z-variants bake Rn=31, so OR-ing rn=31 there is a no-op.
_AUT_OPCODES = {
    "autia": 0xDAC11000, "autib": 0xDAC11400, "autda": 0xDAC11800, "autdb": 0xDAC11C00,
    "autiza": 0xDAC133E0, "autizb": 0xDAC137E0, "autdza": 0xDAC13BE0, "autdzb": 0xDAC13FE0,
}


def aut_word(mnemonic: str, rd: int, rn: int) -> int:
    """AUT* over Rd with Rn as context (rn=31 for the Z-variants, which take no context)."""
    return _AUT_OPCODES[mnemonic] | ((rn & 0x1F) << 5) | (rd & 0x1F)


def read_word32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ValueError(f"read offset {offset} out of range (len {len(data)})")
    return int.from_bytes(data[offset:offset + 4], "little")


# ---- relocation: splice fixed little-endian words into a byte skeleton --------------------------

class RelocType(Enum):
    WORD32 = 1


class Relocation(NamedTuple):
    offset: int
    value: int
    rtype: RelocType = RelocType.WORD32


def apply_relocations(base: bytes, relocs: List[Relocation]) -> bytes:
    """`base` with every relocation applied; never mutates `base`. Rejects out-of-range offsets,
    values wider than the slot, and overlapping edits rather than silently corrupting."""
    out = bytearray(base)
    written: List[int] = []
    for r in relocs:
        if r.rtype is not RelocType.WORD32:
            raise ValueError(f"unsupported relocation type {r.rtype}")
        if r.offset < 0 or r.offset + 4 > len(out):
            raise ValueError(f"relocation offset {r.offset} out of range (len {len(out)})")
        if r.value < 0 or r.value > 0xFFFFFFFF:
            raise ValueError(f"relocation value 0x{r.value:x} does not fit 32 bits")
        if any(abs(r.offset - w) < 4 for w in written):
            raise ValueError(f"overlapping relocation at offset {r.offset}")
        written.append(r.offset)
        out[r.offset:r.offset + 4] = r.value.to_bytes(4, "little")
    return bytes(out)
