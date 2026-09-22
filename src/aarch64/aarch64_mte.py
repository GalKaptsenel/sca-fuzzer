"""MTE allocation-tag modelling used while parsing the contract-executor trace.

MteTagState tracks the allocation tag of each memory granule across speculation. It is a stack of
layers indexed by speculation depth: layer 0 is architectural; entering deeper speculation copies
the current top, and unwinding pops it — that pop is the revert of the speculative tag stores. It
starts uniform (the region's initial tag); per-cell initial tags can be pre-seeded for dynamic
tagging.
"""
from typing import Dict, List, Optional, Tuple

from .aarch64_disasm import decode_tag_store

MTE_GRANULE = 16  # bytes covered by one allocation tag
MTE_INITIAL_DEFAULT_TAG = 6

# STG-family memory-tag stores -> number of granules each tags.
_MTE_TAG_STORES = {"stg": 1, "stzg": 1, "st2g": 2, "stz2g": 2}


class MteTagState:
    def __init__(self, default_tag: int):
        self._default = default_tag & 0xF
        self._stack: List[Dict[int, int]] = [{}]  # one tag-override layer per live speculation depth

    @staticmethod
    def granule(addr: int) -> int:
        return (addr & ((1 << 56) - 1)) & ~(MTE_GRANULE - 1)  # drop the tag byte and in-granule offset

    def to_depth(self, nesting: int) -> None:
        """Track the current speculation depth: grow by copying the top (speculation inherits the
        live state), shrink by popping (reverting the deeper levels' speculative stores)."""
        while len(self._stack) <= nesting:
            self._stack.append(dict(self._stack[-1]))
        del self._stack[nesting + 1:]

    def set(self, addr: int, tag: int, n_granules: int = 1) -> None:
        """Tag granules in the current (deepest live) layer."""
        layer, g = self._stack[-1], self.granule(addr)
        for i in range(n_granules):
            layer[g + i * MTE_GRANULE] = tag & 0xF

    def tag_at(self, addr: int) -> int:
        """The tag visible at the current speculation depth (architectural + live speculative stores)."""
        return self._stack[-1].get(self.granule(addr), self._default)


def _reg_value(cpu, name: str) -> int:
    name = name.lower()
    if name == "sp":
        return cpu.sp
    if name.startswith("x") and name[1:].isdigit():
        return cpu.gpr[int(name[1:])]
    return 0


def mte_tag_store_effect(ite) -> Optional[Tuple[int, int, int]]:
    """If ite is an STG-family tag store, return (addr, tag, n_granules); else None. STG writes the
    allocation TAG of the granule its address covers (not data memory); the tag is the logical tag of
    Xt. The CE logs post-hook state, so a writeback form's base register is ALREADY incremented in the
    trace (mte_emulator_hook writes it back before the entry is logged). Recover the access address
    from the addressing mode (imm9<<4, sign-extended) so pre/post-index land on the right granule:
      signed offset (mode 10): no writeback,      addr = base + off
      pre-index     (mode 11): base holds Xn+off,  addr = base        (the writeback target == EA)
      post-index    (mode 01): base holds Xn+off,  addr = base - off  (EA is the pre-writeback Xn)."""
    dec = decode_tag_store(ite.cpu.encoding, ite.cpu.pc)
    if dec is None:
        return None
    mn, xt, base, _ = dec
    enc = ite.cpu.encoding
    imm9 = (enc >> 12) & 0x1FF
    off = (imm9 - 0x200 if imm9 & 0x100 else imm9) * MTE_GRANULE
    mode = (enc >> 10) & 0x3
    base_val = _reg_value(ite.cpu, base)
    if mode == 0b01:
        addr = base_val - off
    elif mode == 0b11:
        addr = base_val
    else:
        addr = base_val + off
    return addr, (_reg_value(ite.cpu, xt) >> 56) & 0xF, _MTE_TAG_STORES[mn]


